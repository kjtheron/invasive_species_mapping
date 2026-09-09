# Azure setup — chipping on a VM next to the data

Why this exists: `ingest-chips` reads about **2.6 TB** from the Microsoft Planetary Computer to write **61 GB** of chips, a 42:1 ratio. The Planetary Computer Sentinel-2 L2A blobs live in Azure, so a VM in the same cloud removes almost all of that latency encountered with a local download run.

This is a runbook, not a log. Set the variables in section 0 and the commands below work as written.

---

## 0. Variables

Set these once per shell. Everything after this point uses them.

```bash
export RG=cmrv-rg                    # resource group
export LOC=germanywestcentral        # see section 2 before you change this
export SA=<globally-unique-name>     # 3-24 chars, lowercase letters and digits only
export CONTAINER=cmrv
export VM=cmrv-chipper
export VM_USER=<your-linux-username>
export REPO=<your-github-org>/<your-repo>

export SUB=$(az account show --query id -o tsv)
export TENANT=$(az account show --query tenantId -o tsv)
export ME=$(az ad signed-in-user show --query id -o tsv)
```

Check the storage account name is free before you use it:

```bash
az storage account check-name --name $SA --query nameAvailable -o tsv
```

---

## 1. Account and providers

If more than one subscription is logged in, select the right one first:

```bash
az account list -o table
az account set --subscription <subscription-id-or-name>
az account show -o table
```

A new subscription has no resource providers registered. Registration is free and takes a few minutes.

```bash
for p in Microsoft.Compute Microsoft.Storage Microsoft.Network; do
  az provider register -n $p
done

# poll until all three say "Registered"
for p in Microsoft.Compute Microsoft.Storage Microsoft.Network; do
  printf "%-22s %s\n" "$p" "$(az provider show -n $p --query registrationState -o tsv)"
done
```

---

## 2. Region choice

The Planetary Computer data is in **West Europe** (Amsterdam), which is the ideal region. It may refuse you:

```
(RequestDisallowedByAzure) Resource '<name>' was disallowed by Azure:
The selected region is currently not accepting new customers.
```

West Europe is frequently at capacity and closed to new subscriptions. Probe nearby regions in order of latency to Amsterdam and take the first that accepts. This run landed on **`germanywestcentral`** (Frankfurt), about 10 ms from the data — still 30× better than 300 ms from home.

```bash
for L in germanywestcentral uksouth francecentral northeurope \
         switzerlandnorth swedencentral italynorth polandcentral; do
  az storage account create -n $SA -g $RG -l $L --sku Standard_LRS --kind StorageV2 \
    --access-tier Hot --min-tls-version TLS1_2 --allow-blob-public-access false \
    --query primaryLocation -o tsv 2>&1 && { echo "OK $L"; break; } || echo "no $L"
done
```

### Zone restriction — the first trap

`az vm list-skus` may report every D/F/E size in a region as `NotAvailableForSubscription`. That looks like a hard block. Check the restriction `type` before you believe it:

```bash
az vm list-skus -l $LOC --resource-type virtualMachines --size Standard_D8ls_v6 \
  --query "[].{all:locationInfo[0].zones, blocked:restrictions[0].restrictionInfo.zones, type:restrictions[0].type}" -o json
```

```json
{ "all": ["1","2","3"], "blocked": ["1"], "type": "Zone" }
```

`type: Zone` means only the listed zones are blocked. The rest are open, so **`az vm create` must pass an explicit `--zone`.** Without it Azure may place the VM in the blocked zone and refuse.

A filter such as `[?length(restrictions)==`0`]` is wrong here. It discards every size that has a merely zonal restriction, and returns nothing.

### Quota

```bash
az vm list-usage -l $LOC -o table
```

A new Pay-As-You-Go subscription typically starts with:

| Quota | Typical limit |
| --- | --- |
| Total Regional vCPUs | 10 |
| Total Regional Low-priority (Spot) vCPUs | 3 |

An 8-vCPU VM fits the regional quota. **Spot is usually not usable** at that size, because 3 vCPU is below the 8 the VM needs. Spot rents Azure's unused capacity at roughly 85% off, but Azure can reclaim it on 30 seconds notice. For a job of a few hours the saving is a dollar or two, so on-demand is the simpler choice even when quota allows spot.

---

## 3. Blob storage

Blob is the durable home for the data. It outlives every VM, costs about $0.0184/GB per month in the Hot tier, and any VM in any zone can read it.

A managed data disk was rejected. A disk attaches to one VM in one zone, and 128 GB of Premium SSD costs about $20/month whether attached or not, against about $1.12/month for 61 GB of blob.

```bash
az group create -n $RG -l $LOC        # the group's location is metadata only

az storage account create \
  -n $SA -g $RG -l $LOC \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false

az storage container create -n $CONTAINER --account-name $SA --auth-mode login
```

### Data-plane rights — the second trap

Subscription Owner does **not** grant blob read or write. That is a separate data-plane role and must be assigned explicitly.

```bash
az role assignment create \
  --assignee $ME \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$SA"
```

Allow about 2 minutes for the assignment to propagate before you use it.

---

## 4. The VM

```bash
az vm create \
  -g $RG -n $VM -l $LOC \
  --zone 2 \
  --image Ubuntu2404 \
  --size Standard_D8ls_v6 \
  --os-disk-size-gb 256 \
  --storage-sku StandardSSD_LRS \
  --admin-username $VM_USER \
  --ssh-key-values ~/.ssh/id_rsa.pub \
  --assign-identity \
  --nic-delete-option delete \
  --os-disk-delete-option delete \
  --public-ip-sku Standard

export VMIP=$(az vm show -g $RG -n $VM -d --query publicIps -o tsv)
```

| Field | Value |
| --- | --- |
| Size | `Standard_D8ls_v6` — 8 vCPU, 16 GB RAM, no local NVMe |
| Zone | must be an unblocked zone — see section 2 |
| Image | Ubuntu 24.04 LTS |
| OS disk | 256 GB `StandardSSD_LRS` |
| Identity | system-assigned |

### Give the VM its own blob access

The managed identity removes every key and SAS token from the VM.

```bash
export VMID=$(az vm show -g $RG -n $VM --query identity.principalId -o tsv)

az role assignment create --assignee $VMID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$SA"
```

---

## 5. AzCopy

### Install

AzCopy is a single static binary. Do **not** use `snap install azcli` — that is a different package.

```bash
# workstation, no sudo needed
mkdir -p ~/.local/bin
curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
tar xzf /tmp/azcopy.tgz -C /tmp
mv /tmp/azcopy_linux_amd64_*/azcopy ~/.local/bin/ && chmod +x ~/.local/bin/azcopy
hash -r     # bash caches the earlier "not found" result

# VM
curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
tar xzf /tmp/azcopy.tgz -C /tmp && sudo mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
```

### Authentication — the third trap

`azcopy login` can fail on a tenant with Conditional Access enabled:

```
Error Code: 530035
App name: Azure Storage AzCopy
Device state: Unregistered
```

Error **530035** means a Conditional Access policy requires a registered or compliant device. A workstation that is not Entra-joined is blocked, while the Azure CLI app is exempt from the same policy.

The fix is to stop using `azcopy login` and let AzCopy borrow a token from a client that already works:

```bash
# workstation — reuse the Azure CLI token
echo 'export AZCOPY_AUTO_LOGIN_TYPE=AZCLI' >> ~/.bashrc

# VM — use the managed identity; no interactive sign-in, so no Conditional Access
echo 'export AZCOPY_AUTO_LOGIN_TYPE=MSI' >> ~/.bashrc
```

Verify with `azcopy list "https://$SA.blob.core.windows.net/$CONTAINER"`. It should print `INFO: Authenticating to source using Azure AD`.

---

## 6. Push local data up

Existing chips are worth keeping — `ingest-chips` is incremental and continues from the manifest, so the VM does not re-fetch them.

`data/tmp/` holds abandoned `s2dl_*` scratch directories from killed runs. `--exclude-path "tmp"` leaves them behind without deleting anything.

```bash
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI

azcopy sync ./data "https://$SA.blob.core.windows.net/$CONTAINER/data" \
  --recursive --exclude-path "tmp"
```

Measured on this run: **7,134 files, 6.24 GB, 33 minutes** at about 25 Mbit/s on a 40 Mbit line, 0 failures. Azure does not charge for ingress.

Check progress from another shell:

```bash
az storage blob list -c $CONTAINER --account-name $SA --auth-mode login \
  --query "length(@)" -o tsv
```

---

## 7. Configure the VM

```bash
ssh $VM_USER@$VMIP
```

```bash
sudo apt-get update && sudo apt-get install -y build-essential gh tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
tar xzf /tmp/azcopy.tgz -C /tmp && sudo mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
echo 'export AZCOPY_AUTO_LOGIN_TYPE=MSI' >> ~/.bashrc && source ~/.bashrc

gh auth login
gh repo clone <your-github-org>/<your-repo>
cd <your-repo>
uv sync
```

`build-essential` must install before `uv sync`, which builds GDAL and rasterio wheels.

### Pull the data down

Wait for the upload in section 6 to finish first.

```bash
azcopy sync "https://<storage-account>.blob.core.windows.net/<container>/data" ./data --recursive
```

This runs inside Azure. The same 5.9 GB that took 33 minutes to upload came down in **under a minute**.

### Gitignored files

`gh repo clone` does not bring these, and they are only a few hundred KB. Send them from the workstation **after** the clone, so git does not refuse a non-empty directory:

```bash
tar czf - CLAUDE.md tasks src/inspo scripts \
  | ssh $VM_USER@$VMIP 'tar xzf - -C ~/<your-repo>'
```

Do **not** copy `.venv`. Rebuild it with `uv sync`.

---

## 8. Run the chipping

Use `tmux`, otherwise the job dies when the SSH session drops.

```bash
tmux new -s chip                     # reattach later with: tmux attach -t chip
cd ~/<your-repo>
uv run cmrv ingest-chips --max-workers 16 --read-pool 8
```

Detach with `Ctrl+b` then `d`.

Start at 16×8, not the 20×4 used at home. Latency is now ~10 ms instead of 300 ms, so more requests must be in flight.

Measured on `Standard_D8ls_v6` at 16×8:

```
0.64 - 0.87 obs/s      home line was 0.075 obs/s  ->  about 11x faster
load average 1.9 / 8   about 24% CPU
```

The job is **not** CPU-bound, despite the low latency. At 24% CPU the Planetary Computer rate limit is the ceiling, so a bigger VM does not help. Use `scripts/concurrency_probe.sh` to find the best setting for your own run.

Watch for HTTP 429 in the log. That is the Planetary Computer rate limit, which `s2.stac_retry` absorbs with a jittered backoff. If 429s are frequent, step the concurrency down. **The rate limit, not bandwidth, is the new ceiling.**

Both Ctrl+C and `pkill` reach the save path — see the stopping notes in `CLAUDE.md`. Expect a stop to take up to about 2 minutes while open reads unwind.

### Save the results

Run this every few hours, not only at the end. The OS disk survives a deallocate, but nothing survives a deleted VM.

```bash
azcopy sync ./data "https://<storage-account>.blob.core.windows.net/<container>/data" \
  --recursive --exclude-path "tmp"
```

---

## 9. Costs and teardown

| Item | Rate | Notes |
| --- | --- | --- |
| `Standard_D8ls_v6` | ~$0.46/hr | only while the VM runs |
| 256 GB StandardSSD OS disk | ~$19/month | charged even when deallocated |
| Blob, Hot, 61 GB | ~$1.12/month | |
| Blob, Cool, 61 GB | ~$0.61/month | move here after `embed` succeeds |
| Reading 2.6 TB from the Planetary Computer | **free** | the source account pays egress |
| Upload from the workstation (ingress) | **free** | |

Measured, not estimated. At 0.64-0.87 obs/s the remaining ~84,000 chips need **27 to 36 hours** of VM time:

```
VM      27-36 h x ~$0.44/hr   =  $12 - $16
disk    27-36 h x ~$0.026/hr  =  $0.70 - $0.94
IP      27-36 h x ~$0.005/hr  =  $0.14 - $0.18
--------------------------------------------
total                            $13 - $17
```

An earlier estimate of "3.5 hours, under $5" assumed the job would become CPU-bound once latency disappeared. It does not. Budget about **$20** for a full run from scratch.

```bash
az vm deallocate -g $RG -n $VM        # stops compute charges, keeps the disk
az vm start      -g $RG -n $VM        # resume
az vm delete     -g $RG -n $VM --yes  # also deletes the disk and NIC
az group delete  -n $RG --yes         # deletes everything, blobs included
```

Delete the VM when chipping finishes. Keep the storage account — it holds the results and costs about a dollar a month.

---
