"""Benchmark MPC Sentinel-2 reads: /vsicurl vs /vsiaz_streaming vs azure-storage-blob.

All three paths fetch the same 1024x1024 window from the same COG and decode
the same chunk of data. The question is which method delivers the bytes
fastest for our stackstac-style windowed workload.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from urllib.parse import urlparse

import planetary_computer as pc
import pystac_client
import rasterio
from azure.storage.blob import BlobClient
from rasterio.io import MemoryFile
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
AOI_BBOX = [18.4, -34.2, 19.4, -33.5]  # Cape Peninsula
N_ITEMS = 3
WIN = Window(5000, 5000, 1024, 1024)
BAND = "B04"


def fetch_items():
    client = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = client.search(
        collections=["sentinel-2-l2a"],
        bbox=AOI_BBOX,
        datetime="2022-04-01/2022-04-30",
        query={"eo:cloud_cover": {"lt": 40}},
        max_items=N_ITEMS,
    ).item_collection()
    pc.sign_inplace(items)
    return items


def vsicurl_read(url: str) -> tuple[float, int]:
    t0 = time.perf_counter()
    with rasterio.open(url) as src:
        arr = src.read(1, window=WIN)
    return time.perf_counter() - t0, arr.nbytes


def vsiaz_streaming_read(url: str) -> tuple[float, int]:
    # GDAL /vsiaz_streaming/ wants container/blob + AZURE_STORAGE_* env vars
    parsed = urlparse(url)
    account = parsed.netloc.split(".")[0]
    parts = parsed.path.lstrip("/").split("/", 1)
    container, blob = parts[0], parts[1]
    sas = parsed.query

    os.environ["AZURE_STORAGE_ACCOUNT"] = account
    os.environ["AZURE_STORAGE_SAS_TOKEN"] = sas

    vsi = f"/vsiaz_streaming/{container}/{blob}"
    t0 = time.perf_counter()
    with rasterio.open(vsi) as src:
        arr = src.read(1, window=WIN)
    return time.perf_counter() - t0, arr.nbytes


def azblob_full_read(url: str) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    blob = BlobClient.from_blob_url(url)
    data = blob.download_blob().readall()
    bytes_down = len(data)
    with MemoryFile(BytesIO(data)) as mem, mem.open() as src:
        arr = src.read(1, window=WIN)
    return time.perf_counter() - t0, arr.nbytes, bytes_down


def main() -> None:
    items = fetch_items()
    urls = [item.assets[BAND].href for item in items]
    print(f"\nBenchmarking {len(urls)} items, window={WIN}, band={BAND}\n")

    # warm-up — first read of any COG pays extra TLS handshake etc.
    print("warm-up…")
    vsicurl_read(urls[0])

    print(f"\n{'item':<6}{'vsicurl (s)':<14}{'vsiaz_str (s)':<16}{'azblob full (s)':<18}{'blob MB':<10}")
    print("-" * 70)
    t_curl = t_az = t_blob = 0.0
    total_blob_bytes = 0
    for i, url in enumerate(urls):
        a, _ = vsicurl_read(url)
        t_curl += a
        try:
            b, _ = vsiaz_streaming_read(url)
        except Exception as e:
            b = float("nan")
            print(f"vsiaz_streaming failed on item {i}: {e}")
        else:
            t_az += b
        c, _, blob_bytes = azblob_full_read(url)
        t_blob += c
        total_blob_bytes += blob_bytes
        print(f"{i:<6}{a:<14.2f}{b:<16.2f}{c:<18.2f}{blob_bytes / 1e6:<10.1f}")

    n = len(urls)
    print("-" * 70)
    print(f"{'avg':<6}{t_curl / n:<14.2f}{t_az / n:<16.2f}{t_blob / n:<18.2f}")
    print(f"\ntotal bytes via azure-storage-blob: {total_blob_bytes / 1e6:.1f} MB")
    print(f"effective throughput (azblob): {total_blob_bytes / 1e6 / t_blob:.1f} MB/s")


if __name__ == "__main__":
    main()
