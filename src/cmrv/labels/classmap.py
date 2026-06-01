"""Single-source-of-truth class crosswalk: ``species_normalized`` → ``class_id``.

Reads ``class_maps.<name>`` from ``configs/labels_schema.yaml`` and exposes a
``ClassMap`` object that resolves a species string to a ``class_id`` via:

1. Exact lowercase binomial match against ``members[]``.
2. Genus-only fallback — first word of the species name matched against
   ``genus`` of any class with ``genus_fallback: true``.

If neither resolves, returns ``None`` (caller logs + drops).

During migration the loader also accepts the legacy top-level ``species_map``
block; when present, an info-level log records that the schema is using the
old shape.  Once all configs migrate, the fallback branch can be removed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from cmrv.io import load_config


@dataclass(frozen=True)
class ClassMap:
    """Resolved crosswalk for one ``class_maps.<name>`` entry."""

    name: str
    binomial_to_class: dict[str, int] = field(default_factory=dict)
    genus_to_class: dict[str, int] = field(default_factory=dict)
    class_meta: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def class_ids(self) -> set[int]:
        return set(self.class_meta.keys())

    def resolve(self, species: str | None) -> tuple[int | None, str]:
        """Return ``(class_id, resolution_path)`` for a species name.

        ``resolution_path`` ∈ ``{"exact", "genus_fallback", "unmapped"}``.
        Empty / None / NaN strings are treated as unmapped.
        """
        if not species:
            return None, "unmapped"
        norm = species.lower().strip()
        if not norm:
            return None, "unmapped"
        if norm in self.binomial_to_class:
            return self.binomial_to_class[norm], "exact"
        genus = norm.split()[0]
        if genus and genus in self.genus_to_class:
            return self.genus_to_class[genus], "genus_fallback"
        return None, "unmapped"


def build_lookup(
    schema_path: str | Path,
    class_map_name: str,
) -> ClassMap:
    """Build a :class:`ClassMap` from a schema YAML.

    Prefers the new ``class_maps.<name>.<id>.members[]`` shape.  If that block
    is missing/empty for the requested class map, falls back to the legacy
    top-level ``species_map`` block (treated as binomial-only — the legacy
    in-line genus split that ``chips.py`` used is replicated here so behavior
    is identical to the pre-refactor code).

    Validation is **warn-only** — duplicates, conflicting genus_fallback
    claims, and missing genus fields all log warnings but never raise.
    """
    schema = load_config(schema_path)
    class_maps = schema.get("class_maps", {}) or {}

    if class_map_name not in class_maps:
        raise KeyError(
            f"class_map '{class_map_name}' not found in {schema_path} "
            f"(have: {sorted(class_maps.keys())})"
        )

    cm_block = class_maps[class_map_name] or {}

    has_members = any(
        isinstance(v, dict) and v.get("members") for v in cm_block.values()
    )

    if has_members:
        return _build_from_members(class_map_name, cm_block)

    legacy_species_map = schema.get("species_map") or {}
    if legacy_species_map:
        logger.info(
            "class_map '{}': using legacy species_map fallback "
            "(no members[] in class_maps.{}.<id>)",
            class_map_name,
            class_map_name,
        )
        return _build_from_legacy(class_map_name, cm_block, legacy_species_map)

    raise ValueError(
        f"class_map '{class_map_name}' has no members[] entries and no "
        f"top-level species_map fallback — nothing to crosswalk"
    )


def _build_from_members(name: str, cm_block: dict[Any, Any]) -> ClassMap:
    binomial_to_class: dict[str, int] = {}
    genus_to_class: dict[str, int] = {}
    class_meta: dict[int, dict[str, Any]] = {}

    binomial_owners: dict[str, list[int]] = {}
    genus_claims: list[tuple[str, int]] = []

    for raw_id, entry in cm_block.items():
        if not isinstance(entry, dict):
            logger.warning(
                "class_map '{}': class {} is not a mapping — skipping", name, raw_id
            )
            continue
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning(
                "class_map '{}': class id {!r} is not an int — skipping", name, raw_id
            )
            continue

        members = list(entry.get("members") or [])
        # Classes with ``source:`` set are fed by a non-species resolver
        # (vegmap biome encoding, NLC class names).  Treat the class's own
        # ``name`` as an implicit binomial member so manifest rows whose
        # ``species`` matches it (e.g. ``"fynbos"``, ``"indigenous_forest"``)
        # resolve to this class via the standard lookup.
        if entry.get("source") and entry.get("name"):
            members.append(str(entry["name"]))
        if not members and not entry.get("source"):
            logger.warning(
                "class_map '{}': class {} has empty members[] and no "
                "'source:' annotation",
                name,
                class_id,
            )

        class_meta[class_id] = {
            k: v for k, v in entry.items() if k != "members"
        }

        for m in members:
            key = str(m).lower().strip()
            if not key:
                continue
            binomial_owners.setdefault(key, []).append(class_id)
            binomial_to_class[key] = class_id  # last-write-wins

        if entry.get("genus_fallback", False):
            genus = entry.get("genus")
            if not genus and members:
                genus = str(members[0]).split()[0].lower()
                logger.warning(
                    "class_map '{}': class {} has genus_fallback=true but no "
                    "explicit 'genus:' field — inferring '{}' from first member",
                    name,
                    class_id,
                    genus,
                )
            elif not genus:
                logger.warning(
                    "class_map '{}': class {} has genus_fallback=true but no "
                    "'genus:' and no members — skipping fallback",
                    name,
                    class_id,
                )
                continue
            g = str(genus).lower().strip()
            genus_claims.append((g, class_id))
            genus_to_class[g] = class_id  # last-write-wins

    for binomial, owners in binomial_owners.items():
        if len(owners) > 1:
            logger.warning(
                "class_map '{}': binomial {!r} appears in multiple classes {} "
                "— last-write-wins ({})",
                name,
                binomial,
                owners,
                owners[-1],
            )

    genus_count = Counter(g for g, _ in genus_claims)
    for g, n in genus_count.items():
        if n > 1:
            owners = [cid for gg, cid in genus_claims if gg == g]
            logger.warning(
                "class_map '{}': genus {!r} claimed by multiple classes {} "
                "— last-write-wins ({})",
                name,
                g,
                owners,
                owners[-1],
            )

    return ClassMap(
        name=name,
        binomial_to_class=binomial_to_class,
        genus_to_class=genus_to_class,
        class_meta=class_meta,
    )


def gbif_taxa_from_schema(
    schema_path: str | Path,
    class_map_name: str,
) -> list[dict[str, Any]]:
    """Derive the GBIF download taxa list from ``class_maps.<name>.members[]``.

    Each output row: ``{"name": <member>, "class_id": <parent class_id>}``.
    Falls back to the legacy ``gbif.taxa`` block when no members[] are present
    (parity with pre-refactor behavior).
    """
    schema = load_config(schema_path)
    cm_block = (schema.get("class_maps") or {}).get(class_map_name) or {}

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_id, entry in cm_block.items():
        if not isinstance(entry, dict):
            continue
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            continue
        for m in entry.get("members") or []:
            name = str(m).strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "class_id": cid})

    if out:
        return out

    legacy = (schema.get("gbif") or {}).get("taxa") or []
    if legacy:
        logger.info(
            "class_map '{}': using legacy gbif.taxa[] for download list "
            "(no members[] in class_maps)",
            class_map_name,
        )
    return [dict(t) for t in legacy]


def _build_from_legacy(
    name: str,
    cm_block: dict[Any, Any],
    legacy_species_map: dict[str, Any],
) -> ClassMap:
    """Reproduce the pre-refactor lookup: keys with no space act as genus."""
    binomial_to_class: dict[str, int] = {}
    genus_to_class: dict[str, int] = {}
    for k, v in legacy_species_map.items():
        key = str(k).lower().strip()
        cid = int(v)
        if " " in key:
            binomial_to_class[key] = cid
        else:
            genus_to_class[key] = cid

    class_meta: dict[int, dict[str, Any]] = {}
    for raw_id, entry in cm_block.items():
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(entry, dict):
            class_meta[cid] = {k: v for k, v in entry.items() if k != "members"}

    return ClassMap(
        name=name,
        binomial_to_class=binomial_to_class,
        genus_to_class=genus_to_class,
        class_meta=class_meta,
    )
