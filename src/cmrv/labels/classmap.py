"""Single-source-of-truth class crosswalk: ``species_normalized`` → ``class_id``.

Reads ``class_maps.<name>`` from ``configs/labels_schema.yaml`` and exposes a
``ClassMap`` object that resolves a species string to a ``class_id`` via:

1. Exact lowercase binomial match against ``members[]``.
2. Genus-only fallback — first word of the species name matched against
   ``genus`` of any class with ``genus_fallback: true``.

If neither resolves, returns ``None`` (caller logs + drops).
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

    def resolve(self, species: str | None) -> int | None:
        """Return the ``class_id`` for a species name, or ``None`` if unmapped.

        Exact lowercase binomial first, then genus fallback. Empty / None / NaN
        strings are unmapped.
        """
        if not species:
            return None
        norm = species.lower().strip()
        if not norm:
            return None
        if norm in self.binomial_to_class:
            return self.binomial_to_class[norm]
        genus = norm.split()[0]
        return self.genus_to_class.get(genus) if genus else None


def build_lookup(
    schema_path: str | Path,
    class_map_name: str,
) -> ClassMap:
    """Build a :class:`ClassMap` from a schema YAML.

    Reads the ``class_maps.<name>.<id>.members[]`` block. Validation is
    **warn-only** — duplicates, conflicting genus_fallback claims, and missing
    genus fields all log warnings but never raise.
    """
    schema = load_config(schema_path)
    class_maps = schema.get("class_maps", {}) or {}

    if class_map_name not in class_maps:
        raise KeyError(
            f"class_map '{class_map_name}' not found in {schema_path} "
            f"(have: {sorted(class_maps.keys())})"
        )

    cm_block = class_maps[class_map_name] or {}
    has_members = any(isinstance(v, dict) and v.get("members") for v in cm_block.values())
    if not has_members:
        raise ValueError(f"class_map '{class_map_name}' has no members[] entries to crosswalk")
    return _build_from_members(class_map_name, cm_block)


def _build_from_members(name: str, cm_block: dict[Any, Any]) -> ClassMap:
    binomial_to_class: dict[str, int] = {}
    genus_to_class: dict[str, int] = {}
    class_meta: dict[int, dict[str, Any]] = {}

    binomial_owners: dict[str, list[int]] = {}
    genus_claims: list[tuple[str, int]] = []

    for raw_id, entry in cm_block.items():
        if not isinstance(entry, dict):
            logger.warning("class_map '{}': class {} is not a mapping — skipping", name, raw_id)
            continue
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("class_map '{}': class id {!r} is not an int — skipping", name, raw_id)
            continue

        members = list(entry.get("members") or [])
        # Classes with a ``source:`` annotation are fed by a non-species
        # resolver (e.g. a future raster/polygon source for native classes).
        # Treat the class's own ``name`` as an implicit binomial member so
        # manifest rows whose ``species`` equals it resolve via the standard
        # lookup.  (No such classes in the IAP-only Phase 0 schema.)
        if entry.get("source") and entry.get("name"):
            members.append(str(entry["name"]))
        if not members and not entry.get("source"):
            logger.warning(
                "class_map '{}': class {} has empty members[] and no 'source:' annotation",
                name,
                class_id,
            )

        class_meta[class_id] = {k: v for k, v in entry.items() if k != "members"}

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
                "class_map '{}': genus {!r} claimed by multiple classes {} — last-write-wins ({})",
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
