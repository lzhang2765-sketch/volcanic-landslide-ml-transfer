"""Inventory-level child records derived only from existing Stage-1 evidence."""

from __future__ import annotations

from .schema import INVENTORY_COLUMNS

UNKNOWN = "UNKNOWN"


def build_inventory_rows(candidates: list[dict], sources: list[dict]) -> list[dict]:
    """Create one supported child record per known landscape-level inventory.

    The schema permits multiple rows per landscape. Stage 1 does not split a
    publication or dataset into extra inventories unless the existing records
    establish that distinction.
    """
    by_landscape: dict[str, list[dict]] = {}
    for source in sources:
        by_landscape.setdefault(source["landscape_id"], []).append(source)

    rows: list[dict] = []
    for candidate in sorted(candidates, key=lambda row: row["landscape_id"]):
        if candidate["inventory_available"] != "TRUE" or candidate["inventory_name"] == UNKNOWN:
            continue
        landscape_sources = by_landscape.get(candidate["landscape_id"], [])
        y_sources = sorted((s for s in landscape_sources if s["data_role"] == "Y"), key=lambda s: s["source_id"])
        omega_sources = sorted((s for s in landscape_sources if s["data_role"] == "OMEGA"), key=lambda s: s["source_id"])
        primary_y = y_sources[0]["source_id"] if y_sources else UNKNOWN
        if omega_sources:
            omega_source = omega_sources[0]["source_id"]
        elif y_sources and candidate["observation_domain_source"] != UNKNOWN:
            omega_source = primary_y
        else:
            omega_source = UNKNOWN

        access_route = candidate["inventory_download_url"]
        if access_route == UNKNOWN:
            access_route = candidate["inventory_url_or_identifier"]
        row = {
            "inventory_id": f"INV_{candidate['landscape_id']}_01",
            "landscape_id": candidate["landscape_id"],
            "inventory_name": candidate["inventory_name"],
            "inventory_source_id": primary_y,
            "inventory_year": candidate["inventory_year"],
            "inventory_trigger": candidate["inventory_trigger"],
            "inventory_mapping_method": candidate["inventory_mapping_method"],
            "geometry_type": candidate["inventory_geometry_type"],
            "target_tier": candidate["inventory_tier"],
            "feature_count_estimate": candidate["inventory_count_estimate"],
            "mapped_area": candidate["inventory_area"],
            "observation_domain_source_id": omega_source,
            "observation_domain_establishable": candidate["observation_domain_establishable"],
            "verification_status": candidate["y_verification"],
            "downloadable": candidate["inventory_downloadable"],
            "access_route": access_route,
            "licence": candidate["licence"],
            "notes": "Stage-1 child record; landscape summary retained for screening convenience; refine inventory-specific evidence in Stage 2",
        }
        rows.append({column: row[column] for column in INVENTORY_COLUMNS})
    return rows
