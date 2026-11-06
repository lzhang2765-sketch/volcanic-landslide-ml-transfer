"""Fail-closed validation for Stage 2A."""
from __future__ import annotations
from .schema import FUTURE_FIELDS, ROOT
from .stage2a_records import R

def validate(candidates, sources, inventories, stage1_hashes):
    errors=[]
    ids=[r["landscape_id"] for r in candidates]; sids=[r["source_id"] for r in sources]; iids=[r["inventory_id"] for r in inventories]
    if len(ids)!=len(set(ids)): errors.append("duplicate landscape_id")
    if len(sids)!=len(set(sids)): errors.append("duplicate source_id")
    if len(iids)!=len(set(iids)): errors.append("duplicate inventory_id")
    idset,sidset=set(ids),set(sids)
    for source in sources:
        if source["landscape_id"] not in idset|{"GLOBAL"}: errors.append(f"bad source landscape FK {source['source_id']}")
    for inv in inventories:
        if inv["landscape_id"] not in idset: errors.append(f"bad inventory landscape FK {inv['inventory_id']}")
        if inv["inventory_source_id"] not in sidset|{"UNKNOWN"}: errors.append(f"bad inventory source FK {inv['inventory_id']}")
        if inv["observation_domain_source_id"] not in sidset|{"UNKNOWN"}: errors.append(f"bad inventory Omega FK {inv['inventory_id']}")
    for row in candidates:
        if any(row[field] for field in FUTURE_FIELDS): errors.append(f"future ML metric populated {row['landscape_id']}")
        if row["experimental_role"]!="UNASSIGNED" or row["lockbox_status"]!="UNASSIGNED": errors.append(f"role assigned {row['landscape_id']}")
        status=row["stage2a_readiness"]
        if status in {"READY_CORE","READY_CORE_H"} and (row["y_gate"],row["omega_gate"],row["m_gate"]) != ("PASS","PASS","PASS"):
            errors.append(f"ready without Y/Omega/M {row['landscape_id']}")
        if status=="READY_CORE_H" and row["h_status"]!="PASS": errors.append(f"READY_CORE_H without H PASS {row['landscape_id']}")
        if row["stage2a_screened"]=="TRUE":
            for sid in row["stage2a_source_ids"].split("|"):
                if sid and sid not in sidset: errors.append(f"missing claim source {row['landscape_id']}:{sid}")
    for rel,digest in stage1_hashes.items():
        import hashlib
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: errors.append(f"Stage1 modified {rel}")
    return {"valid":not errors,"error_count":len(errors),"errors":errors,"checks":{"candidates":len(candidates),"sources":len(sources),"inventories":len(inventories),"screened":len(R),"stage1_preserved":not any(e.startswith('Stage1 modified') for e in errors)}}

