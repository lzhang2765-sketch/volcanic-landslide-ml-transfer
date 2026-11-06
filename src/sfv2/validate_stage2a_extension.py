from __future__ import annotations
import hashlib
from .schema import ROOT, FUTURE_FIELDS, VOCAB

READINESS = {"READY_CORE", "READY_CORE_H", "PARTIAL", "BLOCKED", "REJECT", "NOT_SCREENED"}
LINEAGE_RISK = {"LOW", "MODERATE", "HIGH", "UNRESOLVED"}
LEAKAGE_RISK = {"LOW", "MODERATE", "HIGH", "UNRESOLVED"}
SOURCE_ROLES = {"Y", "OMEGA", "DEM", "GEOLOGY", "ENVIRONMENT", "AGE", "H_CONTROL", "REFERENCE", "Y_METHOD", "DISCOVERY_CATALOGUE"}

def validate(candidates,sources,inventories,prior_hashes,acq):
    errors=[]; ids=[r["landscape_id"] for r in candidates];sids=[r["source_id"] for r in sources];iids=[r["inventory_id"] for r in inventories]
    for label,vals in (("landscape",ids),("source",sids),("inventory",iids)):
        if len(vals)!=len(set(vals)):errors.append(f"duplicate {label} IDs")
    idset,sidset=set(ids),set(sids)
    for s in sources:
        if s["landscape_id"] not in idset|{"GLOBAL"}:errors.append(f"bad source FK {s['source_id']}")
        unknown_roles=set(s["data_role"].split("|"))-SOURCE_ROLES
        if unknown_roles:errors.append(f"uncontrolled source role {s['source_id']}:{sorted(unknown_roles)}")
    for i in inventories:
        if i["landscape_id"] not in idset:errors.append(f"bad inventory landscape FK {i['inventory_id']}")
        if i["inventory_source_id"] not in sidset|{"UNKNOWN"}:errors.append(f"bad inventory source FK {i['inventory_id']}")
        if i["observation_domain_source_id"] not in sidset|{"UNKNOWN"}:errors.append(f"bad inventory Omega FK {i['inventory_id']}")
        if i["target_tier"] not in VOCAB["inventory_tier"]:errors.append(f"uncontrolled target tier {i['inventory_id']}")
        if i["verification_status"] not in VOCAB["verification"]:errors.append(f"uncontrolled inventory verification {i['inventory_id']}")
        if i["downloadable"] not in VOCAB["tristate"]:errors.append(f"uncontrolled inventory downloadable {i['inventory_id']}")
    for r in candidates:
        status=r["stage2a_extension_readiness"]
        for field in ("y_gate","omega_gate","m_gate","h_status"):
            if r[field] not in VOCAB["gate"]:errors.append(f"uncontrolled {field} {r['landscape_id']}:{r[field]}")
        if r["inventory_tier"] not in VOCAB["inventory_tier"]:errors.append(f"uncontrolled inventory tier {r['landscape_id']}")
        if status not in READINESS:errors.append(f"uncontrolled readiness {r['landscape_id']}:{status}")
        if r["stage2a_extension_investigated"]=="TRUE":
            if r["stage2a_extension_lineage_risk"] not in LINEAGE_RISK:errors.append(f"uncontrolled lineage risk {r['landscape_id']}")
            if r["stage2a_extension_m_leakage_risk"] not in LEAKAGE_RISK:errors.append(f"uncontrolled M leakage risk {r['landscape_id']}")
            for sid in r["stage2a_extension_source_ids"].split("|"):
                if sid and sid not in sidset:errors.append(f"missing extension claim source {r['landscape_id']}:{sid}")
        if status.startswith("READY_") and (r["y_gate"],r["omega_gate"],r["m_gate"]) != ("PASS","PASS","PASS"):errors.append(f"ready gates invalid {r['landscape_id']}")
        if status=="READY_CORE_H" and r["h_status"]!="PASS":errors.append(f"ready H invalid {r['landscape_id']}")
        if any(r[f] for f in FUTURE_FIELDS):errors.append(f"future metric populated {r['landscape_id']}")
        if r["experimental_role"]!="UNASSIGNED" or r["lockbox_status"]!="UNASSIGNED":errors.append(f"role assigned {r['landscape_id']}")
        if r["landscape_id"]=="MATZACO_PUEBLA_2017" and r["omega_gate"]!="FAIL":errors.append("Matzaco incomplete survey treated as Omega")
        if r["landscape_id"]=="WILSONS_RIVER_NSW_2022" and r["stage2a_extension_lineage_risk"]!="HIGH":errors.append("Wilsons lineage weakened")
    for rel,digest in prior_hashes.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:errors.append(f"prior checkpoint/table changed {rel}")
    aids=[r["acquisition_id"] for r in acq]
    if len(aids)!=len(set(aids)):errors.append("duplicate acquisition ID")
    for r in acq:
        if r["checksum"]:errors.append(f"premature checksum {r['acquisition_id']}")
        if r["download_status"]=="COMPLETE":errors.append(f"false completed download {r['acquisition_id']}")
    return {"valid":not errors,"error_count":len(errors),"errors":errors,"checks":{"candidates":len(candidates),"sources":len(sources),"inventories":len(inventories),"investigated":sum(r["stage2a_extension_investigated"]=="TRUE" for r in candidates),"prior_preserved":not any(e.startswith("prior") for e in errors),"acquisition_rows":len(acq)}}
