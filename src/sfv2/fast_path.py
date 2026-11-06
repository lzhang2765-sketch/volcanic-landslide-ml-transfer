"""FAST_PATH gate manifest: reusable, explicit, fail-closed production gating for a new landscape.

Automation may compute evidence, but it can never turn a scientific decision into PASS: every gate flagged
`requires_decision` needs a decision record (who/what document decided it) before it may be PASS, and every PASS
needs evidence.  Production may start only when every gate is PASS.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUSES = ("PASS", "FAIL", "PENDING", "DECISION_REQUIRED")

# Ordered fast-path steps, the existing V2 implementation each one reuses, and the gate it feeds.
STEPS = [
    ("RAW_VALIDATION", "raw_store.acquire / stage2f_transfer.register_incoming (immutable, hashed, originals preserved)", "PROVENANCE"),
    ("PROVENANCE", "metadata/stage2b_raw_registry.json append-only record (URL, publisher, DOI, licence, access date, SHA256, bytes, role)", "PROVENANCE"),
    ("Y_ONTOLOGY", "explicit target family + tier + phase freeze from source documentation (stage2f2 TARGET_FREEZE pattern)", "Y"),
    ("OMEGA_VALIDATION", "published/search/analysis extent minus documented non-observation masks; never derived from Y", "OMEGA"),
    ("M_TEMPORAL_VALIDATION", "Copernicus GLO-30 XML acquisition dates < event; EDM/FLM quality layers carried", "M_TEMPORAL"),
    ("GEOLOGY_HARMONISATION", "authoritative unit polygons -> frozen ontology (stage2c_geology CLASSES, DOMAIN_RULE); UNKNOWN preserved", "GEOLOGY"),
    ("VOLCANIC_DOMAIN", "volcanic class AND Cenozoic age (frozen DOMAIN_RULE)", "VOLCANIC_DOMAIN"),
    ("OMEGA_ANALYSIS", "Omega - visibility - DEM nodata - water, restricted to the volcanic domain; every removal quantified", "OMEGA"),
    ("TERRAIN_UNITS_90M", "stage2f2_production.build_polygon_landscape (byte-equivalent mirror of frozen Stage 2D)", "TERRAIN_GENERATION"),
    ("FROZEN_M_FEATURES", "stage2d_production.feature_arrays (30 features / 60 columns, contract 9232a018...)", "TERRAIN_GENERATION"),
    ("TARGET_LABELS", "frozen 9-cell-centre polygon rule / frozen point rule; sensitivity phases written separately", "TARGET_SEMANTICS"),
    ("QC", "stage2d_production._qc + DEM-quality ALL_VALID vs STRICT_DEM_QUALITY quantification", "TERRAIN_GENERATION"),
    ("CHECKPOINT", "scientific checkpoint JSON + separate non-authoritative volatile diagnostics", "PROVENANCE"),
]

# Gates whose PASS is a scientific judgement, not a computation.
DECISION_GATES = {"Y", "OMEGA", "GEOLOGY", "VOLCANIC_DOMAIN", "TARGET_SEMANTICS"}
REQUIRED_GATES = ("PROVENANCE", "Y", "OMEGA", "M_TEMPORAL", "GEOLOGY", "VOLCANIC_DOMAIN", "DEM_COVERAGE", "TARGET_SEMANTICS",
                  "TERRAIN_GENERATION")


class GateError(ValueError):
    pass


@dataclass
class Gate:
    name: str
    status: str = "PENDING"
    evidence: str = ""
    decision_record: str = ""

    @property
    def requires_decision(self) -> bool:
        return self.name in DECISION_GATES


@dataclass
class GateManifest:
    landscape_id: str
    gates: dict = field(default_factory=lambda: {g: Gate(g) for g in REQUIRED_GATES})

    def set(self, name: str, status: str, evidence: str = "", decision_record: str = "") -> None:
        if name not in self.gates:
            raise GateError(f"unknown gate {name}")
        if status not in STATUSES:
            raise GateError(f"invalid status {status}")
        gate = Gate(name, status, evidence, decision_record)
        if status == "PASS" and not evidence:
            raise GateError(f"{name}: PASS requires evidence")
        if status == "PASS" and gate.requires_decision and not decision_record:
            raise GateError(f"{name}: scientific gate cannot PASS automatically; a decision record is required")
        self.gates[name] = gate

    def production_allowed(self) -> bool:
        return all(g.status == "PASS" for g in self.gates.values())

    def blockers(self) -> list[str]:
        return [f"{g.name}:{g.status}" for g in self.gates.values() if g.status != "PASS"]

    def to_dict(self) -> dict:
        return {"landscape_id": self.landscape_id, "production_allowed": self.production_allowed(),
                "gates": {k: asdict(v) | {"requires_decision": v.requires_decision} for k, v in self.gates.items()}}
