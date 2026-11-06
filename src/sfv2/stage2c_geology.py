"""V2 geology harmonisation ontology and explicit per-source mapping rules (Stage 2C).

Mapping is rule-based on the SOURCE's own lithology/description text; the first matching rule wins and is recorded.
Ambiguous units are not forced: 'lava & pyroclastic' units map to MIXED_VOLCANIC with a separate composition attribute,
and units no rule recognises map to UNKNOWN (LOW confidence).  Original source semantics are always retained.
"""
from __future__ import annotations

import re

CLASSES = ["BASALTIC_LAVA", "INTERMEDIATE_LAVA", "FELSIC_LAVA", "PYROCLASTIC", "VOLCANICLASTIC", "MIXED_VOLCANIC", "INTRUSIVE",
           "SEDIMENTARY_BASEMENT", "METAMORPHIC_BASEMENT", "UNCONSOLIDATED_SURFICIAL", "WATER", "UNKNOWN"]
DOMAIN_RULE = "Volcanic analysis domain = units whose harmonised class is volcanic AND whose formation age is Cenozoic (pre-Cenozoic volcanic rocks are basement, not volcanic landscapes)"
VOLCANIC = {"BASALTIC_LAVA", "INTERMEDIATE_LAVA", "FELSIC_LAVA", "PYROCLASTIC", "VOLCANICLASTIC", "MIXED_VOLCANIC"}
COMPOSITIONS = ["MAFIC", "INTERMEDIATE", "FELSIC", "MIXED", "NOT_APPLICABLE", "UNKNOWN"]


def composition(text: str) -> str:
    t = text.lower()
    mafic = bool(re.search(r"\bbasalt", t)) and "basaltic andesite" not in t or "mafic" in t and "ultramafic" not in t
    inter = "andesite" in t
    felsic = bool(re.search(r"dacite|rhyolite|felsic", t))
    hits = [n for n, f in (("MAFIC", mafic), ("INTERMEDIATE", inter), ("FELSIC", felsic)) if f]
    return hits[0] if len(hits) == 1 else ("MIXED" if hits else "UNKNOWN")


# (rule_id, regex on lowercase source text, class, confidence, note) -- order matters: first match wins
SEAMLESS_RULES = [
    ("S01_WATER", r"^(water|sea|lake)\b|water area", "WATER", "HIGH", "Water body"),
    ("S02_INTRUSIVE", r"intrusive|dike|granit|granodiorite|tonalite|diorite|gabbro|plutonic|porphyr", "INTRUSIVE", "HIGH", "Plutonic/hypabyssal"),
    ("S02b_ULTRAMAFIC", r"ultramafic|peridotite", "INTRUSIVE", "MEDIUM", "Ultramafic (plutonic/mantle) rocks"),
    ("S02c_ACCRETIONARY", r"accretionary complex", "SEDIMENTARY_BASEMENT", "MEDIUM",
     "Accretionary-complex unit (incl. accreted oceanic basalt/greenstone blocks): basement, not volcanic terrain"),
    ("S03_METAMORPHIC", r"schist|gneiss|metamorphic|hornfels|phyllite|serpentinite|amphibolite|slate", "METAMORPHIC_BASEMENT", "HIGH", "Metamorphic"),
    ("S04_MIXED_LAVA_PYRO", r"lava.*pyroclastic|pyroclastic.*lava", "MIXED_VOLCANIC", "HIGH", "Lava and pyroclastic rocks undivided; composition kept separately"),
    ("S05_LAVA", r"\blava", "LAVA_BY_COMPOSITION", "HIGH", "Lava; class from composition"),
    ("S06_PYROCLASTIC", r"pyroclastic|tuff|ignimbrite|welded|pumice|scoria|tephra|ash fall|air-fall", "PYROCLASTIC", "HIGH", "Pyroclastic deposits"),
    ("S07_VOLCANICLASTIC", r"volcanic.*(fan|debris|avalanche|lahar)|debris avalanche|volcaniclastic|lahar", "VOLCANICLASTIC", "MEDIUM",
     "Reworked/epiclastic volcanic deposits"),
    ("S08_VOLCANIC_UNDIV", r"volcanic rocks?|\bbasalt|\bandesite|\bdacite|\brhyolite", "MIXED_VOLCANIC", "MEDIUM", "Volcanic rocks undivided"),
    ("S09a_QUATERNARY_SEDIMENTS", r"\bsediments\b.*\|\|.*quaternary", "UNCONSOLIDATED_SURFICIAL", "MEDIUM",
     "Quaternary (non-)marine sediments: unconsolidated/semi-consolidated"),
    ("S09_SURFICIAL", r"terrace|valley floor|alluvi|levee|dune|coastal plain|river|talus|landslide deposit|fan deposit|beach|marsh|peat",
     "UNCONSOLIDATED_SURFICIAL", "HIGH", "Unconsolidated Quaternary surficial deposits"),
    ("S10_SEDIMENTARY", r"sandstone|mudstone|conglomerate|limestone|chert|siltstone|shale|accretionary|sedimentary", "SEDIMENTARY_BASEMENT", "HIGH",
     "Consolidated sedimentary or accretionary rocks"),
]
# Aso 1:50,000 (GSJ Volcano Map 04): units are named eruptive products; rules read LEGEND05E-LEGEND08E text
ASO_RULES = [
    ("A01_SCORIA_CONE", r"scoria cone|scoria", "PYROCLASTIC", "HIGH", "Scoria cone"),
    ("A02_PYRO_FLOW", r"pyroclastic flow|pyroclastic|air-fall|ash|pumice|tephra", "PYROCLASTIC", "HIGH", "Pyroclastic"),
    ("A03_LAVA", r"\blava", "LAVA_BY_COMPOSITION", "MEDIUM", "Named lava; composition from description where given"),
    ("A04_VOLCANO", r"volcano", "MIXED_VOLCANIC", "MEDIUM", "Named volcanic edifice (lava + pyroclastics undivided)"),
    ("A05_FILL", r"caldera-fill|lake deposit|lacustrine", "UNCONSOLIDATED_SURFICIAL", "MEDIUM", "Caldera-lake fill (mud, sand, gravel): sediments, not volcanic rock"),
    ("A06_SURFICIAL", r"talus|terrace|fan deposit|alluvi|river|gravel", "UNCONSOLIDATED_SURFICIAL", "HIGH", "Surficial deposits"),
    ("A06b_SEDIMENTARY", r"sandstone|mudstone|conglomerate", "SEDIMENTARY_BASEMENT", "HIGH", "Pre-volcanic sedimentary basement"),
    ("A07_FORMATION", r"formation", "UNKNOWN", "LOW", "Formation name without lithology"),
]


def classify(text: str, rules) -> tuple[str, str, str, str, str]:
    t = text.lower()
    comp = composition(t)
    for rule_id, pattern, cls, conf, note in rules:
        if re.search(pattern, t):
            if cls == "LAVA_BY_COMPOSITION":
                cls = {"MAFIC": "BASALTIC_LAVA", "INTERMEDIATE": "INTERMEDIATE_LAVA", "FELSIC": "FELSIC_LAVA"}.get(comp, "MIXED_VOLCANIC")
                conf = conf if comp in ("MAFIC", "INTERMEDIATE", "FELSIC") else "LOW"
            return cls, conf, rule_id, note, comp if cls in VOLCANIC or cls == "INTRUSIVE" else "NOT_APPLICABLE"
    return "UNKNOWN", "LOW", "NO_RULE", "No rule matched; not forced into a class", comp


def seamless_mapping(legend: list[dict]) -> list[dict]:
    rows = []
    for u in sorted(legend, key=lambda u: u["symbol"]):
        text = f"{u.get('lithology_en', '')} | {u.get('group_en', '')}"
        cls, conf, rule, note, comp = classify(f"{u.get('lithology_en', '')} || {u.get('formationAge_en', '')}", SEAMLESS_RULES)
        if cls == "UNKNOWN":  # second pass on group text only when lithology is silent
            cls, conf, rule, note, comp = classify(text, SEAMLESS_RULES)
            conf = "LOW" if cls != "UNKNOWN" else conf
        rows.append(dict(source="GSJ_SEAMLESS_V2", source_unit_code=u["symbol"], source_unit_name=u.get("title", ""),
                         source_lithology=u.get("lithology_en", ""), source_lithology_ja=u.get("lithology_ja", ""), source_age=u.get("formationAge_en", ""),
                         source_description=u.get("group_en", ""), source_colour=u["value"].lower(), harmonised_class=cls,
                         volcanic_composition=comp, is_volcanic=str(cls in VOLCANIC).upper(),
                         domain_eligible=str(cls in VOLCANIC and "cenozoic" in u.get("formationAge_en", "").lower()).upper(),
                         mapping_confidence=conf, mapping_rule=rule, mapping_notes=note))
    return rows


def aso_mapping(units: list[dict]) -> list[dict]:
    rows = []
    for u in sorted(units, key=lambda u: u["SYMBOL"]):
        text = " | ".join(str(u.get(k) or "") for k in ("LEGEND05E", "LEGEND06E", "LEGEND07E", "LEGEND08E"))
        cls, conf, rule, note, comp = classify(text, ASO_RULES)
        rows.append(dict(source="GSJ_ASO_VOLCANO_MAP_1_50000", source_unit_code=u["SYMBOL"], source_unit_name=str(u.get("LEGEND05E") or ""),
                         source_lithology=" | ".join(str(u.get(k) or "") for k in ("LEGEND06E", "LEGEND07E", "LEGEND08E")),
                         source_age=" | ".join(str(u.get(k) or "") for k in ("LEGEND02E", "LEGEND03E")), source_description=str(u.get("LEGEND04E") or ""),
                         harmonised_class=cls, volcanic_composition=comp, is_volcanic=str(cls in VOLCANIC).upper(),
                         domain_eligible=str(cls in VOLCANIC and "cenozoic" in str(u.get("LEGEND01E") or "").lower()).upper(), mapping_confidence=conf,
                         mapping_rule=rule, mapping_notes=note))
    return rows
