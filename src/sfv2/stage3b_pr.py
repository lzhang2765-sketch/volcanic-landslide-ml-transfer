"""Stage 3B Track B: Puerto Rico (Hurricane Maria 2017) - Y verification, exact OFR 98-38 harmonisation, M audit, Omega options.

Geology uses the FROZEN Stage 2C ontology (classes, VOLCANIC set, DOMAIN_RULE, composition()) with explicit source-specific
rules for OFR 98-38 (as Stage 2C did for the Aso map).  The rules read each unit's DOMINANT lithology = the first sentence of
its official description (fallback: unit name); the earliest-listed lithology wins.  No unit is forced: unmatched -> UNKNOWN.
"""
from __future__ import annotations

import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

from .raw_store import load_registry
from .schema import ROOT
from .stage2c_geology import CLASSES, DOMAIN_RULE, VOLCANIC, composition

LID = "PUERTO_RICO_MARIA_2017"
UTM = "EPSG:32619"
EVENT = "2017-09-20"  # Hurricane Maria landfall
PRIOR_STORM = "2017-09-06"  # Hurricane Irma passed NE of PR; pre-event M must precede both
Y_ZIP = ROOT / "data/raw/stage2e/PUERTO_RICO_MARIA_2017/V1_TRANSFER/Y/landslides.zip"
GEO_ACQ = "ACQ_PUERTO_RICO_GEOLOGY_USGS_OFR98_38"

CENOZOIC = r"paleocene|eocene|oligocene|miocene|pliocene|pleistocene|holocene|tertiary|quaternary|paleogene|neogene"
MESOZOIC = r"cretaceous|jurassic|maestrichtian|maastrichtian|campanian|santonian|coniacian|turonian|cenomanian|albian|aptian|kimme"

# (rule_id, regex on dominant lithology, class, confidence, note); earliest match position in the sentence wins.
NAME_RULES = [  # applied first, on the unit NAME, for units whose identity is defined by the name itself
    ("PRN01_SURFICIAL", r"alluvium|beach deposits|landslide deposits|swamp deposits|blanket sand|artificial fill", "UNCONSOLIDATED_SURFICIAL", "HIGH",
     "Quaternary surficial deposit (name)"),
    ("PRN02_ALTERED", r"hydrothermally altered", "UNKNOWN", "LOW", "Hydrothermal alteration product: no frozen ontology class"),
    ("PRN03_FAULT_BRECCIA", r"fault breccia", "UNKNOWN", "LOW", "Tectonic breccia of variable protolith"),
    ("PRN04_METAMORPHIC", r"amphibolite|serpentinite|metavolcanic", "METAMORPHIC_BASEMENT", "HIGH", "Metamorphic / ophiolitic basement"),
    ("PRN05_INTRUSIVE_NAME", r"porphyry|syenite|granodiorite|diorite|gabbro|monzonite", "INTRUSIVE", "MEDIUM",
     "Unit named as a porphyry/plutonic rock (hypabyssal or plutonic body)"),
]
EMPTY_NAME_RULE = ("PRN06_UNDIVIDED_NO_DESCRIPTION", r"undivided|interbedded|formations", "UNKNOWN", "LOW",
                   "Combined/undivided formations without an official description")
LITH_RULES = [
    ("PR01_INTRUSIVE", r"granodiorite|diorite|gabbro|monzonite|syenite|hornblendite|\bporphyry\b|pluton", "INTRUSIVE", "HIGH", "Plutonic/hypabyssal"),
    ("PR02_LAVA", r"\blavas?\b|\bflows\b", "LAVA_BY_COMPOSITION", "HIGH", "Lava flows; class from composition"),
    ("PR03_PYROCLASTIC", r"ash-flow|\btuff\b|\btuffs\b|pumiceous|ignimbrite|crystal-lithic tuff|vitric", "PYROCLASTIC", "HIGH", "Pyroclastic deposits"),
    ("PR04_HYALOCLASTITE", r"hyaloclastite", "MIXED_VOLCANIC", "MEDIUM", "Autoclastic volcanic breccia (hyaloclastite)"),
    ("PR05_VOLCANICLASTIC", r"volcaniclastic|volcanic (sandstone|breccia|conglomerate|siltstone|mudstone)|tuffaceous (sandstone|siltstone|mudstone)|epiclastic",
     "VOLCANICLASTIC", "MEDIUM", "Reworked/epiclastic volcanic deposits"),
    ("PR06_VOLCANIC_UNDIV", r"\b(?:trachy)?basalts?\b|\bandesites?\b|\bdacites?\b|\brhyodacites?\b|\brhyolites?\b|keratophyre|volcanic rocks",
     "MIXED_VOLCANIC", "MEDIUM", "Volcanic rock named as a noun, form undivided (composition adjectives alone never decide)"),
    ("PR07_SEDIMENTARY", r"limestone|calcarenite|calcirudite|calcilutite|chalk|marl|\bclay\b|claystone|\bsand\b|sandstone|siltstone|mudstone|"
                         r"conglomerate|\bchert\b|biomicrite|gravel|shale", "SEDIMENTARY_BASEMENT", "HIGH", "Consolidated sedimentary rocks"),
]
FRAGMENTAL = r"tuff|breccia|hyaloclastite|pyroclastic|volcanic sandstone"
INTRUSIVE_HINT = r"intrusion|intrusive|\bdikes?\b|\bsills?\b|\bstocks?\b|\bplugs?\b"


def units() -> pd.DataFrame:
    reg = {e["acquisition_id"]: e for e in load_registry()}
    lines = zipfile.ZipFile(ROOT / reg[GEO_ACQ]["raw_path"]).read("prunit.txt").decode("latin-1").splitlines()
    rows = [[c.strip() for c in ln.split("|")] for ln in lines[2:] if ln.count("|") >= 4]
    return pd.DataFrame([r[:5] for r in rows], columns=["symbol", "name", "age", "descript", "ref"]).drop_duplicates("symbol").sort_values("symbol")


def age_class(age: str) -> str:
    a = age.lower()
    cen, mes = bool(re.search(CENOZOIC, a)), bool(re.search(MESOZOIC, a))
    return "CENOZOIC" if cen and not mes else "PRE_CENOZOIC" if mes and not cen else "K_T_STRADDLING" if cen and mes else "AGE_UNRESOLVED"


def classify_unit(name: str, descript: str) -> dict:
    dominant = (descript.split(". ")[0] if descript else "").strip()
    rules = NAME_RULES + ([EMPTY_NAME_RULE] if not descript.strip() else [])
    for rid, pat, cls, conf, note in rules:
        if re.search(pat, name.lower()):
            return dict(dominant_lithology=dominant or name, harmonised_class=cls, mapping_confidence=conf, mapping_rule=rid, rationale=note,
                        composition="NOT_APPLICABLE")
    text = (dominant or name).lower()
    hits = sorted((m.start(), i) for i, (_, pat, *_r) in enumerate(LITH_RULES) for m in [re.search(pat, text)] if m)
    if not hits:
        return dict(dominant_lithology=dominant or name, harmonised_class="UNKNOWN", mapping_confidence="LOW", mapping_rule="NO_RULE",
                    rationale="No rule matched; not forced into a class", composition="UNKNOWN")
    rid, _, cls, conf, note = LITH_RULES[hits[0][1]]
    comp = composition(text)
    if cls == "LAVA_BY_COMPOSITION":
        if re.search(FRAGMENTAL, text):
            cls, rid, note = "MIXED_VOLCANIC", rid + "+FRAGMENTAL", "Lava with breccia/tuff in the dominant lithology"
        else:
            cls = {"MAFIC": "BASALTIC_LAVA", "INTERMEDIATE": "INTERMEDIATE_LAVA", "FELSIC": "FELSIC_LAVA"}.get(comp, "MIXED_VOLCANIC")
            conf = conf if comp in ("MAFIC", "INTERMEDIATE", "FELSIC") else "LOW"
    elif cls == "PYROCLASTIC" and re.search(r"\blavas?\b|\bflows\b", text):
        cls, rid, note = "MIXED_VOLCANIC", rid + "+LAVA", "Pyroclastic with lava in the dominant lithology"
    if cls in VOLCANIC and re.search(INTRUSIVE_HINT, descript.lower()) and not re.search(FRAGMENTAL + r"|sandstone|lava|flows", text):
        cls, rid, conf, note = "INTRUSIVE", rid + "+INTRUSION_HINT", "MEDIUM", "Igneous body described with intrusion/dike features"
    return dict(dominant_lithology=dominant, harmonised_class=cls, mapping_confidence=conf, mapping_rule=rid, rationale=note,
                composition=comp if cls in VOLCANIC or cls == "INTRUSIVE" else "NOT_APPLICABLE")


def mapping() -> pd.DataFrame:
    rows = []
    for u in units().itertuples():
        c = classify_unit(u.name, u.descript)
        ac = age_class(u.age)
        assert c["harmonised_class"] in CLASSES
        rows.append(dict(source_unit_code=u.symbol, source_unit_name=u.name, source_age=u.age, age_class=ac,
                         source_lithology=c["dominant_lithology"], harmonised_class=c["harmonised_class"], composition=c["composition"],
                         mapping_confidence=c["mapping_confidence"], mapping_rule=c["mapping_rule"], rationale=c["rationale"],
                         is_volcanic=c["harmonised_class"] in VOLCANIC,
                         domain_eligible=c["harmonised_class"] in VOLCANIC and ac == "CENOZOIC",
                         evidence=f"{GEO_ACQ} prunit.txt; {u.ref}"))
    return pd.DataFrame(rows)


def geology_frame() -> gpd.GeoDataFrame:
    reg = {e["acquisition_id"]: e for e in load_registry()}
    g = gpd.read_file(f"zip://{ROOT / reg[GEO_ACQ]['raw_path']}!PR_Geology.shp").to_crs(UTM)
    m = mapping().set_index("source_unit_code")
    g["FMATN"] = g.FMATN.fillna("UNMAPPED_POLYGON")
    for col in ("harmonised_class", "age_class", "domain_eligible", "is_volcanic"):
        g[col] = g.FMATN.map(m[col])
    g["harmonised_class"] = g.harmonised_class.fillna("UNKNOWN")
    g["age_class"] = g.age_class.fillna("AGE_UNRESOLVED")
    g["domain_eligible"] = g.domain_eligible.fillna(False).astype(bool)
    g["is_volcanic"] = g.is_volcanic.fillna(False).astype(bool)
    return g


def main_island(g: gpd.GeoDataFrame):
    """Main-island land from the official geology polygons (independent of Y): largest connected landmass."""
    parts = gpd.GeoSeries([g.union_all()], crs=UTM).explode(index_parts=False)
    return parts.iloc[int(np.argmax(parts.area.to_numpy()))]


def y_points() -> gpd.GeoDataFrame:
    return gpd.read_file(f"zip://{Y_ZIP}").to_crs(UTM)


def y_verification(y: gpd.GeoDataFrame, island) -> dict:
    xy = np.round(np.c_[y.geometry.x, y.geometry.y], 3)
    _, inv, cnt = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    attrs = [c for c in y.columns if c != "geometry"]
    return dict(source="Hughes et al. 2019, USGS data release doi:10.5066/P9BVMD74 (V1-transferred RAW, byte-identical)",
                event="Hurricane Maria landfall 2017-09-20; imagery 2017-09-22..2017-11-30", features=len(y), geometry=sorted(set(y.geom_type)),
                attributes=attrs, exact_duplicate_coordinate_groups=int((cnt > 1).sum()), points_in_duplicate_groups=int(cnt[cnt > 1].sum()),
                points_on_main_island=int(y.within(island).sum()), points_off_main_island=int((~y.within(island)).sum()),
                municipios=int(y["Municipio"].nunique()) if "Municipio" in y else None,
                target_family="POINT_HEADSCARP_TIER1", target_meaning="Headscarp centre of each slope failure (initiation zone): TIER_1",
                temporal_semantics="Event inventory; metadata warns some failures may pre-date Maria (Irma, 2017-09-06) or post-date it before imagery",
                small_island_exclusion="Desecheo, Mona, Monito, Caja de Muertos not examined; none mapped on Vieques/Culebra",
                duplicate_handling="Exact-coordinate duplicates are counted once per terrain unit by the frozen point rule (no extra events)",
                y_altered=False)


def geology_quantification(g: gpd.GeoDataFrame, y: gpd.GeoDataFrame, island) -> dict:
    gi = g.copy()
    gi["geometry"] = gi.geometry.intersection(island)
    gi = gi[~gi.geometry.is_empty]
    gi["km2"] = gi.area / 1e6
    j = gpd.sjoin(y[["geometry"]], gi[["harmonised_class", "age_class", "domain_eligible", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    j["harmonised_class"] = j.harmonised_class.fillna("OUTSIDE_GEOLOGY")
    j["age_class"] = j.age_class.fillna("OUTSIDE_GEOLOGY")
    j["domain_eligible"] = j.domain_eligible.fillna(False).astype(bool)
    key = lambda df: df.harmonised_class + "|" + df.age_class
    by_area = gi.groupby(key(gi)).km2.sum().round(3).sort_index().to_dict()
    by_pts = j.groupby(key(j)).size().sort_index().astype(int).to_dict()
    vol = gi.is_volcanic.astype(bool)
    return dict(main_island_land_km2=round(gi.km2.sum(), 3), area_km2_by_class_and_age=by_area, headscarps_by_class_and_age=by_pts,
                cenozoic_volcanic_eligible_km2=round(gi[gi.domain_eligible].km2.sum(), 3),
                cenozoic_volcanic_eligible_headscarps=int(j.domain_eligible.sum()),
                pre_cenozoic_volcanic_km2=round(gi[vol & (gi.age_class == "PRE_CENOZOIC")].km2.sum(), 3),
                pre_cenozoic_volcanic_headscarps=int(((j.harmonised_class.isin(VOLCANIC)) & (j.age_class == "PRE_CENOZOIC")).sum()),
                k_t_straddling_volcanic_km2=round(gi[vol & (gi.age_class == "K_T_STRADDLING")].km2.sum(), 3),
                k_t_straddling_volcanic_headscarps=int(((j.harmonised_class.isin(VOLCANIC)) & (j.age_class == "K_T_STRADDLING")).sum()),
                headscarps_total=len(y), eligible_domain_parts=int(len(gpd.GeoSeries([gi[gi.domain_eligible].union_all()]).explode())),
                eligible_units=sorted(gi[gi.domain_eligible].FMATN.unique().tolist()), domain_rule=DOMAIN_RULE, rule_changed=False)
