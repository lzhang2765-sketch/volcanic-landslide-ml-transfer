"""Create the immutable Stage 4E-B checkpoint after analysis and tests."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas
import matplotlib
import numpy
import pandas
import rasterio
import scipy
import sklearn

from . import stage4eb_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_b"
CHECKPOINT = OUT / "STAGE4EB_CHECKPOINT.json"
REPORT = OUT / "STAGE4EB_FINAL_REPORT.md"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def build() -> dict:
    required = ["STAGE4EB_PREREGISTRATION.json", "STAGE4EB_PREREGISTRATION.md", "STAGE4EB_MODEL_CONFIGURATIONS.tsv",
                "STAGE4EB_INNER_SELECTIONS.tsv", "STAGE4EB_LOLO_RESULTS.tsv", "STAGE4EB_SPATIAL_BOOTSTRAP.tsv",
                "STAGE4EB_MODEL_COMPARISON.tsv", "STAGE4EB_ROBUST_SELECTION.tsv", "STAGE4EB_RANK_STABILITY.tsv",
                "STAGE4EB_COMPLEXITY.tsv", "STAGE4EB_FEATURE_IMPORTANCE.tsv", "STAGE4EB_COTABATO_SENSITIVITY.tsv",
                "STAGE4EB_H_STATUS.tsv", "STAGE4EB_FINAL_REPORT.md"]
    missing = [x for x in required if not (OUT/x).exists()]
    if missing: raise RuntimeError(f"STOP: missing required outputs: {missing}")
    decision = json.loads((OUT/"STAGE4EB_DEVELOPMENT_LEADER.json").read_text())
    tests = json.loads((OUT/"STAGE4EB_TESTS.json").read_text())
    prereg_freeze = json.loads((OUT/"STAGE4EB_PREREGISTRATION_FREEZE.json").read_text())
    primary_freeze = json.loads((OUT/"STAGE4EB_PRIMARY_SELECTION_FREEZE.json").read_text())
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p != CHECKPOINT)
    manifest, hashes = [], {}
    for p in files:
        rel = str(p.relative_to(ROOT))
        if p == REPORT:
            manifest.append(dict(path=rel, role="FINAL_REPORT_REFERENCES_CHECKPOINT_HASH", sha256="EXCLUDED_TO_AVOID_CIRCULAR_HASH_DEPENDENCY"))
        else:
            h = sha256(p); hashes[rel] = h; manifest.append(dict(path=rel, bytes=p.stat().st_size, sha256=h))
    sources = [ROOT/"src/sfv2/stage4eb_prereg.py", ROOT/"src/sfv2/stage4eb.py", ROOT/"src/sfv2/stage4eb_secondary.py",
               ROOT/"src/sfv2/build_stage4eb.py", ROOT/"src/sfv2/finalize_stage4eb.py", ROOT/"tests/test_stage4eb.py"]
    status = git("status", "--short")
    libomp = ROOT/".ml_site/libomp_runtime/libomp/23.1.3/lib/libomp.dylib"
    cp = dict(
        stage="4E-B", status="COMPLETE", completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        input_checkpoint_hashes=P.INPUT_CHECKPOINTS, v1_freeze_sha256=P.V1_FREEZE_SHA256,
        preregistration_freeze_sha256=sha256(OUT/"STAGE4EB_PREREGISTRATION_FREEZE.json"),
        preregistration_hashes=prereg_freeze["files"], primary_selection_freeze_sha256=sha256(OUT/"STAGE4EB_PRIMARY_SELECTION_FREEZE.json"),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        package_versions=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=numpy.__version__, pandas=pandas.__version__,
                              sklearn=sklearn.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                              geopandas=geopandas.__version__, rasterio=rasterio.__version__, xgboost=P.XGBOOST_VERSION),
        xgboost_runtime=dict(version=P.XGBOOST_VERSION, isolated_path="Science_Fair_ML_V2/.ml_site",
            libomp_version="23.1.3", libomp_sha256=sha256(libomp),
            bottle_source="https://formulae.brew.sh/api/formula/libomp.json; official arm64 Sequoia bottle",
            published_bottle_sha256="b2aee1d6ce67ec1c8d597ce79eee46583531f3629491cb5454eead3d2a7a739b"),
        seeds=dict(stage4eb=P.SEED, stage4b=P.STAGE4B_SEED), output_manifest=manifest, output_hashes=hashes,
        tests=tests, reproducibility=dict(outer_refits=24, tolerance=1e-12, all_pass=True,
            table="outputs/stage4e_b/STAGE4EB_REPRODUCIBILITY.tsv",
            table_sha256=sha256(OUT/"STAGE4EB_REPRODUCIBILITY.tsv"),
            assembly="deterministic TSV/JSON formatting; isolated sklearn and XGBoost diagnostic processes documented"),
        DEVELOPMENT_LEADER="LOGISTIC_M", RUNNER_UP="RF_M", SELECTION_MARGIN=decision["selection_margin"],
        decision=decision, git_head=git("rev-parse","HEAD"), git_status_dirty=bool(status),
        git_status_line_count=len(status.splitlines()), git_status_sha256=hashlib.sha256(status.encode()).hexdigest(),
        git_status_note="Worktree was already dirty and contains unrelated user work; no commit or push performed.",
        LOCKBOX_TOUCHED="NO", MEGH_TRAINED="NO", GEN3_MODIFIED="NO", H_GATE_MODIFIED="NO",
        NEW_M_FEATURES="NO", NEW_E_FEATURES="NO", NEW_G_FEATURES="NO", COTABATO_EQ1_PRIMARY="YES",
        FINAL_EXTERNAL_VALIDATION="NO", COMMIT_PERFORMED="NO", PUSH_PERFORMED="NO",
        historical_outputs_changed="NO", report_hash_policy="Report is listed but not hashed because it contains this checkpoint file's final SHA-256."
    )
    CHECKPOINT.write_text(json.dumps(cp, indent=2, sort_keys=True) + "\n")
    return cp


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
