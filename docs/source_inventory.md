# Source inventory and authority decisions

## Inspected locations

- `Tweed_Volcano_Project_Code/Science_Fair_ML_V2/` — canonical active V2 implementation, frozen checkpoints, outputs, AUSSEF manuscript lineage and mentor package.
- `Tweed_Volcano_Project_Code/Science_Fair_Extension/ML_V2/` — historical namespace pointer confirming the canonical V2 path and separation from frozen V1.
- `Tweed_Volcano_Project_Code/Science_Fair_Extension/00_Governance/ML_V1_FREEZE/` — frozen V1 boundary; not copied or modified.
- `Tweed_Volcano_Project/` (authoritative Google Drive scientific root) — searched read-only for AUSSEF/manuscript/science-fair/landslide-ML candidates; no competing ML V2 source was found.
- Parent Git metadata — only the unrelated Tweed reconstruction remote was found.

## Authority decisions

- **V2 code:** `Science_Fair_ML_V2/src/sfv2/` and the experiment-specific reproducibility scripts.
- **Primary model comparison:** frozen Stage 4E-B files under `outputs/stage4e_b/`.
- **Dataset and selected-model contracts:** Stage 5A outputs.
- **Sensitivity analyses:** elevation and transferability-factor packages plus Stage 4E-A/B secondary outputs.
- **Geological history:** Stage 5G and the Stage D audit package; feasibility only.
- **Final manuscript:** G5.1 corrected DOCX copied into the mentor package, SHA-256 `160579d...1018b`.
- **V1:** separate frozen pilot; excluded from this repository except for explanatory references.
- **Unfinished candidate/lockbox work:** not presented as completed validation and no results copied.

## Manuscript preservation discrepancy

The G5.1 correction log records the pre-correction source `AUSSEF FINAL REVIEW.docx` as SHA-256 `28461738...083f0`. Investigation found that the read-only backup `00_ORIGINAL_BACKUP/AUSSEF FINAL REVIEW (original backup).docx` still has that exact hash, size 4,556,017 bytes, revision 1 and an internal modification time of 18 October 2026 03:20 UTC. The root file was subsequently resaved or edited: it hashes to `1aafab42...d7e7d`, is 3,965,193 bytes, identifies revision 4 and has an internal modification time of 25 October 2026 04:02 UTC. Its document text is substantively different (429 non-empty paragraphs versus 382, with 79 non-equal paragraph blocks), so this was not a metadata-only change.

This explains the hash mismatch: the correction log describes the preserved original backup, while the root predecessor later changed. The exact editing process and provenance of the 25 October revision are not recorded in Git and remain unresolved. The corrected G5.1 DOCX still matches its recorded hash `160579d9...1018b` and its mentor-package copy exactly, so the later predecessor edit does not change the selected final manuscript authority. No manuscript file was modified during this investigation.
