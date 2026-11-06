"""Read-only command-line interface for Stage 5B-A candidate intake."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import inbox as intake_inbox
from . import intake


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sfv2.intake_cli", description="Read-only scientific candidate intake")
    parser.add_argument("--metadata-only", action="store_true", help="do not hash or inspect file content")
    parser.add_argument("--restricted-lockbox-safe", action="store_true", help="enforce unopened-lockbox content restrictions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "audit", "qualify", "report", "draft-questions"):
        p = sub.add_parser(name); p.add_argument("--manifest", required=True)
        if name in {"audit", "qualify", "report", "draft-questions"}:
            p.add_argument("--audit-json")
        if name == "qualify":
            p.add_argument("--gates", required=True); p.add_argument("--judgments")
    inv = sub.add_parser("inventory"); inv.add_argument("--manifest", required=True); inv.add_argument("--root", required=True)
    for name in ("inbox-status", "inbox-audit", "inbox-paths"):
        p = sub.add_parser(name)
        p.add_argument("--candidate", choices=sorted(intake_inbox.CANDIDATE_PATHS))
        p.add_argument("--config", default=str(intake_inbox.DEFAULT_CONFIG))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in {"inbox-status", "inbox-audit", "inbox-paths"}:
        guard = intake_inbox.InboxGuard.from_config(Path(args.config))
        if args.command == "inbox-paths":
            _emit(guard.paths())
        else:
            result = guard.status(args.candidate)
            result["operation"] = args.command.replace("-", "_").upper()
            result["access_events"] = guard.events
            _emit(result)
        return
    manifest = _load(args.manifest)
    validation = intake.validate_manifest(manifest)
    if args.command == "register":
        _emit({"operation": "REGISTER", "read_only": True, "validation": validation,
               "state": "REGISTERED" if validation["valid"] else "BLOCKED_MISSING_EVIDENCE"})
        return
    if not validation["valid"]:
        _emit({"operation": args.command.upper(), "state": "BLOCKED_MISSING_EVIDENCE", "validation": validation})
        return
    if args.command == "inventory":
        _emit(intake.inventory_files(Path(args.root), manifest, metadata_only=args.metadata_only,
                                     restricted_lockbox_safe=args.restricted_lockbox_safe or manifest["ACCESS_CLASSIFICATION"] in intake.LOCKBOX_CLASSES))
        return
    audit = _load(args.audit_json) if getattr(args, "audit_json", None) else {}
    if args.command == "audit":
        _emit({"operation": "AUDIT", "read_only": True, "metadata_only": args.metadata_only,
               "restricted_lockbox_safe": args.restricted_lockbox_safe, "manifest_validation": validation,
               "omega": intake.qualify_omega(audit.get("OMEGA_EVIDENCE")),
               "dem": intake.qualify_dem(manifest, audit.get("DEM_EVIDENCE")),
               "diversity": intake.assess_diversity(audit.get("DIVERSITY_ATTRIBUTES")),
               "historical_topography": intake.assess_historical_topography(audit.get("HISTORICAL_PRODUCTS"))})
    elif args.command == "qualify":
        judgments = _load(args.judgments) if args.judgments else []
        _emit(intake.qualify(manifest, _load(args.gates), judgments))
    elif args.command == "draft-questions":
        _emit({"dataset_id": manifest["DATASET_ID"], "draft_only": True,
               "questions": intake.draft_questions(manifest, audit)})
    else:
        _emit({"dataset_id": manifest["DATASET_ID"], "report_type": "INTAKE_STATUS",
               "read_only": True, "manifest_validation": validation, "audit": audit,
               "notice": "No qualification is implied by file readability."})


if __name__ == "__main__":
    main()
