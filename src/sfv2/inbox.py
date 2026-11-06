"""Fail-closed, metadata-only candidate-inbox controls.

The module never opens restricted files, never extracts archives, and never scans
outside the one configured project-relative root.  Git ignore is complementary
hygiene and is not used as an access-control decision.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .schema import ROOT


CANDIDATE_PATHS = {
    "REUNION_SALAZIE_1978_1984": "reunion_salazie",
    "SAINT_LUCIA_TOMAS_2010": "saint_lucia_tomas",
    "VANUATU_HAROLD_2020": "vanuatu_harold",
    "POPOCATEPETL_PUEBLA_2017": "popocatepetl_puebla",
    "MOUNT_ELGON": "mount_elgon",
    "CHUUK_2002": "chuuk",
}
PUBLIC_SECTIONS = {"metadata", "documentation"}
ALL_SECTIONS = PUBLIC_SECTIONS | {"restricted"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
PLACEHOLDER_NAMES = {".gitkeep", ".inbox-placeholder", "README.placeholder"}
SENSITIVE_NAME_TOKENS = {"landslide_inventory", "outcome_labels", "ground_truth", "labels", "outcomes"}
DEFAULT_CONFIG = ROOT / "outputs/stage5b_b1/STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json"


class InboxAccessDenied(PermissionError):
    """Raised before an unauthorized path is opened or traversed."""


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(path.read_text())
    if config.get("authorized_root") != "candidate_inbox":
        raise InboxAccessDenied("authorized inbox root must be candidate_inbox")
    if config.get("candidate_paths") != CANDIDATE_PATHS:
        raise InboxAccessDenied("candidate path registry differs from frozen mapping")
    return config


@dataclass
class InboxGuard:
    root: Path
    candidate_paths: dict[str, str] = field(default_factory=lambda: dict(CANDIDATE_PATHS))
    test_mode: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, path: Path = DEFAULT_CONFIG) -> "InboxGuard":
        config = load_config(path)
        lexical = ROOT / config["authorized_root"]
        if lexical.is_symlink():
            raise InboxAccessDenied("authorized inbox root may not be a symlink")
        canonical = lexical.resolve(strict=True)
        expected = (ROOT / "candidate_inbox").resolve(strict=True)
        if canonical != expected or not _within(canonical, ROOT.resolve(strict=True)):
            raise InboxAccessDenied("authorized inbox does not resolve to the project-scoped root")
        return cls(canonical, dict(config["candidate_paths"]))

    def _candidate_root(self, candidate_id: str) -> Path:
        if candidate_id not in self.candidate_paths:
            self.events.append({"action": "RESOLVE_CANDIDATE", "decision": "DENY", "candidate_id": "UNKNOWN"})
            raise InboxAccessDenied("candidate ID is not registered")
        lexical = self.root / self.candidate_paths[candidate_id]
        if lexical.is_symlink():
            raise InboxAccessDenied("candidate directory may not be a symlink")
        resolved = lexical.resolve(strict=True)
        if not _within(resolved, self.root):
            raise InboxAccessDenied("candidate directory escapes authorized root")
        return resolved

    def resolve(self, candidate_id: str, section: str, relative: str = ".", *, metadata_only: bool = True) -> Path:
        section = section.lower()
        if section not in ALL_SECTIONS:
            raise InboxAccessDenied("unregistered inbox section")
        if metadata_only and section == "restricted":
            self.events.append({"action": "RESOLVE", "decision": "DENY", "candidate_id": candidate_id, "section": "restricted", "path_logged": False})
            raise InboxAccessDenied("restricted access requires explicit future authorization")
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise InboxAccessDenied("absolute paths and parent traversal are denied")
        candidate_root = self._candidate_root(candidate_id)
        section_lexical = candidate_root / section
        if section_lexical.is_symlink():
            raise InboxAccessDenied("section directory may not be a symlink")
        section_root = section_lexical.resolve(strict=True)
        target = (section_root / Path(*rel.parts)).resolve(strict=False)
        if not _within(target, section_root) or not _within(target, self.root):
            raise InboxAccessDenied("resolved path escapes authorized section")
        # Resolve every existing ancestor so a symlink cannot redirect a future leaf.
        cursor = section_root
        for part in rel.parts:
            cursor = cursor / part
            if cursor.exists() or cursor.is_symlink():
                if cursor.is_symlink() or not _within(cursor.resolve(strict=True), section_root):
                    raise InboxAccessDenied("symlink or path alias escapes authorized section")
        if target.is_file():
            name = target.name.lower().replace("-", "_").replace(" ", "_")
            if target.lstat().st_nlink > 1:
                raise InboxAccessDenied("hard-linked public file is conservatively blocked")
            if target.suffix.lower() in ARCHIVE_SUFFIXES or name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
                raise InboxAccessDenied("archive is quarantined and cannot be resolved for metadata access")
            if any(token in name for token in SENSITIVE_NAME_TOKENS):
                raise InboxAccessDenied("sensitive filename is quarantined")
        self.events.append({"action": "RESOLVE", "decision": "ALLOW", "candidate_id": candidate_id, "section": section, "metadata_only": metadata_only})
        return target

    def _public_section_summary(self, candidate_id: str, section: str) -> dict:
        section_root = self.resolve(candidate_id, section, metadata_only=True)
        counts = {"regular_files": 0, "placeholders": 0, "blocked_symlinks": 0, "blocked_hardlinks": 0, "quarantined_archives": 0, "quarantined_sensitive_names": 0}
        # Explicit section-only walk. Symlink directories are pruned and never followed.
        for current, directories, files in os.walk(section_root, topdown=True, followlinks=False):
            current_path = Path(current)
            safe_directories = []
            for name in directories:
                child = current_path / name
                if child.is_symlink():
                    counts["blocked_symlinks"] += 1
                elif _within(child.resolve(strict=True), section_root):
                    safe_directories.append(name)
                else:
                    counts["blocked_symlinks"] += 1
            directories[:] = safe_directories
            for name in files:
                path = current_path / name
                stat = path.lstat()
                if path.is_symlink():
                    counts["blocked_symlinks"] += 1
                elif stat.st_nlink > 1:
                    counts["blocked_hardlinks"] += 1
                elif name in PLACEHOLDER_NAMES:
                    counts["placeholders"] += 1
                elif path.suffix.lower() in ARCHIVE_SUFFIXES or name.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
                    counts["quarantined_archives"] += 1
                elif any(token in name.lower().replace("-", "_").replace(" ", "_") for token in SENSITIVE_NAME_TOKENS):
                    counts["quarantined_sensitive_names"] += 1
                else:
                    counts["regular_files"] += 1
        return counts

    def candidate_status(self, candidate_id: str) -> dict:
        candidate_root = self._candidate_root(candidate_id)
        metadata = self._public_section_summary(candidate_id, "metadata")
        documentation = self._public_section_summary(candidate_id, "documentation")
        restricted = candidate_root / "restricted"
        if restricted.is_symlink():
            restricted_status = "INVALID_SYMLINK_DENIED"
        else:
            restricted_status = "EXISTS_REDACTED_NOT_ENUMERATED" if restricted.is_dir() else "MISSING_FAIL_CLOSED"
        regular = metadata["regular_files"] + documentation["regular_files"]
        archive_quarantined = metadata["quarantined_archives"] + documentation["quarantined_archives"]
        sensitive_quarantined = metadata["quarantined_sensitive_names"] + documentation["quarantined_sensitive_names"]
        quarantined = archive_quarantined + sensitive_quarantined
        suspicious = metadata["blocked_symlinks"] + documentation["blocked_symlinks"] + metadata["blocked_hardlinks"] + documentation["blocked_hardlinks"]
        if suspicious or quarantined:
            arrival = "QUARANTINED_PENDING_MANUAL_CLASSIFICATION"
        elif regular:
            arrival = "FILES_PRESENT_PENDING_RECEIPT_REGISTRATION"
        else:
            arrival = "NOT_RECEIVED"
        return {
            "candidate_id": candidate_id, "relative_path": self.candidate_paths[candidate_id],
            "metadata_file_count": metadata["regular_files"], "documentation_file_count": documentation["regular_files"],
            "placeholder_count": metadata["placeholders"] + documentation["placeholders"],
            "blocked_symlink_count": metadata["blocked_symlinks"] + documentation["blocked_symlinks"],
            "blocked_hardlink_count": metadata["blocked_hardlinks"] + documentation["blocked_hardlinks"],
            "quarantined_archive_count": archive_quarantined, "restricted_directory": restricted_status,
            "quarantined_sensitive_name_count": sensitive_quarantined,
            "restricted_contents_enumerated": False, "restricted_contents_hashed": False,
            "non_placeholder_public_files_present": bool(regular), "arrival_status": arrival,
            "qualification_status": "UNQUALIFIED", "dataset_role": "UNDECIDED",
            "access_classification": "NEW_DEVELOPMENT_CANDIDATE", "lockbox_status": "NOT_DESIGNATED",
        }

    def status(self, candidate_id: str | None = None) -> dict:
        ids = [candidate_id] if candidate_id else list(self.candidate_paths)
        rows = [self.candidate_status(value) for value in ids]
        return {
            "operation": "INBOX_STATUS", "read_only": True, "metadata_only": True,
            "authorized_root": "candidate_inbox", "restricted_authorization": "DENIED_STAGE5B_B1",
            "candidates": rows, "scan_scope": "explicit registered public sections only",
            "restricted_security_note": "Directory naming and Git ignore are not security boundaries; code denies traversal.",
        }

    def paths(self) -> dict:
        return {
            "authorized_root": "candidate_inbox", "canonical_root": str(self.root),
            "candidate_paths": dict(self.candidate_paths), "public_sections": sorted(PUBLIC_SECTIONS),
            "restricted_section": "restricted", "restricted_default": "DENY",
        }
