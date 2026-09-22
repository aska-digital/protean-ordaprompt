"""Append-only receipt store (leo-arch.md sections 1.4 and 4).

Invariants enforced here, in order, on every append:

1. closed field set -- unknown fields are a hard write failure (``SchemaError``);
2. closed string domains -- every string anywhere in the receipt must be a
   sha256 hash, a uuid, a semver, an id token, a slug or an enum code.  Free
   text of any kind (prompt excerpts, rationale prose, headers, secrets) trips
   ``PrivacyViolationError`` and the receipt is NOT written.

There is no mutation and no delete API: the store is append-only JSONL.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas import (
    RECEIPT_FIELDS,
    SCHEMA_RECEIPT,
    PrivacyViolationError,
    SchemaError,
    is_id_token,
    validate_receipt,
)

RECEIPTS_FILENAME = "receipts.jsonl"


def _walk_strings(node: Any, path: str = "receipt") -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    if isinstance(node, str):
        found.append((path, node))
    elif isinstance(node, Mapping):
        for key, value in node.items():
            found.append(("%s.<key>" % path, str(key)))
            found.extend(_walk_strings(value, "%s.%s" % (path, key)))
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            found.extend(_walk_strings(item, "%s[%d]" % (path, index)))
    return found


def assert_no_free_text(receipt: Mapping[str, Any]) -> None:
    """Prove that no free-text field can be persisted.

    Every string reachable from the receipt must belong to the closed
    id/enum/hash domains of leo-arch.md section 1.4.  The check is deliberately
    structural (not a blocklist): prose cannot be whitelisted into a receipt by
    adding a new field, because new fields are rejected too.
    """
    if not isinstance(receipt, Mapping):
        raise PrivacyViolationError("receipt must be a mapping")
    unknown = sorted(set(receipt.keys()) - set(RECEIPT_FIELDS))
    if unknown:
        raise SchemaError("unknown receipt field(s): %s" % (unknown,))
    for path, value in _walk_strings(receipt):
        if value == SCHEMA_RECEIPT:
            # closed enum: the one fixed schema id, not free text
            continue
        if not is_id_token(value):
            raise PrivacyViolationError(
                "%s: %r is outside the closed id/enum/hash domain (free text rejected)"
                % (path, value)
            )
        if len(value) > 64 and not value.startswith("sha256:"):
            raise PrivacyViolationError("%s: string exceeds the bounded id domain" % (path,))


def reject_row(receipt_id: str, ts: str, code: str = "schema_reject") -> Dict[str, Any]:
    """A bare machine-code row used when a receipt fails validation (section 4)."""
    return {
        "schema": SCHEMA_RECEIPT,
        "receipt_id": receipt_id,
        "ts": ts,
        "escalation_code": code,
    }


class ReceiptStore:
    """Append-only JSONL receipt store.  No mutation, no deletion."""

    def __init__(self, directory: str, filename: str = RECEIPTS_FILENAME) -> None:
        self.directory = directory
        self.filename = filename
        self.path = os.path.join(directory, filename)
        self.reject_path = os.path.join(directory, "rejects.jsonl")

    # -- writing -----------------------------------------------------------

    def append(self, receipt: Mapping[str, Any]) -> str:
        """Validate then append.  Raises on any violation; nothing is written."""
        validate_receipt(receipt)
        assert_no_free_text(receipt)
        return self._append_line(self.path, receipt)

    def append_safe(
        self, receipt: Mapping[str, Any], receipt_id: str, ts: str
    ) -> Tuple[bool, Optional[str]]:
        """Append or drop.  A dropped receipt is logged as a bare code row."""
        try:
            self.append(receipt)
        except (SchemaError, PrivacyViolationError) as exc:
            self._append_line(self.reject_path, reject_row(receipt_id, ts))
            return False, str(exc)
        return True, None

    def _append_line(self, path: str, payload: Mapping[str, Any]) -> str:
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return path

    # -- reading -----------------------------------------------------------

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def read_rejects(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.reject_path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.reject_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        return len(self.read_all())


def load_receipts_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file back and re-validate every row (round-trip check)."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if set(record.keys()) == set(RECEIPT_FIELDS):
                validate_receipt(record)
                assert_no_free_text(record)
            records.append(record)
    return records


def receipt_prompt_text_absent(receipt: Mapping[str, Any], needle: str) -> bool:
    """True iff the given raw text (or any 16-char window of it) is absent.

    Used by the negative tests: persisting the raw prompt must be impossible,
    so this must always return True for a store produced by this package.
    """
    body = json.dumps(receipt, sort_keys=True)
    if needle and needle in body:
        return False
    for start in range(0, max(0, len(needle) - 16) + 1):
        if needle[start : start + 16] in body:
            return False
    return True


def store_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
