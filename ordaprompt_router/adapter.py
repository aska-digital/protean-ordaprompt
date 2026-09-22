"""Backend adapters for the OrdaPilot classifier router (leo-arch.md section 2).

- :class:`ClassificationBackend` -- the batch-comparison interface (ONE call per
  surface, all candidates including sentinels).
- :class:`SyntheticBackend` -- deterministic, offline, no network.  Used for the
  eval harness and as the scoring source whenever external backends are off.
- :class:`OpenRouterBackend` -- DEFAULT OFF.  Raises ``DisabledByPolicy`` unless
  explicitly enabled by config flag, and even when enabled it is gated to
  novel-topic taxonomy proposals only and may send hashes + candidate labels
  only, never raw prompts.

Hosted TypeSafe Jev is NO-GO (leo-arch.md section 2) and is deliberately absent.

Stdlib only.  No network calls are made anywhere in this module: the optional
transport seam exists so tests can assert the payload shape offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas import (
    SLUG_RE,
    CandidateSet,
    PrivacyViolationError,
)

# --------------------------------------------------------------------------
# Redaction / hashing helpers
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")

#: closed domain for model identifiers in a gated payload
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,79}$")
PAYLOAD_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,79}$")


def canonicalize_redacted(text: str) -> str:
    """Canonicalize text for hashing: collapse whitespace, strip, drop control chars.

    Redaction happens *before* hashing so that two differently-spaced prompts
    hash identically.  The canonical text itself is never persisted by this
    package (receipts store the hash only).
    """
    if not isinstance(text, str):
        raise TypeError("canonicalize_redacted expects str")
    cleaned = "".join(ch for ch in text if ch == "\n" or ch >= " ")
    return _WS_RE.sub(" ", cleaned).strip()


def sha256_hex(text: str) -> str:
    """Return ``sha256:<hex>`` of the canonicalized redacted input."""
    return "sha256:" + hashlib.sha256(canonicalize_redacted(text).encode("utf-8")).hexdigest()


def hash_of(text: str) -> str:
    """Raw digest (no prefix) -- used for synthetic id derivation."""
    return hashlib.sha256(canonicalize_redacted(text).encode("utf-8")).hexdigest()


class DisabledByPolicy(RuntimeError):
    """Raised when a policy-gated backend is used without an explicit opt-in."""


class BackendError(RuntimeError):
    """Backend failure: timeout, schema reject, rate limit, missing transport."""


# --------------------------------------------------------------------------
# Request handle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestHandle:
    """What a backend is allowed to see: hashes and ids only, never raw text."""

    request_id: str
    prompt_hash: str
    project_id_hash: str
    config_rev: str
    context_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "project_id_hash": self.project_id_hash,
            "config_rev": self.config_rev,
            "context_hash": self.context_hash,
        }


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


class ClassificationBackend(ABC):
    """Batch comparative classifier surface (leo-arch.md section 2)."""

    name: str = "abstract"

    def __init__(self) -> None:
        #: per-surface count of batch_score calls -- proves the ONE-call rule
        self.batch_calls: Dict[str, int] = {"topic": 0, "session": 0}

    @abstractmethod
    def batch_score(
        self,
        request_handle: RequestHandle,
        candidates: CandidateSet,
        surface: str,
    ) -> List[Dict[str, Any]]:
        """ONE call scoring ALL candidates for ``surface``, sentinels included."""

    @abstractmethod
    def propose_taxonomy(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """High-confidence novel-topic taxonomy proposal, or None."""


# --------------------------------------------------------------------------
# Synthetic backend (deterministic, always available, offline)
# --------------------------------------------------------------------------


class SyntheticBackend(ClassificationBackend):
    """Deterministic hash scorer -- byte-identical across runs, zero network.

    ``raw = 0.5 + 0.5 * cos(vec(prompt_hash, surface), vec(candidate, surface))``
    with 8-dimensional hash-derived unit vectors, so the score is a pure
    function of the prompt hash and the candidate ids (leo-arch.md section 2).

    A ``fixture_table`` mapping ``(prompt_hash, surface) -> {candidate: raw}``
    can be supplied to replay recorded fixture scores verbatim (offline eval);
    any pair absent from the table falls back to the hash scorer.
    """

    name = "synthetic"
    DIMS = 16
    #: link stretch: cosine similarity is mapped through tanh so the synthetic
    #: score has a usable dynamic range instead of being squeezed near 0.5
    STRETCH = 3.0

    def __init__(self, fixture_table: Optional[Mapping[Tuple[str, str], Mapping[str, float]]] = None) -> None:
        super().__init__()
        self._fixture_table: Dict[Tuple[str, str], Dict[str, float]] = {}
        if fixture_table:
            for key, scores in fixture_table.items():
                self._fixture_table[(str(key[0]), str(key[1]))] = {
                    str(k): float(v) for k, v in scores.items()
                }
        self._fixture_hits = 0
        self._hash_calls = 0

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _vector(key: str, dims: int) -> List[float]:
        out: List[float] = []
        counter = 0
        while len(out) < dims:
            block = hashlib.sha256(("%s|%d" % (key, counter)).encode("utf-8")).digest()
            for offset in range(0, 32, 4):
                if len(out) >= dims:
                    break
                raw = int.from_bytes(block[offset : offset + 4], "big") / 4294967295.0
                out.append(raw * 2.0 - 1.0)
            counter += 1
        norm = math.sqrt(sum(value * value for value in out)) or 1.0
        return [value / norm for value in out]

    def pair_score(self, prompt_hash: str, candidate: str, surface: str) -> float:
        """Deterministic raw score in [0, 1] for one (prompt, candidate) pair."""
        prompt_vec = self._vector("%s|%s" % (prompt_hash, surface), self.DIMS)
        cand_vec = self._vector("candidate|%s|%s" % (surface, candidate), self.DIMS)
        cosine = sum(a * b for a, b in zip(prompt_vec, cand_vec))
        self._hash_calls += 1
        return min(1.0, max(0.0, 0.5 + 0.5 * math.tanh(self.STRETCH * cosine)))

    def raw_for(self, prompt_hash: str, candidate: str, surface: str) -> float:
        table = self._fixture_table.get((prompt_hash, surface))
        if table is not None and candidate in table:
            self._fixture_hits += 1
            return min(1.0, max(0.0, float(table[candidate])))
        return self.pair_score(prompt_hash, candidate, surface)

    def stats(self) -> Dict[str, Any]:
        return {
            "fixture_hits": self._fixture_hits,
            "hash_scored": self._hash_calls,
            "batch_calls": dict(self.batch_calls),
        }

    # -- interface ---------------------------------------------------------

    def batch_score(
        self,
        request_handle: RequestHandle,
        candidates: CandidateSet,
        surface: str,
    ) -> List[Dict[str, Any]]:
        if surface == "topic":
            ids = candidates.topic_candidate_ids()
        elif surface == "session":
            ids = candidates.session_candidate_ids()
        else:
            raise BackendError("unknown surface %r" % (surface,))
        self.batch_calls[surface] = self.batch_calls.get(surface, 0) + 1
        return [
            {"candidate": candidate, "raw": self.raw_for(request_handle.prompt_hash, candidate, surface)}
            for candidate in ids
        ]

    def propose_taxonomy(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """The deterministic offline backend never invents taxonomy labels."""
        return None


# --------------------------------------------------------------------------
# OpenRouter backend (DEFAULT OFF, gated, hash-only payloads)
# --------------------------------------------------------------------------


def assert_hash_only_payload(payload: Any, path: str = "payload") -> None:
    """Reject any payload string that is not a hash / id token / model id.

    This is the adapter-side half of the "never send raw prompts" invariant: a
    caller cannot smuggle free text (or a prompt excerpt) into the gated call.
    """
    if isinstance(payload, str):
        if not PAYLOAD_TOKEN_RE.match(payload):
            raise PrivacyViolationError(
                "%s: %r is outside the closed hash/id domain (possible free text)" % (path, payload)
            )
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_hash_only_payload(key, "%s.<key>" % path)
            assert_hash_only_payload(value, "%s.%s" % (path, key))
        return
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_hash_only_payload(item, "%s[%d]" % (path, index))
        return
    if isinstance(payload, (int, float, bool)) or payload is None:
        return
    raise PrivacyViolationError("%s: unsupported payload node type %s" % (path, type(payload).__name__))


class OpenRouterBackend(ClassificationBackend):
    """Flag-gated OpenRouter adapter.  Default OFF; hashes/labels only.

    * ``enabled=False`` (default) -- every entry point raises ``DisabledByPolicy``.
    * ``enabled=True`` -- routine ``batch_score`` still raises ``BackendError``
      (the adapter is gated to novel-topic taxonomy proposals only, section 2)
      and ``propose_taxonomy`` builds a hash-only payload, asserts it, and calls
      the injected transport.  No transport is bound in this lane, so no network
      call is ever performed; the seam exists for offline payload assertions.
    """

    name = "openrouter"

    def __init__(
        self,
        enabled: bool = False,
        model: str = "openai/gpt-4o-mini",
        transport: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        min_confidence: float = 0.90,
    ) -> None:
        super().__init__()
        self._enabled = bool(enabled)
        self.model = model
        self._transport = transport
        self.min_confidence = float(min_confidence)
        self.last_payload: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise DisabledByPolicy(
                "openrouter backend disabled by policy "
                "(router.backends.openrouter.enabled=false); refusing to proceed"
            )

    def batch_score(
        self,
        request_handle: RequestHandle,
        candidates: CandidateSet,
        surface: str,
    ) -> List[Dict[str, Any]]:
        self._require_enabled()
        raise BackendError(
            "openrouter batch scoring is not permitted: the adapter is gated to "
            "high-confidence novel-topic taxonomy proposals only (section 2)"
        )

    def build_proposal_payload(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Dict[str, Any]:
        """Assemble the gated payload: hashes, ids and labels -- no text at all.

        Delta note: leo-arch.md section 6.2 optionally allows the redacted
        canonical text of the single triggering prompt; this implementation is
        the stricter subset (stage-4 brief: "only hashes/candidate labels").
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "propose-taxonomy",
                    "prompt_hash": request_handle.prompt_hash,
                    "context_hash": request_handle.context_hash or "none",
                    "project_id_hash": request_handle.project_id_hash,
                    "exemplar_hashes": list(exemplar_hashes),
                }
            ],
        }
        assert_hash_only_payload(payload)
        return payload

    def propose_taxonomy(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        self._require_enabled()
        payload = self.build_proposal_payload(request_handle, exemplar_hashes)
        self.last_payload = payload
        if self._transport is None:
            raise BackendError(
                "openrouter transport not bound: no network call is performed in this lane"
            )
        response = self._transport(payload)
        return self.validate_proposal(response)

    def validate_proposal(self, response: Any) -> Optional[Dict[str, Any]]:
        """Accept a proposal only at >= min_confidence with a well-formed slug."""
        if not isinstance(response, Mapping):
            raise BackendError("proposal response is not an object")
        slug = response.get("slug")
        confidence = response.get("confidence")
        if not isinstance(slug, str) or not SLUG_RE.match(slug):
            return None
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None
        if float(confidence) < self.min_confidence:
            return None
        return {
            "slug": slug,
            "parent_slug": response.get("parent_slug"),
            "display_hash": response.get("display_hash"),
            "confidence": float(confidence),
        }
