"""Closed schemas for the OrdaPilot classifier router (leo-arch.md section 1).

Every schema here is CLOSED: unknown fields are rejected.  Strings are
constrained to closed id/enum/hash domains so that free text can never be
validated into a receipt (leo-arch.md sections 1.4 and 4, locked #4 and #8).

Stdlib only.  No I/O, no network, no Hermes imports.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Schema identifiers
# --------------------------------------------------------------------------

SCHEMA_REQUEST = "ordapilot.classification_request/1"
SCHEMA_CANDIDATE_SET = "ordapilot.candidate_set/1"
SCHEMA_DECISION = "ordapilot.routing_decision/1"
SCHEMA_RECEIPT = "ordapilot.routing_receipt/1"

# --------------------------------------------------------------------------
# Enumerations (closed domains)
# --------------------------------------------------------------------------

BANDS = ("automatic", "fallback_escalate", "abstain_or_new_session")
DECISIONS = ("topic_assign", "novel_propose", "session_reuse", "new_session", "escalate")
LABEL_STATES = ("provisional", "promoted")
CONTAMINATION_FLAGS = (
    "project_mismatch",
    "secret_presence",
    "topic_drift",
    "stale_context",
)
BACKENDS = ("synthetic", "openrouter", "openjev")

#: machine codes only -- never prose (leo-arch.md 1.4 field 24)
ESCALATION_CODES = (
    "below_min_score",
    "below_margin",
    "ambiguous_won",
    "calibration_missing",
    "backend_error",
    "schema_reject",
    "explicit_directive",
    "backend_disabled_by_policy",
    "new_session_argmax",
    "contamination_high",
    "below_min_utility",
    "taxonomy_rejected",
    "novel_below_threshold",
    "legacy_mode",
)

#: contamination weights, config-rev pinned (leo-arch.md section 3)
CONTAMINATION_WEIGHTS = {
    "project_mismatch": 0.7,
    "secret_presence": 0.5,
    "topic_drift": 0.4,
    "stale_context": 0.2,
}

# --------------------------------------------------------------------------
# Closed string domains
# --------------------------------------------------------------------------

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,39}$")
UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
#: generic machine id token: no whitespace, bounded length, closed alphabet
ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
#: label transition code, e.g. "topic-billing:promoted"
TRANSITION_RE = re.compile(
    r"^[a-z0-9][a-z0-9-]{2,39}:(provisional|promoted|demoted|archived)$"
)


class SchemaError(ValueError):
    """Raised whenever a payload violates a closed schema."""


class PrivacyViolationError(ValueError):
    """Raised when free text is found where only ids/hashes/enums are allowed.

    Raised by the receipt store and by the gated backend payload assertion --
    the two places where text could otherwise leak out of the process.
    """


# --------------------------------------------------------------------------
# Primitive validators
# --------------------------------------------------------------------------


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HASH_RE.match(value))


def is_iso_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.utcoffset() == _dt.timedelta(0)


def is_id_token(value: Any) -> bool:
    """Closed machine-id domain: hashes, uuids, semver, slugs, enum codes.

    Anything containing whitespace, an unexpected character class, or an
    unbounded length is rejected -- this is the structural half of the
    "no free text" invariant.
    """
    if not isinstance(value, str):
        return False
    if HASH_RE.match(value) or UUID4_RE.match(value) or SEMVER_RE.match(value):
        return True
    if SLUG_RE.match(value) or TRANSITION_RE.match(value):
        return True
    return bool(ID_TOKEN_RE.match(value))


def _need(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise SchemaError("%s: %s" % (path, message))


def _closed(obj: Any, allowed: Sequence[str], path: str) -> Dict[str, Any]:
    _need(isinstance(obj, dict), path, "expected an object, got %s" % type(obj).__name__)
    unknown = sorted(set(obj.keys()) - set(allowed))
    _need(not unknown, path, "unknown field(s) rejected by closed schema: %s" % (unknown,))
    return obj


def _str(obj: Dict[str, Any], key: str, path: str, pattern: Optional[re.Pattern] = None) -> str:
    value = obj.get(key)
    _need(isinstance(value, str) and value != "", "%s.%s" % (path, key), "expected non-empty string")
    if pattern is not None:
        _need(
            bool(pattern.match(value)),
            "%s.%s" % (path, key),
            "value %r is outside the closed domain %s" % (value, pattern.pattern),
        )
    return value


def _int(obj: Dict[str, Any], key: str, path: str, minimum: int = 0) -> int:
    value = obj.get(key)
    _need(isinstance(value, int) and not isinstance(value, bool), "%s.%s" % (path, key), "expected int")
    _need(value >= minimum, "%s.%s" % (path, key), "expected int >= %d" % minimum)
    return value


def _float(obj: Dict[str, Any], key: str, path: str, lo: float = 0.0, hi: float = 1.0) -> float:
    value = obj.get(key)
    _need(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "%s.%s" % (path, key),
        "expected number",
    )
    value = float(value)
    _need(lo <= value <= hi, "%s.%s" % (path, key), "expected %s <= value <= %s" % (lo, hi))
    return value


def _enum(obj: Dict[str, Any], key: str, allowed: Sequence[str], path: str) -> str:
    value = obj.get(key)
    _need(value in allowed, "%s.%s" % (path, key), "expected one of %s, got %r" % (tuple(allowed), value))
    return value


# --------------------------------------------------------------------------
# 1.1 ClassificationRequest
# --------------------------------------------------------------------------

REQUEST_FIELDS = (
    "schema",
    "request_id",
    "ts",
    "prompt_hash",
    "context_hash",
    "context_token_estimate",
    "project_id_hash",
    "config_rev",
    "explicit_directive",
)

DIRECTIVE_RE = re.compile(r"^(direct|handoff):[a-z0-9][a-z0-9-]{2,39}$")


@dataclass(frozen=True)
class ClassificationRequest:
    request_id: str
    ts: str
    prompt_hash: str
    project_id_hash: str
    config_rev: str
    context_hash: Optional[str] = None
    context_token_estimate: int = 0
    explicit_directive: Optional[str] = None
    schema: str = SCHEMA_REQUEST

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "ts": self.ts,
            "prompt_hash": self.prompt_hash,
            "context_hash": self.context_hash,
            "context_token_estimate": self.context_token_estimate,
            "project_id_hash": self.project_id_hash,
            "config_rev": self.config_rev,
            "explicit_directive": self.explicit_directive,
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "ClassificationRequest":
        path = "ClassificationRequest"
        data = _closed(obj, REQUEST_FIELDS, path)
        _need(data.get("schema") == SCHEMA_REQUEST, path + ".schema", "unexpected schema id")
        request_id = _str(data, "request_id", path, UUID4_RE)
        ts = _str(data, "ts", path)
        _need(is_iso_utc(ts), path + ".ts", "expected ISO-8601 UTC timestamp")
        prompt_hash = _str(data, "prompt_hash", path, HASH_RE)
        context_hash = data.get("context_hash")
        if context_hash is not None:
            _need(is_hash(context_hash), path + ".context_hash", "expected sha256 hash or null")
        directive = data.get("explicit_directive")
        if directive is not None:
            _need(
                isinstance(directive, str) and bool(DIRECTIVE_RE.match(directive)),
                path + ".explicit_directive",
                "expected 'direct:<slug>' / 'handoff:<slug>' or null",
            )
        return cls(
            request_id=request_id,
            ts=ts,
            prompt_hash=prompt_hash,
            project_id_hash=_str(data, "project_id_hash", path, HASH_RE),
            config_rev=_str(data, "config_rev", path, SEMVER_RE),
            context_hash=context_hash,
            context_token_estimate=_int(data, "context_token_estimate", path),
            explicit_directive=directive,
        )


# --------------------------------------------------------------------------
# 1.2 CandidateSet
# --------------------------------------------------------------------------

TOPIC_FIELDS = ("topic_id", "label_state", "label_age_days", "evidence_count")
SESSION_FIELDS = (
    "session_id",
    "project_id_hash",
    "context_token_cost",
    "idle_days",
    "contamination_flags",
)
CANDIDATE_SET_FIELDS = (
    "schema",
    "request_id",
    "topics",
    "topic_sentinels",
    "sessions",
    "session_sentinels",
)

TOPIC_SENTINELS = ("novel", "ambiguous")
SESSION_SENTINELS = ("new_session",)


@dataclass(frozen=True)
class TopicCandidate:
    topic_id: str
    label_state: str = "promoted"
    label_age_days: int = 0
    evidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "label_state": self.label_state,
            "label_age_days": self.label_age_days,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, obj: Any, path: str) -> "TopicCandidate":
        data = _closed(obj, TOPIC_FIELDS, path)
        return cls(
            topic_id=_str(data, "topic_id", path, SLUG_RE),
            label_state=_enum(data, "label_state", LABEL_STATES, path),
            label_age_days=_int(data, "label_age_days", path),
            evidence_count=_int(data, "evidence_count", path),
        )


@dataclass(frozen=True)
class SessionCandidate:
    session_id: str
    project_id_hash: str
    context_token_cost: int = 0
    idle_days: int = 0
    contamination_flags: Sequence[str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id_hash": self.project_id_hash,
            "context_token_cost": self.context_token_cost,
            "idle_days": self.idle_days,
            "contamination_flags": list(self.contamination_flags),
        }

    @classmethod
    def from_dict(cls, obj: Any, path: str) -> "SessionCandidate":
        data = _closed(obj, SESSION_FIELDS, path)
        flags = data.get("contamination_flags")
        _need(isinstance(flags, list), path + ".contamination_flags", "expected a list")
        for index, flag in enumerate(flags):
            _need(
                flag in CONTAMINATION_FLAGS,
                "%s.contamination_flags[%d]" % (path, index),
                "unknown contamination flag %r" % (flag,),
            )
        _need(
            len(set(flags)) == len(flags),
            path + ".contamination_flags",
            "duplicate contamination flags",
        )
        return cls(
            session_id=_str(data, "session_id", path, ID_TOKEN_RE),
            project_id_hash=_str(data, "project_id_hash", path, HASH_RE),
            context_token_cost=_int(data, "context_token_cost", path),
            idle_days=_int(data, "idle_days", path),
            contamination_flags=tuple(flags),
        )


@dataclass(frozen=True)
class CandidateSet:
    request_id: str
    topics: Sequence[TopicCandidate]
    sessions: Sequence[SessionCandidate] = ()
    topic_sentinels: Sequence[str] = TOPIC_SENTINELS
    session_sentinels: Sequence[str] = SESSION_SENTINELS
    schema: str = SCHEMA_CANDIDATE_SET

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "topics": [t.to_dict() for t in self.topics],
            "topic_sentinels": list(self.topic_sentinels),
            "sessions": [s.to_dict() for s in self.sessions],
            "session_sentinels": list(self.session_sentinels),
        }

    # -- candidate id views -------------------------------------------------

    def topic_candidate_ids(self) -> List[str]:
        """ALL topic candidates including sentinels, sentinel-last."""
        return [t.topic_id for t in self.topics] + list(self.topic_sentinels)

    def session_candidate_ids(self) -> List[str]:
        return [s.session_id for s in self.sessions] + list(self.session_sentinels)

    def topic_index(self) -> Dict[str, TopicCandidate]:
        return {t.topic_id: t for t in self.topics}

    def session_index(self) -> Dict[str, SessionCandidate]:
        return {s.session_id: s for s in self.sessions}

    @classmethod
    def from_dict(cls, obj: Any) -> "CandidateSet":
        path = "CandidateSet"
        data = _closed(obj, CANDIDATE_SET_FIELDS, path)
        _need(data.get("schema") == SCHEMA_CANDIDATE_SET, path + ".schema", "unexpected schema id")
        request_id = _str(data, "request_id", path, UUID4_RE)

        topics_raw = data.get("topics")
        _need(isinstance(topics_raw, list), path + ".topics", "expected a list")
        topics = [
            TopicCandidate.from_dict(item, "%s.topics[%d]" % (path, i))
            for i, item in enumerate(topics_raw)
        ]
        ids = [t.topic_id for t in topics]
        _need(len(set(ids)) == len(ids), path + ".topics", "duplicate topic_id values")
        collisions = sorted(set(ids) & set(TOPIC_SENTINELS))
        _need(
            not collisions,
            path + ".topics",
            "topic_id collides with a reserved sentinel: %s" % (collisions,),
        )

        sentinels = data.get("topic_sentinels")
        _need(
            isinstance(sentinels, list) and tuple(sentinels) == TOPIC_SENTINELS,
            path + ".topic_sentinels",
            "expected exactly %s (sentinels are ordinary candidates, never optional)"
            % (list(TOPIC_SENTINELS),),
        )

        sessions_raw = data.get("sessions")
        _need(isinstance(sessions_raw, list), path + ".sessions", "expected a list")
        sessions = [
            SessionCandidate.from_dict(item, "%s.sessions[%d]" % (path, i))
            for i, item in enumerate(sessions_raw)
        ]
        sids = [s.session_id for s in sessions]
        _need(len(set(sids)) == len(sids), path + ".sessions", "duplicate session_id values")
        s_collisions = sorted(set(sids) & set(SESSION_SENTINELS))
        _need(
            not s_collisions,
            path + ".sessions",
            "session_id collides with a reserved sentinel: %s" % (s_collisions,),
        )

        s_sentinels = data.get("session_sentinels")
        _need(
            isinstance(s_sentinels, list) and tuple(s_sentinels) == SESSION_SENTINELS,
            path + ".session_sentinels",
            "expected exactly %s" % (list(SESSION_SENTINELS),),
        )

        return cls(
            request_id=request_id,
            topics=tuple(topics),
            sessions=tuple(sessions),
            topic_sentinels=TOPIC_SENTINELS,
            session_sentinels=SESSION_SENTINELS,
        )


# --------------------------------------------------------------------------
# 1.3 RoutingDecision
# --------------------------------------------------------------------------

THRESHOLD_FIELDS = (
    "tau_topic",
    "mu_topic",
    "tau_novel",
    "mu_novel",
    "tau_sess_sim",
    "tau_sess_util",
    "mu_sess",
)

TOPIC_SCORE_FIELDS = ("candidate", "raw", "calibrated")
SESSION_SCORE_FIELDS = ("candidate", "raw", "calibrated", "utility")
TOP_FIELDS = ("candidate", "calibrated")
SESSION_TOP_FIELDS = ("candidate", "utility")

DECISION_FIELDS = (
    "schema",
    "request_id",
    "band",
    "decision",
    "topic_scores",
    "session_scores",
    "topic_top",
    "topic_second",
    "topic_margin",
    "session_top",
    "session_margin",
    "thresholds_used",
    "calibration_model_id",
    "contamination_score",
    "token_cost_estimate",
)


def _topic_score(obj: Any, path: str) -> Dict[str, Any]:
    data = _closed(obj, TOPIC_SCORE_FIELDS, path)
    return {
        "candidate": _str(data, "candidate", path, ID_TOKEN_RE),
        "raw": _float(data, "raw", path),
        "calibrated": _float(data, "calibrated", path),
    }


def _session_score(obj: Any, path: str) -> Dict[str, Any]:
    data = _closed(obj, SESSION_SCORE_FIELDS, path)
    return {
        "candidate": _str(data, "candidate", path, ID_TOKEN_RE),
        "raw": _float(data, "raw", path),
        "calibrated": _float(data, "calibrated", path),
        # utility is not bounded by [0,1]: U = c - 0.30*C - 0.50*K
        "utility": _float(data, "utility", path, -1.0, 1.0),
    }


@dataclass
class RoutingDecision:
    request_id: str
    band: str
    decision: str
    topic_scores: List[Dict[str, Any]] = field(default_factory=list)
    session_scores: List[Dict[str, Any]] = field(default_factory=list)
    topic_top: Optional[Dict[str, Any]] = None
    topic_second: Optional[Dict[str, Any]] = None
    topic_margin: float = 0.0
    session_top: Optional[Dict[str, Any]] = None
    session_margin: float = 0.0
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    calibration_model_id: str = "none"
    contamination_score: float = 0.0
    token_cost_estimate: int = 0
    schema: str = SCHEMA_DECISION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "band": self.band,
            "decision": self.decision,
            "topic_scores": [dict(s) for s in self.topic_scores],
            "session_scores": [dict(s) for s in self.session_scores],
            "topic_top": dict(self.topic_top) if self.topic_top else None,
            "topic_second": dict(self.topic_second) if self.topic_second else None,
            "topic_margin": self.topic_margin,
            "session_top": dict(self.session_top) if self.session_top else None,
            "session_margin": self.session_margin,
            "thresholds_used": dict(self.thresholds_used),
            "calibration_model_id": self.calibration_model_id,
            "contamination_score": self.contamination_score,
            "token_cost_estimate": self.token_cost_estimate,
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "RoutingDecision":
        path = "RoutingDecision"
        data = _closed(obj, DECISION_FIELDS, path)
        _need(data.get("schema") == SCHEMA_DECISION, path + ".schema", "unexpected schema id")
        topic_scores = data.get("topic_scores")
        session_scores = data.get("session_scores")
        _need(isinstance(topic_scores, list), path + ".topic_scores", "expected a list")
        _need(isinstance(session_scores, list), path + ".session_scores", "expected a list")
        thresholds = _closed(data.get("thresholds_used"), THRESHOLD_FIELDS, path + ".thresholds_used")
        topic_top = data.get("topic_top")
        topic_second = data.get("topic_second")
        session_top = data.get("session_top")
        return cls(
            request_id=_str(data, "request_id", path, UUID4_RE),
            band=_enum(data, "band", BANDS, path),
            decision=_enum(data, "decision", DECISIONS, path),
            topic_scores=[_topic_score(s, "%s.topic_scores[%d]" % (path, i)) for i, s in enumerate(topic_scores)],
            session_scores=[
                _session_score(s, "%s.session_scores[%d]" % (path, i)) for i, s in enumerate(session_scores)
            ],
            topic_top=(
                _closed(topic_top, TOP_FIELDS, path + ".topic_top") if topic_top is not None else None
            ),
            topic_second=(
                _closed(topic_second, TOP_FIELDS, path + ".topic_second") if topic_second is not None else None
            ),
            topic_margin=_float(data, "topic_margin", path, -1.0, 1.0),
            session_top=(
                _closed(session_top, SESSION_TOP_FIELDS, path + ".session_top")
                if session_top is not None
                else None
            ),
            session_margin=_float(data, "session_margin", path, -1.0, 1.0),
            thresholds_used={k: float(v) for k, v in thresholds.items()},
            calibration_model_id=_str(data, "calibration_model_id", path, ID_TOKEN_RE),
            contamination_score=_float(data, "contamination_score", path),
            token_cost_estimate=_int(data, "token_cost_estimate", path),
        )


# --------------------------------------------------------------------------
# 1.4 RoutingReceipt -- the 24 enumerated fields, nothing else
# --------------------------------------------------------------------------

RECEIPT_FIELDS = (
    "schema",
    "receipt_id",
    "ts",
    "router_version",
    "config_rev",
    "calibration_model_id",
    "prompt_hash",
    "context_hash",
    "project_id_hash",
    "topic_candidate_ids",
    "session_candidate_ids",
    "topic_scores",
    "session_scores",
    "topic_margin",
    "session_margin",
    "band",
    "decision",
    "thresholds_used",
    "contamination_score",
    "token_cost_estimate",
    "backend_used",
    "backend_latency_ms",
    "label_state_touched",
    "escalation_code",
)

ROUTER_VERSION = "1.0.0"
CALIBRATION_MODEL_ID_RE = re.compile(r"^(none|platt-v[0-9]{1,3})$")


def _receipt_string_list(obj: Dict[str, Any], key: str, path: str) -> List[str]:
    value = obj.get(key)
    _need(isinstance(value, list), "%s.%s" % (path, key), "expected a list of strings")
    for index, item in enumerate(value):
        _need(
            is_id_token(item),
            "%s.%s[%d]" % (path, key, index),
            "value %r is outside the closed id/enum/hash domain" % (item,),
        )
    return list(value)


def build_receipt(
    decision: RoutingDecision,
    candidates: CandidateSet,
    request: ClassificationRequest,
    backend_used: str,
    backend_latency_ms: int,
    receipt_id: str,
    ts: str,
    escalation_code: Optional[str] = None,
    label_state_touched: Optional[Sequence[str]] = None,
    router_version: str = ROUTER_VERSION,
) -> Dict[str, Any]:
    """Assemble a receipt dict from enumerated fields only (leo-arch.md 1.4)."""
    return {
        "schema": SCHEMA_RECEIPT,
        "receipt_id": receipt_id,
        "ts": ts,
        "router_version": router_version,
        "config_rev": request.config_rev,
        "calibration_model_id": decision.calibration_model_id,
        "prompt_hash": request.prompt_hash,
        "context_hash": request.context_hash,
        "project_id_hash": request.project_id_hash,
        "topic_candidate_ids": list(candidates.topic_candidate_ids()),
        "session_candidate_ids": list(candidates.session_candidate_ids()),
        "topic_scores": [dict(s) for s in decision.topic_scores],
        "session_scores": [dict(s) for s in decision.session_scores],
        "topic_margin": float(decision.topic_margin),
        "session_margin": float(decision.session_margin),
        "band": decision.band,
        "decision": decision.decision,
        "thresholds_used": dict(decision.thresholds_used),
        "contamination_score": float(decision.contamination_score),
        "token_cost_estimate": int(decision.token_cost_estimate),
        "backend_used": backend_used,
        "backend_latency_ms": int(backend_latency_ms),
        "label_state_touched": list(label_state_touched) if label_state_touched else None,
        "escalation_code": escalation_code,
    }


def validate_receipt(obj: Any) -> Dict[str, Any]:
    """Structural validation of a RoutingReceipt (closed fields + domains)."""
    path = "RoutingReceipt"
    data = _closed(obj, RECEIPT_FIELDS, path)
    _need(data.get("schema") == SCHEMA_RECEIPT, path + ".schema", "unexpected schema id")
    _str(data, "receipt_id", path, UUID4_RE)
    ts = _str(data, "ts", path)
    _need(is_iso_utc(ts), path + ".ts", "expected ISO-8601 UTC timestamp")
    _str(data, "router_version", path, SEMVER_RE)
    _str(data, "config_rev", path, SEMVER_RE)
    _str(data, "calibration_model_id", path, CALIBRATION_MODEL_ID_RE)
    _str(data, "prompt_hash", path, HASH_RE)
    context_hash = data.get("context_hash")
    if context_hash is not None:
        _need(is_hash(context_hash), path + ".context_hash", "expected sha256 hash or null")
    _str(data, "project_id_hash", path, HASH_RE)
    _receipt_string_list(data, "topic_candidate_ids", path)
    _receipt_string_list(data, "session_candidate_ids", path)
    topic_scores = data.get("topic_scores")
    session_scores = data.get("session_scores")
    _need(isinstance(topic_scores, list), path + ".topic_scores", "expected a list")
    _need(isinstance(session_scores, list), path + ".session_scores", "expected a list")
    for index, score in enumerate(topic_scores):
        _topic_score(score, "%s.topic_scores[%d]" % (path, index))
    for index, score in enumerate(session_scores):
        _session_score(score, "%s.session_scores[%d]" % (path, index))
    _float(data, "topic_margin", path, -1.0, 1.0)
    _float(data, "session_margin", path, -1.0, 1.0)
    _enum(data, "band", BANDS, path)
    _enum(data, "decision", DECISIONS, path)
    thresholds = _closed(data.get("thresholds_used"), THRESHOLD_FIELDS, path + ".thresholds_used")
    for key in THRESHOLD_FIELDS:
        _float(thresholds, key, path + ".thresholds_used", 0.0, 1.0)
    _float(data, "contamination_score", path)
    _int(data, "token_cost_estimate", path)
    _enum(data, "backend_used", BACKENDS, path)
    _int(data, "backend_latency_ms", path)
    touched = data.get("label_state_touched")
    if touched is not None:
        _need(isinstance(touched, list), path + ".label_state_touched", "expected a list or null")
        for index, item in enumerate(touched):
            _need(
                isinstance(item, str) and bool(TRANSITION_RE.match(item)),
                "%s.label_state_touched[%d]" % (path, index),
                "expected '<topic-slug>:<transition>' transition code",
            )
    code = data.get("escalation_code")
    if code is not None:
        _enum(data, "escalation_code", ESCALATION_CODES, path)
    return data
