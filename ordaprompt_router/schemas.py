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

#: the closed comparison-surface domain (D8 / leo-provider-contract.md section 1).
#: `profile` is the phase-2 additive member: the ABC already expresses "ONE call per
#: surface, all candidates", so only this closed domain had to widen.  No signature
#: change anywhere: pre-profile callers keep passing "topic" / "session".
SURFACES = ("topic", "session", "profile")

#: profile-surface sentinel.  Eligibility is computed UPSTREAM (by the caller) and
#: arrives as the candidate set; the adapter never invents, widens or reorders it.
#: The sentinel is an ordinary candidate in the same batch, exactly like `novel`,
#: `ambiguous` and `new_session` -- it is what makes the abstain outcome explicit.
PROFILE_SENTINELS = ("no_suitable_profile",)
PROFILE_SENTINEL_ABSTAIN = PROFILE_SENTINELS[0]

#: closed domains for the whitelisted profile metadata (ids/labels/enums only).
PROFILE_SCOPES = ("global", "project", "session")
PROFILE_PRIVACY_CLASSES = ("public", "internal", "private", "restricted")

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
    #: additive (phase 2, D8): the explicit `no_suitable_profile` abstain candidate won
    #: the profile surface.  A machine code, not prose; the domain stays closed.
    "profile_abstain_argmax",
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
#: the ONLY profile metadata that may ever leave the process (whitelist, D8 item 5):
#: id, scope, privacy class and labels.  Nothing else is carried, and the payload
#: builder is the single choke point that serialises it.
PROFILE_FIELDS = ("profile_id", "scope", "privacy_class", "labels")
CANDIDATE_SET_FIELDS = (
    "schema",
    "request_id",
    "topics",
    "topic_sentinels",
    "sessions",
    "session_sentinels",
    "profiles",
    "profile_sentinels",
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
class ProfileCandidate:
    """One ELIGIBLE profile on the `profile` surface (D8).

    Eligibility is decided UPSTREAM: the caller (Orda) computes which profiles may be
    considered for this request and passes exactly that set.  The adapter adds nothing,
    drops nothing and reorders nothing -- a reply that does not cover exactly this id
    set is a hard ``ProviderContractError``.

    ``labels`` are closed-domain id tokens (slugs), never display text.  ``scope`` and
    ``privacy_class`` are closed enums, so no free text can be validated into a request
    payload or a receipt.
    """

    profile_id: str
    scope: str = "project"
    privacy_class: str = "private"
    labels: Sequence[str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "scope": self.scope,
            "privacy_class": self.privacy_class,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, obj: Any, path: str) -> "ProfileCandidate":
        data = _closed(obj, PROFILE_FIELDS, path)
        labels = data.get("labels", [])
        _need(isinstance(labels, list), path + ".labels", "expected a list")
        for index, label in enumerate(labels):
            _need(
                isinstance(label, str) and bool(SLUG_RE.match(label)),
                "%s.labels[%d]" % (path, index),
                "expected a label slug outside the closed id domain",
            )
        _need(
            len(set(labels)) == len(labels),
            path + ".labels",
            "duplicate labels",
        )
        return cls(
            profile_id=_str(data, "profile_id", path, SLUG_RE),
            scope=_enum(data, "scope", PROFILE_SCOPES, path),
            privacy_class=_enum(data, "privacy_class", PROFILE_PRIVACY_CLASSES, path),
            labels=tuple(labels),
        )


@dataclass(frozen=True)
class CandidateSet:
    request_id: str
    topics: Sequence[TopicCandidate]
    sessions: Sequence[SessionCandidate] = ()
    topic_sentinels: Sequence[str] = TOPIC_SENTINELS
    session_sentinels: Sequence[str] = SESSION_SENTINELS
    #: phase-2 additive block (D8).  Absent/empty means "no profile surface": the
    #: pre-profile path is byte-identical because nothing profile-related is called,
    #: serialised or emitted.
    profiles: Sequence[ProfileCandidate] = ()
    profile_sentinels: Sequence[str] = PROFILE_SENTINELS
    schema: str = SCHEMA_CANDIDATE_SET

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "topics": [t.to_dict() for t in self.topics],
            "topic_sentinels": list(self.topic_sentinels),
            "sessions": [s.to_dict() for s in self.sessions],
            "session_sentinels": list(self.session_sentinels),
            "profiles": [p.to_dict() for p in self.profiles],
            "profile_sentinels": list(self.profile_sentinels),
        }

    # -- candidate id views -------------------------------------------------

    def topic_candidate_ids(self) -> List[str]:
        """ALL topic candidates including sentinels, sentinel-last."""
        return [t.topic_id for t in self.topics] + list(self.topic_sentinels)

    def session_candidate_ids(self) -> List[str]:
        return [s.session_id for s in self.sessions] + list(self.session_sentinels)

    def profile_candidate_ids(self) -> List[str]:
        """ALL eligible profiles, in the caller's order, sentinel LAST.

        The `no_suitable_profile` sentinel is unconditionally in this set, so the
        explicit abstain candidate is always on the ballot; it is never optional.
        """
        return [p.profile_id for p in self.profiles] + list(self.profile_sentinels)

    def candidate_ids_for(self, surface: str) -> List[str]:
        """The ONE place a surface maps to its candidate id list (D8).

        Both backends and the payload builder go through here, which is what makes
        "the adapter cannot widen, reorder, add or drop a candidate" structural: the
        list returned is derived from exactly one surface's candidates, in the order
        the caller declared them.  An unknown surface raises ``ValueError`` (the
        callers translate it into their own structured error).
        """
        if surface == "topic":
            return self.topic_candidate_ids()
        if surface == "session":
            return self.session_candidate_ids()
        if surface == "profile":
            return self.profile_candidate_ids()
        raise ValueError("unknown surface %r" % (surface,))

    def profile_index(self) -> Dict[str, ProfileCandidate]:
        return {p.profile_id: p for p in self.profiles}

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

        # -- phase-2 additive profile block (D8) ---------------------------
        # Absent == no profile surface == today's behaviour and today's receipts.
        profiles_raw = data.get("profiles", [])
        _need(isinstance(profiles_raw, list), path + ".profiles", "expected a list")
        profiles = [
            ProfileCandidate.from_dict(item, "%s.profiles[%d]" % (path, i))
            for i, item in enumerate(profiles_raw)
        ]
        pids = [p.profile_id for p in profiles]
        _need(len(set(pids)) == len(pids), path + ".profiles", "duplicate profile_id values")
        p_collisions = sorted(set(pids) & set(PROFILE_SENTINELS))
        _need(
            not p_collisions,
            path + ".profiles",
            "profile_id collides with the reserved sentinel: %s" % (p_collisions,),
        )
        p_sentinels = data.get("profile_sentinels")
        if p_sentinels is not None:
            _need(
                isinstance(p_sentinels, list) and tuple(p_sentinels) == PROFILE_SENTINELS,
                path + ".profile_sentinels",
                "expected exactly %s (the abstain sentinel is always a candidate)"
                % (list(PROFILE_SENTINELS),),
            )

        return cls(
            request_id=request_id,
            topics=tuple(topics),
            sessions=tuple(sessions),
            topic_sentinels=TOPIC_SENTINELS,
            session_sentinels=SESSION_SENTINELS,
            profiles=tuple(profiles),
            profile_sentinels=PROFILE_SENTINELS,
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
#: the profile surface is a comparative ranking surface like `topic`: raw + calibrated
#: only (no utility term, no cost/contamination pricing of its own).
PROFILE_SCORE_FIELDS = ("candidate", "raw", "calibrated")
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
    # phase-2 additive block (D8): present ONLY when the profile surface ran
    "profile_candidate_ids",
    "profile_scores",
    "profile_top",
    "profile_margin",
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


def _profile_score(obj: Any, path: str) -> Dict[str, Any]:
    data = _closed(obj, PROFILE_SCORE_FIELDS, path)
    return {
        "candidate": _str(data, "candidate", path, ID_TOKEN_RE),
        "raw": _float(data, "raw", path),
        "calibrated": _float(data, "calibrated", path),
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
    #: phase-2 additive profile block (D8).  Empty == the profile surface did not run
    #: (no profile candidates were declared), which is the pre-profile default.
    profile_candidate_ids: List[str] = field(default_factory=list)
    profile_scores: List[Dict[str, Any]] = field(default_factory=list)
    profile_top: Optional[Dict[str, Any]] = None
    profile_margin: float = 0.0
    schema: str = SCHEMA_DECISION

    def to_dict(self) -> Dict[str, Any]:
        out = {
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
        if self.profile_candidate_ids:
            out["profile_candidate_ids"] = list(self.profile_candidate_ids)
            out["profile_scores"] = [dict(s) for s in self.profile_scores]
            out["profile_top"] = dict(self.profile_top) if self.profile_top else None
            out["profile_margin"] = float(self.profile_margin)
        return out

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
        profile_ids = data.get("profile_candidate_ids", [])
        profile_scores_raw = data.get("profile_scores", [])
        profile_top = data.get("profile_top")
        _need(
            isinstance(profile_ids, list), path + ".profile_candidate_ids", "expected a list"
        )
        for index, item in enumerate(profile_ids):
            _need(
                is_id_token(item),
                "%s.profile_candidate_ids[%d]" % (path, index),
                "value %r is outside the closed id/enum/hash domain" % (item,),
            )
        _need(isinstance(profile_scores_raw, list), path + ".profile_scores", "expected a list")
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
            profile_candidate_ids=list(profile_ids),
            profile_scores=[
                _profile_score(s, "%s.profile_scores[%d]" % (path, i))
                for i, s in enumerate(profile_scores_raw)
            ],
            profile_top=(
                _closed(profile_top, TOP_FIELDS, path + ".profile_top")
                if profile_top is not None
                else None
            ),
            profile_margin=_float(data, "profile_margin", path, -1.0, 1.0),
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
    # Optional, provider-lane additive field (D3): present ONLY when an
    # operator-declared fallback chain fired; absent on the default path, so
    # local-only receipts stay byte-identical to the earlier release.
    "fallback_from",
    # Optional, profile-lane additive block (D8): present ONLY when the `profile`
    # surface actually ran (i.e. the caller declared profile candidates).  Absent on
    # the default path, so local-only receipts stay byte-identical.
    "profile_candidate_ids",
    "profile_scores",
    "profile_top",
    "profile_margin",
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
    fallback_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a receipt dict from enumerated fields only (leo-arch.md 1.4).

    ``fallback_from`` (locked decision D3 of the provider lane) is emitted ONLY
    when an operator-declared fallback chain actually fired, so a receipt from the
    local-only path keeps exactly the original field set.
    """
    receipt = {
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
    if fallback_from is not None:
        receipt["fallback_from"] = fallback_from
    # D8: the profile block is emitted ONLY when the profile surface actually ran, so a
    # receipt from the (still default) topic/session-only path keeps exactly the
    # original field set and byte-identical content.
    if decision.profile_candidate_ids:
        receipt["profile_candidate_ids"] = list(decision.profile_candidate_ids)
        receipt["profile_scores"] = [dict(s) for s in decision.profile_scores]
        receipt["profile_top"] = dict(decision.profile_top) if decision.profile_top else None
        receipt["profile_margin"] = float(decision.profile_margin)
    return receipt


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
    fallback_from = data.get("fallback_from")
    if fallback_from is not None:
        _need(
            is_id_token(fallback_from),
            path + ".fallback_from",
            "expected a provider id token or null",
        )
    # phase-2 additive profile block (D8): validated only when present.
    if "profile_candidate_ids" in data:
        profile_ids = _receipt_string_list(data, "profile_candidate_ids", path)
        profile_scores = data.get("profile_scores")
        _need(isinstance(profile_scores, list), path + ".profile_scores", "expected a list")
        score_rows: List[Any] = profile_scores if isinstance(profile_scores, list) else []
        for index, score in enumerate(score_rows):
            _profile_score(score, "%s.profile_scores[%d]" % (path, index))
        scored = [row["candidate"] for row in score_rows]
        missing = [c for c in profile_ids if c not in scored]
        extra = sorted(c for c in scored if c not in set(profile_ids))
        _need(
            not missing and not extra,
            path + ".profile_scores",
            "profile scores must cover exactly profile_candidate_ids (missing=%s extra=%s)"
            % (missing[:8], extra[:8]),
        )
        profile_top = data.get("profile_top")
        _need(
            profile_top is None or isinstance(profile_top, dict),
            path + ".profile_top",
            "expected an object or null",
        )
        if profile_top is not None:
            _closed(profile_top, TOP_FIELDS, path + ".profile_top")
        _float(data, "profile_margin", path, -1.0, 1.0)
    return data
