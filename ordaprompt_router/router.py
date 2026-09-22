"""Router core: batch comparison, three-band policy, session utility, labels.

Pure decision logic -- no storage I/O, no network, no prompt text
(leo-arch.md section 0 module boundaries).

Locked constraints implemented here:
  #1 batch comparison  -- exactly ONE backend call per surface, sentinels are
                          ordinary candidates in the same batch;
  #2 three-band policy -- automatic requires a minimum calibrated score AND a
                          top-vs-second margin;
  #3 stricter session  -- reuse prices token cost and contamination risk and
                          always yields to ``new_session`` under ambiguity;
  #5 calibration gate  -- with no *validated* calibration the automatic band is
                          unreachable;
  #6 provisional labels-- promotion needs k=3 corroborations across >= 2
                          sessions and >= 2 days with no human override.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .adapter import (
    BackendError,
    ClassificationBackend,
    DisabledByPolicy,
    OpenRouterBackend,
    RequestHandle,
    SyntheticBackend,
)
from .providers import ProviderConfigError
from .schemas import (
    CONTAMINATION_WEIGHTS,
    PROFILE_SENTINEL_ABSTAIN,
    ROUTER_VERSION,
    SCHEMA_RECEIPT,
    CandidateSet,
    ClassificationRequest,
    RoutingDecision,
    build_receipt,
)

BAND_ORDER = {"automatic": 0, "fallback_escalate": 1, "abstain_or_new_session": 2}
BAND_CONSERVATIVE = ("automatic", "fallback_escalate", "abstain_or_new_session")

TOPIC_SENTINEL_NOVEL = "novel"
TOPIC_SENTINEL_AMBIGUOUS = "ambiguous"
SESSION_SENTINEL_NEW = "new_session"
#: the profile surface's explicit abstain candidate (see schemas.PROFILE_SENTINELS),
#: re-exported here under the name the router uses.
PROFILE_SENTINEL_NO_SUITABLE = PROFILE_SENTINEL_ABSTAIN


# --------------------------------------------------------------------------
# Thresholds (leo-arch.md section 3 defaults, config-rev pinned)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    tau_topic: float = 0.85
    mu_topic: float = 0.20
    tau_novel: float = 0.85
    mu_novel: float = 0.25
    tau_sess_sim: float = 0.90
    tau_sess_util: float = 0.55
    mu_sess: float = 0.25
    # band floor / pricing constants from section 3 (not part of thresholds_used)
    tau_low: float = 0.60
    k_block: float = 0.50
    w_cost: float = 0.30
    w_contam: float = 0.50
    new_session_utility: float = 0.35
    context_cost_norm: int = 8000

    def to_dict(self) -> Dict[str, float]:
        return {
            "tau_topic": self.tau_topic,
            "mu_topic": self.mu_topic,
            "tau_novel": self.tau_novel,
            "mu_novel": self.mu_novel,
            "tau_sess_sim": self.tau_sess_sim,
            "tau_sess_util": self.tau_sess_util,
            "mu_sess": self.mu_sess,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Thresholds":
        allowed = set(cls.__dataclass_fields__.keys())
        unknown = sorted(set(obj.keys()) - allowed)
        if unknown:
            raise ValueError("unknown threshold field(s): %s" % (unknown,))
        return cls(**{k: float(v) for k, v in obj.items()})


# --------------------------------------------------------------------------
# Calibration (leo-arch.md section 7)
# --------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class CalibrationModel:
    """Per-surface Platt scaling.  ``active`` is the automatic-band gate.

    ``active`` MUST only be set True once the section-7 unlock criteria pass on
    the HOLDOUT split (ECE <= 0.05, Brier <= 0.10, would-be-automatic precision
    >= 0.98 with >= 50 holdout samples in band).  With ``active=False`` the
    calibrator returns raw scores (identity) and nothing can be automatic.

    LOCKED consistency gate (leo-adapter-architecture.md section 6.4 / L23): a
    calibration state is valid only as the CONSISTENT pair
    ``model_id == "none"`` <-> ``active == False``.  Any other combination -- including
    the exact counterexample ``CalibrationModel(model_id="none", active=True)`` -- is
    rejected at LOAD time with ``calibration_state_invalid``.  There is no bare boolean
    anywhere that alone unlocks the ``automatic`` band: ``active`` additionally needs a
    fitted key that matches the running ``(kind, model, transform)`` and is checked by
    ``check_provider_key`` against the selected provider row.
    """

    model_id: str = "none"
    active: bool = False
    a_topic: float = 1.0
    b_topic: float = 0.0
    a_session: float = 1.0
    b_session: float = 0.0
    #: phase-2 additive (D8): the profile surface's Platt coefficients.  They ship at
    #: IDENTITY (1.0 / 0.0) because no profile calibration has been fit on real
    #: feedback data.  With identity coefficients `calibrate("profile", raw) == raw`,
    #: i.e. a provider's profile score stays a RANKING SIGNAL: it can order candidates
    #: and lower the band, but it can never manufacture a calibrated probability that
    #: unlocks `automatic` while `calibration_model_id` is "none".
    a_profile: float = 1.0
    b_profile: float = 0.0
    #: The FITTED KEY (section 6.3): a fitted parameter set is valid only for the
    #: ``(kind, model, transform)`` triple it was fitted on.  All three are None on an
    #: inactive model, because there is nothing fitted to bind.
    kind: Optional[str] = None
    model: Optional[str] = None
    transform: Optional[str] = None
    unlock_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.model_id == "none" and self.active:
            raise ProviderConfigError(
                "calibration_state_invalid",
                "calibration_model_id 'none' is mutually exclusive with active=true "
                "(L23: this exact pair is the unfitted calibration that must never unlock "
                "the automatic band)",
            )
        if self.model_id != "none" and not self.active:
            raise ProviderConfigError(
                "calibration_state_invalid",
                "calibration_model_id %r requires active=true (a fitted model that is not "
                "active is an ambiguous state, not a silent downgrade)" % (self.model_id,),
            )

    def fitted_key(self) -> Dict[str, Any]:
        """The ``(kind, model, transform)`` triple this parameter set was fitted on."""
        return {"kind": self.kind, "model": self.model, "transform": self.transform}

    def check_provider_key(self, row: Any) -> None:
        """Refuse an ACTIVE calibration that was not fitted for ``row`` (section 6.3/L16).

        ``row`` is a ``ProviderRow`` (or the plain ``{"kind","model","transform"}`` mapping of
        a built-in backend, see ``providers.local_backend_key``).  A mismatch never silently
        downgrades the band: it is a ``calibration_key_mismatch`` hard error, so the run stops
        instead of routing on parameters fitted elsewhere.
        """
        if not self.active:
            return
        actual = self._key_of(row)
        if not self.kind or not self.model:
            raise ProviderConfigError(
                "calibration_key_mismatch",
                "an active calibration must declare the (kind, model[, transform]) key it was "
                "fitted on; refit against real feedback for that exact triple",
            )
        if (self.kind, self.model, self.transform) != (
            actual["kind"],
            actual["model"],
            actual["transform"],
        ):
            raise ProviderConfigError(
                "calibration_key_mismatch",
                "calibration was fitted for %s but the selected provider is %s"
                % (self.fitted_key(), actual),
            )

    @staticmethod
    def _key_of(row: Any) -> Dict[str, Any]:
        if isinstance(row, Mapping):
            return {
                "kind": row.get("kind"),
                "model": row.get("model"),
                "transform": row.get("transform"),
            }
        return {
            "kind": getattr(row, "kind", None),
            "model": getattr(row, "model", None),
            "transform": getattr(row, "transform", None),
        }

    def calibrate(self, surface: str, raw: float) -> float:
        if not self.active:
            return min(1.0, max(0.0, float(raw)))
        if surface == "topic":
            a, b = self.a_topic, self.b_topic
        elif surface == "session":
            a, b = self.a_session, self.b_session
        elif surface == "profile":
            a, b = self.a_profile, self.b_profile
        else:
            raise ValueError("unknown surface %r" % (surface,))
        return min(1.0, max(0.0, _sigmoid(a * _logit(raw) + b)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "active": self.active,
            "a_topic": self.a_topic,
            "b_topic": self.b_topic,
            "a_session": self.a_session,
            "b_session": self.b_session,
            "a_profile": self.a_profile,
            "b_profile": self.b_profile,
            "kind": self.kind,
            "model": self.model,
            "transform": self.transform,
            "unlock_evidence": self.unlock_evidence,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "CalibrationModel":
        """Load path: the L23 consistency gate applies here, before any call is made."""
        return cls(
            model_id=str(obj.get("model_id", "none")),
            active=bool(obj.get("active", False)),
            a_topic=float(obj.get("a_topic", 1.0)),
            b_topic=float(obj.get("b_topic", 0.0)),
            a_session=float(obj.get("a_session", 1.0)),
            b_session=float(obj.get("b_session", 0.0)),
            a_profile=float(obj.get("a_profile", 1.0)),
            b_profile=float(obj.get("b_profile", 0.0)),
            kind=obj.get("kind"),
            model=obj.get("model"),
            transform=obj.get("transform"),
            unlock_evidence=obj.get("unlock_evidence"),
        )


# --------------------------------------------------------------------------
# Provisional labels and the promotion rule (leo-arch.md section 6)
# --------------------------------------------------------------------------


@dataclass
class LabelRecord:
    topic_id: str
    state: str = "provisional"
    corroborations: List[Tuple[str, str, float]] = field(default_factory=list)
    contradictions: int = 0
    overrides: int = 0
    demotions: int = 0


class LabelRegistry:
    """In-memory label states; emits ``<slug>:<transition>`` codes only."""

    def __init__(self, promote_k: int = 3, min_sessions: int = 2, min_days: int = 2) -> None:
        self.promote_k = promote_k
        self.min_sessions = min_sessions
        self.min_days = min_days
        self.labels: Dict[str, LabelRecord] = {}

    # -- state sync --------------------------------------------------------

    def observe(self, candidates: CandidateSet) -> None:
        for topic in candidates.topics:
            record = self.labels.get(topic.topic_id)
            if record is None:
                self.labels[topic.topic_id] = LabelRecord(topic.topic_id, topic.label_state)
            elif topic.label_state == "promoted" and record.state == "provisional":
                # a candidate set may assert promotion; the registry keeps its own
                # history and only promotes through the corroboration rule
                pass

    def create_provisional(self, topic_id: str) -> str:
        record = self.labels.get(topic_id)
        if record is None:
            self.labels[topic_id] = LabelRecord(topic_id, "provisional")
            return "%s:provisional" % topic_id
        return "%s:provisional" % topic_id

    # -- corroboration / promotion ----------------------------------------

    def register_corroboration(
        self,
        topic_id: str,
        session_id: str,
        day: str,
        calibrated: float,
        tau_topic: float,
    ) -> Optional[str]:
        record = self.labels.get(topic_id)
        if record is None:
            return None
        if calibrated < tau_topic:
            return None
        record.corroborations.append((session_id, day, float(calibrated)))
        if record.state != "provisional":
            return None
        sessions = {c[0] for c in record.corroborations}
        days = {c[1] for c in record.corroborations}
        promoted = (
            len(record.corroborations) >= self.promote_k
            and len(sessions) >= self.min_sessions
            and len(days) >= self.min_days
            and record.overrides == 0
        )
        if promoted:
            record.state = "promoted"
            return "%s:promoted" % topic_id
        return None

    def record_human_override(self, topic_id: str) -> str:
        record = self.labels.get(topic_id)
        if record is None:
            record = LabelRecord(topic_id, "provisional")
            self.labels[topic_id] = record
        record.overrides += 1
        if record.state == "promoted":
            record.state = "provisional"
            record.demotions += 1
            return "%s:demoted" % topic_id
        if not record.corroborations:
            record.state = "archived"
            return "%s:archived" % topic_id
        return "%s:provisional" % topic_id

    def record_contradiction(self, topic_id: str, demote_at: int = 2) -> Optional[str]:
        record = self.labels.get(topic_id)
        if record is None:
            return None
        record.contradictions += 1
        if record.state == "promoted" and record.contradictions >= demote_at:
            record.state = "provisional"
            record.demotions += 1
            return "%s:demoted" % topic_id
        return None

    def state_of(self, topic_id: str) -> Optional[str]:
        record = self.labels.get(topic_id)
        return record.state if record else None


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class RouterConfig:
    thresholds: Thresholds = field(default_factory=Thresholds)
    calibration: CalibrationModel = field(default_factory=CalibrationModel)
    backend: Optional[ClassificationBackend] = None
    openrouter_backend: Optional[OpenRouterBackend] = None
    legacy_fallback: bool = False
    label_registry: LabelRegistry = field(default_factory=LabelRegistry)
    router_version: str = ROUTER_VERSION

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = SyntheticBackend()

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "RouterConfig":
        """Build from a JSON config: {"router": {...}} (Hermes profile untouched)."""
        router = obj.get("router", obj) if isinstance(obj, Mapping) else {}
        unknown = sorted(set(router.keys()) - {"legacy_fallback", "backends", "thresholds", "calibration"})
        if unknown:
            raise ValueError("unknown router config field(s): %s" % (unknown,))
        backends = router.get("backends", {}) or {}
        openrouter_cfg = (backends.get("openrouter", {}) or {}) if isinstance(backends, Mapping) else {}
        unknown_b = sorted(set(backends.keys()) - {"openrouter"})
        if unknown_b:
            raise ValueError("unknown backend config field(s): %s" % (unknown_b,))
        openrouter = OpenRouterBackend(
            enabled=bool(openrouter_cfg.get("enabled", False)),
            model=str(openrouter_cfg.get("model", "openai/gpt-4o-mini")),
        )
        thresholds = (
            Thresholds.from_dict(router.get("thresholds", {}) or {}) if router.get("thresholds") else Thresholds()
        )
        calibration = (
            CalibrationModel.from_dict(router.get("calibration", {}) or {})
            if router.get("calibration")
            else CalibrationModel()
        )
        return cls(
            thresholds=thresholds,
            calibration=calibration,
            backend=SyntheticBackend(),
            openrouter_backend=openrouter,
            legacy_fallback=bool(router.get("legacy_fallback", False)),
        )


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


@dataclass
class RouteResult:
    decision: RoutingDecision
    receipt: Dict[str, Any]
    escalation_code: Optional[str]
    topic_escalation_code: Optional[str]
    session_escalation_code: Optional[str]
    backend_latency_ms: int
    latency_ms: float
    transitions: List[str]
    topic_band: str
    session_band: Optional[str]
    #: phase-2 additive (D8): None when the profile surface did not run (no profile
    #: candidates were declared), which is the pre-profile default and keeps this
    #: object's earlier shape meaningful.
    profile_band: Optional[str] = None
    profile_escalation_code: Optional[str] = None
    profile_scores: List[Dict[str, Any]] = field(default_factory=list)


def contamination_risk(flags: Sequence[str]) -> float:
    """K_j = 1 - prod(1 - k_j) over the session's contamination flags."""
    survival = 1.0
    for flag in flags:
        survival *= 1.0 - CONTAMINATION_WEIGHTS[flag]
    return 1.0 - survival


def _rank(scores: Mapping[str, float]) -> List[Tuple[str, float]]:
    # deterministic: score desc, then candidate id asc
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def route(
    request: ClassificationRequest,
    candidates: CandidateSet,
    config: RouterConfig,
    include_session: bool = True,
    receipt_id: Optional[str] = None,
    ts: Optional[str] = None,
) -> RouteResult:
    started = time.perf_counter()
    thresholds = config.thresholds
    calibration = config.calibration
    registry = config.label_registry
    registry.observe(candidates)

    receipt_id = receipt_id or str(uuid.uuid4())
    ts = ts or _utc_now()
    transitions: List[str] = []

    handle = RequestHandle(
        request_id=request.request_id,
        prompt_hash=request.prompt_hash,
        project_id_hash=request.project_id_hash,
        config_rev=request.config_rev,
        context_hash=request.context_hash,
    )

    topic_scores: List[Dict[str, Any]] = []
    session_scores: List[Dict[str, Any]] = []
    topic_top: Optional[Dict[str, Any]] = None
    topic_second: Optional[Dict[str, Any]] = None
    session_top: Optional[Dict[str, Any]] = None
    topic_margin = 0.0
    session_margin = 0.0
    contamination_score = 0.0
    token_cost_estimate = 0
    backend_latency_ns = 0
    topic_band = "abstain_or_new_session"
    session_band: Optional[str] = None
    topic_code: Optional[str] = None
    session_code: Optional[str] = None
    decision_value = "new_session"
    band_value = "abstain_or_new_session"
    # -- profile surface (D8): declared only when the caller sent profile candidates
    profile_scores: List[Dict[str, Any]] = []
    profile_candidate_ids: List[str] = []
    profile_top: Optional[Dict[str, Any]] = None
    profile_margin = 0.0
    profile_band: Optional[str] = None
    profile_code: Optional[str] = None

    # -- explicit directive short-circuit (section 1.1) ---------------------
    if request.explicit_directive:
        decision = _decision(
            request, "fallback_escalate", "escalate", [], [], None, None, 0.0, None, 0.0,
            thresholds, calibration, 0.0, 0,
        )
        receipt = build_receipt(
            decision, candidates, request, _backend_name(config.backend),
            int((time.perf_counter() - started) * 1000), receipt_id, ts,
            escalation_code="explicit_directive", label_state_touched=None,
            router_version=config.router_version,
            fallback_from=_fallback_source(config.backend),
        )
        return RouteResult(
            decision, receipt, "explicit_directive", "explicit_directive", None,
            0, (time.perf_counter() - started) * 1000.0, [], "fallback_escalate", None,
        )

    # -- topic side: ONE batch call over ALL candidates incl. sentinels -----
    try:
        t0 = time.perf_counter_ns()
        raw_topic = config.backend.batch_score(handle, candidates, "topic")
        backend_latency_ns += time.perf_counter_ns() - t0
        calibrated_topic = {
            row["candidate"]: calibration.calibrate("topic", row["raw"]) for row in raw_topic
        }
        topic_scores = [
            {
                "candidate": row["candidate"],
                "raw": float(row["raw"]),
                "calibrated": calibrated_topic[row["candidate"]],
            }
            for row in raw_topic
        ]
        ranked_topic = _rank(calibrated_topic)
        top_candidate, top_cal = ranked_topic[0]
        topic_top = {"candidate": top_candidate, "calibrated": top_cal}
        if len(ranked_topic) > 1:
            second_candidate, second_cal = ranked_topic[1]
            topic_second = {"candidate": second_candidate, "calibrated": second_cal}
            topic_margin = top_cal - second_cal
        topic_band, topic_code = _topic_band(
            ranked_topic, calibrated_topic, thresholds, calibration.active
        )
    except (BackendError, DisabledByPolicy) as exc:
        topic_code = (
            "backend_disabled_by_policy" if isinstance(exc, DisabledByPolicy) else "backend_error"
        )
        topic_band = "fallback_escalate"

    # -- novelty gate (section 6) ------------------------------------------
    if (
        topic_top is not None
        and topic_top["candidate"] == TOPIC_SENTINEL_NOVEL
        and topic_band == "automatic"
    ):
        accepted = _taxonomy_gate(request, candidates, config, handle)
        if accepted is None:
            topic_band = "fallback_escalate"
            topic_code = topic_code or "taxonomy_rejected"
            if isinstance(config.openrouter_backend, OpenRouterBackend) and not config.openrouter_backend.enabled:
                topic_code = "backend_disabled_by_policy"
        else:
            transitions.append(registry.create_provisional(accepted["slug"]))

    # -- session side: ONE batch call over ALL candidates incl. new_session -
    #     Session comparison is in scope only when a live session exists; with
    #     zero session candidates there is nothing to reuse, so the topic side
    #     stands alone (leo-arch.md section 3: "only when reuse is contemplated").
    if include_session and candidates.sessions:
        try:
            t0 = time.perf_counter_ns()
            raw_session = config.backend.batch_score(handle, candidates, "session")
            backend_latency_ns += time.perf_counter_ns() - t0
            session_scores, session_risks, session_top, session_margin, contamination_score = _session_utility(
                raw_session, candidates, calibration, thresholds
            )
            session_band, session_code = _session_band(
                session_scores, session_risks, session_top, session_margin, thresholds, calibration.active
            )
        except (BackendError, DisabledByPolicy) as exc:
            session_code = (
                "backend_disabled_by_policy" if isinstance(exc, DisabledByPolicy) else "backend_error"
            )
            session_band = "abstain_or_new_session"

    # -- profile surface (D8): ONE batch call over ALL eligible profiles plus the
    #     `no_suitable_profile` sentinel.  Engaged ONLY when the caller declared
    #     profile candidates: eligibility is computed upstream, and with no eligible
    #     profile there is nothing to compare, so the pre-profile path is untouched.
    if candidates.profiles:
        profile_candidate_ids = candidates.profile_candidate_ids()
        try:
            t0 = time.perf_counter_ns()
            raw_profile = config.backend.batch_score(handle, candidates, "profile")
            backend_latency_ns += time.perf_counter_ns() - t0
            calibrated_profile = {
                row["candidate"]: calibration.calibrate("profile", row["raw"])
                for row in raw_profile
            }
            profile_scores = [
                {
                    "candidate": row["candidate"],
                    "raw": float(row["raw"]),
                    "calibrated": calibrated_profile[row["candidate"]],
                }
                for row in raw_profile
            ]
            ranked_profile = _rank(calibrated_profile)
            profile_top = {
                "candidate": ranked_profile[0][0],
                "calibrated": ranked_profile[0][1],
            }
            if len(ranked_profile) > 1:
                profile_margin = ranked_profile[0][1] - ranked_profile[1][1]
            profile_band, profile_code = _profile_band(
                ranked_profile, thresholds, calibration.active
            )
        except (BackendError, DisabledByPolicy) as exc:
            # fail closed (D3): no scores => no profile block in the receipt, and the
            # band can only get MORE conservative below.
            profile_code = (
                "backend_disabled_by_policy" if isinstance(exc, DisabledByPolicy) else "backend_error"
            )
            profile_band = "abstain_or_new_session"
            profile_scores = []
            profile_candidate_ids = []

    # -- effective band = the most conservative of the surfaces that ran -----
    #     (identical tie-breaking to the topic/session pair: topic first, then the
    #     other surface's code)
    band_value, code = topic_band, topic_code
    if session_band is not None:
        if BAND_ORDER[session_band] > BAND_ORDER[topic_band]:
            band_value, code = session_band, session_code
        elif BAND_ORDER[session_band] == BAND_ORDER[topic_band]:
            code = code or session_code
    if profile_band is not None:
        if BAND_ORDER[profile_band] > BAND_ORDER[band_value]:
            band_value, code = profile_band, profile_code
        elif BAND_ORDER[profile_band] == BAND_ORDER[band_value]:
            code = code or profile_code

    # -- decision ----------------------------------------------------------
    if band_value == "automatic":
        if topic_top and topic_top["candidate"] == TOPIC_SENTINEL_NOVEL:
            decision_value = "novel_propose"
        elif session_band == "automatic" and session_top and session_top["candidate"] != SESSION_SENTINEL_NEW:
            decision_value = "session_reuse"
            token_cost_estimate = int(
                candidates.session_index()[session_top["candidate"]].context_token_cost
            )
        else:
            decision_value = "topic_assign"
    elif band_value == "fallback_escalate":
        decision_value = "escalate"
    else:
        decision_value = "new_session"

    # -- provisional-label corroboration (section 6) ------------------------
    if topic_top and topic_top["candidate"] not in (TOPIC_SENTINEL_NOVEL, TOPIC_SENTINEL_AMBIGUOUS):
        session_id = (
            session_top["candidate"]
            if session_top and decision_value == "session_reuse"
            else SESSION_SENTINEL_NEW
        )
        transition = registry.register_corroboration(
            topic_top["candidate"],
            session_id,
            request.ts[:10],
            topic_top["calibrated"],
            thresholds.tau_topic,
        )
        if transition:
            transitions.append(transition)

    # -- legacy rollback (section 8) ---------------------------------------
    if config.legacy_fallback:
        band_value = "fallback_escalate"
        decision_value = "escalate"
        code = "legacy_mode"

    decision = _decision(
        request, band_value, decision_value, topic_scores, session_scores, topic_top,
        topic_second, topic_margin, session_top, session_margin, thresholds, calibration,
        contamination_score, token_cost_estimate,
        profile_scores=profile_scores,
        profile_candidate_ids=profile_candidate_ids,
        profile_top=profile_top,
        profile_margin=profile_margin,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    receipt = build_receipt(
        decision,
        candidates,
        request,
        _backend_name(config.backend),
        int(backend_latency_ns / 1_000_000),
        receipt_id,
        ts,
        escalation_code=code,
        label_state_touched=transitions or None,
        router_version=config.router_version,
        fallback_from=_fallback_source(config.backend),
    )
    return RouteResult(
        decision=decision,
        receipt=receipt,
        escalation_code=code,
        topic_escalation_code=topic_code,
        session_escalation_code=session_code,
        backend_latency_ms=int(backend_latency_ns / 1_000_000),
        latency_ms=latency_ms,
        transitions=transitions,
        topic_band=topic_band,
        session_band=session_band,
        profile_band=profile_band,
        profile_escalation_code=profile_code,
        profile_scores=profile_scores,
    )


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _backend_name(backend: ClassificationBackend) -> str:
    name = getattr(backend, "name", "synthetic")
    return name if name in ("synthetic", "openrouter", "openjev") else "synthetic"


def _fallback_source(backend: Any) -> Optional[str]:
    """The provider id a declared fallback chain fell back FROM, or None (D3).

    Read straight off the backend (``ProviderChainBackend`` sets it only when a
    fallback fires); every other backend yields None, so the local-only path never
    gains the receipt field.
    """
    value = getattr(backend, "last_fallback_from", None)
    return value if isinstance(value, str) and value else None


def _decision(
    request: ClassificationRequest,
    band: str,
    decision_value: str,
    topic_scores: List[Dict[str, Any]],
    session_scores: List[Dict[str, Any]],
    topic_top: Optional[Dict[str, Any]],
    topic_second: Optional[Dict[str, Any]],
    topic_margin: float,
    session_top: Optional[Dict[str, Any]],
    session_margin: float,
    thresholds: Thresholds,
    calibration: CalibrationModel,
    contamination_score: float,
    token_cost_estimate: int,
    profile_scores: Optional[List[Dict[str, Any]]] = None,
    profile_candidate_ids: Optional[List[str]] = None,
    profile_top: Optional[Dict[str, Any]] = None,
    profile_margin: float = 0.0,
) -> RoutingDecision:
    return RoutingDecision(
        request_id=request.request_id,
        band=band,
        decision=decision_value,
        topic_scores=topic_scores,
        session_scores=session_scores,
        topic_top=topic_top,
        topic_second=topic_second,
        topic_margin=topic_margin,
        session_top=session_top,
        session_margin=session_margin,
        thresholds_used=thresholds.to_dict(),
        calibration_model_id=calibration.model_id,
        contamination_score=contamination_score,
        token_cost_estimate=token_cost_estimate,
        profile_candidate_ids=list(profile_candidate_ids or []),
        profile_scores=list(profile_scores or []),
        profile_top=profile_top,
        profile_margin=float(profile_margin),
    )


# --------------------------------------------------------------------------
# Band policy (leo-arch.md section 3)
# --------------------------------------------------------------------------


def _topic_band(
    ranked: List[Tuple[str, float]],
    calibrated: Mapping[str, float],
    thresholds: Thresholds,
    calibration_active: bool,
) -> Tuple[str, Optional[str]]:
    top_candidate, top_cal = ranked[0]
    second_candidate = ranked[1][0] if len(ranked) > 1 else None
    margin = top_cal - ranked[1][1] if len(ranked) > 1 else 1.0
    cal_ambiguous = calibrated.get(TOPIC_SENTINEL_AMBIGUOUS, 0.0)

    if not calibration_active:
        band = "abstain_or_new_session" if top_cal < thresholds.tau_low else "fallback_escalate"
        return band, "calibration_missing"

    if TOPIC_SENTINEL_AMBIGUOUS in (top_candidate, second_candidate) and cal_ambiguous >= 0.40:
        return "fallback_escalate", "ambiguous_won"

    if top_candidate == TOPIC_SENTINEL_NOVEL:
        if top_cal >= thresholds.tau_novel and margin >= thresholds.mu_novel:
            return "automatic", None
        if top_cal < thresholds.tau_novel:
            return "fallback_escalate", "novel_below_threshold"
        return "fallback_escalate", "below_margin"

    if top_cal < thresholds.tau_low:
        return "abstain_or_new_session", "below_min_score"
    if top_cal >= thresholds.tau_topic and margin >= thresholds.mu_topic:
        return "automatic", None
    if top_cal >= thresholds.tau_topic:
        return "fallback_escalate", "below_margin"
    return "fallback_escalate", "below_min_score"


def _profile_band(
    ranked: List[Tuple[str, float]],
    thresholds: Thresholds,
    calibration_active: bool,
) -> Tuple[str, Optional[str]]:
    """Profile-surface band (D8).

    Reuses the EXISTING generic gates -- ``tau_low`` for the score floor and
    ``tau_topic`` / ``mu_topic`` for the comparative winner margin -- so the profile
    surface introduces no threshold of its own and cannot weaken the band.  Two
    invariants are stricter than the topic path on purpose:

    * the explicit ``no_suitable_profile`` candidate winning ALWAYS yields the abstain
      band, regardless of its score and regardless of calibration state: an explicit
      abstain is not something a threshold may overrule;
    * with no validated calibration (``CalibrationModel.active`` false, i.e.
      ``calibration_model_id == "none"``) the automatic band is unreachable.
    """
    if not ranked:
        return "abstain_or_new_session", "profile_abstain_argmax"
    top_candidate, top_cal = ranked[0]
    margin = top_cal - ranked[1][1] if len(ranked) > 1 else 1.0
    if top_candidate == PROFILE_SENTINEL_NO_SUITABLE:
        return "abstain_or_new_session", "profile_abstain_argmax"
    if not calibration_active:
        band = "abstain_or_new_session" if top_cal < thresholds.tau_low else "fallback_escalate"
        return band, "calibration_missing"
    if top_cal < thresholds.tau_low:
        return "abstain_or_new_session", "below_min_score"
    if top_cal >= thresholds.tau_topic and margin >= thresholds.mu_topic:
        return "automatic", None
    if top_cal >= thresholds.tau_topic:
        return "fallback_escalate", "below_margin"
    return "fallback_escalate", "below_min_score"


def _session_utility(
    raw_session: List[Dict[str, Any]],
    candidates: CandidateSet,
    calibration: CalibrationModel,
    thresholds: Thresholds,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[Dict[str, Any]], float, float]:
    sessions = candidates.session_index()
    rows: List[Dict[str, Any]] = []
    risks: Dict[str, float] = {}
    for row in raw_session:
        candidate = row["candidate"]
        calibrated = calibration.calibrate("session", row["raw"])
        if candidate == SESSION_SENTINEL_NEW:
            utility = thresholds.new_session_utility
            risk = 0.0
        else:
            session = sessions[candidate]
            cost_term = min(1.0, session.context_token_cost / float(thresholds.context_cost_norm))
            risk = contamination_risk(session.contamination_flags)
            utility = calibrated - thresholds.w_cost * cost_term - thresholds.w_contam * risk
        rows.append(
            {
                "candidate": candidate,
                "raw": float(row["raw"]),
                "calibrated": calibrated,
                "utility": utility,
            }
        )
        risks[candidate] = risk

    ranked = sorted(rows, key=lambda item: (-item["utility"], item["candidate"]))
    top = ranked[0]
    new_row = next((r for r in rows if r["candidate"] == SESSION_SENTINEL_NEW), None)
    second_reuse = next((r for r in ranked[1:] if r["candidate"] != SESSION_SENTINEL_NEW), None)
    reference = max([r["utility"] for r in (new_row, second_reuse) if r is not None] or [0.0])
    margin = top["utility"] - reference
    # contamination_score: risk of the real session with the highest similarity
    real_rows = [r for r in rows if r["candidate"] != SESSION_SENTINEL_NEW]
    contamination_score = (
        risks[max(real_rows, key=lambda r: (r["calibrated"], r["candidate"]))["candidate"]]
        if real_rows
        else 0.0
    )
    return rows, risks, {"candidate": top["candidate"], "utility": top["utility"]}, margin, contamination_score


def _session_band(
    session_scores: List[Dict[str, Any]],
    risks: Mapping[str, float],
    session_top: Optional[Dict[str, Any]],
    session_margin: float,
    thresholds: Thresholds,
    calibration_active: bool,
) -> Tuple[str, Optional[str]]:
    if not session_scores or session_top is None:
        return "abstain_or_new_session", "new_session_argmax"
    if not calibration_active:
        return "abstain_or_new_session", "calibration_missing"

    by_id = {row["candidate"]: row for row in session_scores}
    top = by_id[session_top["candidate"]]
    if top["candidate"] == SESSION_SENTINEL_NEW:
        return "abstain_or_new_session", "new_session_argmax"

    sessions = [row for row in session_scores if row["candidate"] != SESSION_SENTINEL_NEW]
    real_top = max(sessions, key=lambda r: (r["calibrated"], r["candidate"])) if sessions else None
    risk = float(risks.get(real_top["candidate"], 0.0)) if real_top else 0.0
    if risk >= thresholds.k_block:
        return "abstain_or_new_session", "contamination_high"
    if top["calibrated"] < thresholds.tau_sess_sim:
        return "abstain_or_new_session", "below_min_score"
    if top["utility"] < thresholds.tau_sess_util:
        return "abstain_or_new_session", "below_min_utility"
    if session_margin < thresholds.mu_sess:
        return "abstain_or_new_session", "below_margin"
    return "automatic", None


# --------------------------------------------------------------------------
# Novelty gate (leo-arch.md section 6)
# --------------------------------------------------------------------------


def _taxonomy_gate(
    request: ClassificationRequest,
    candidates: CandidateSet,
    config: RouterConfig,
    handle: RequestHandle,
) -> Optional[Dict[str, Any]]:
    backend = config.openrouter_backend
    if backend is None:
        return None
    try:
        return backend.propose_taxonomy(handle, [request.prompt_hash])
    except (DisabledByPolicy, BackendError):
        return None
