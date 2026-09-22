"""Fail-closed negative probes (evidence tooling, stdlib only, offline).

Each probe asserts that the router REFUSES something it must refuse, and prints
PASS/FAIL.  Exit 0 iff every probe passes.  Probe list (brief step 3):
  (a) disabled OpenRouter adapter raises DisabledByPolicy
  (b) malformed candidate set is rejected by the closed schema
  (c) the receipt writer refuses raw prompt text / free text
  (d) a sub-threshold score OR a sub-threshold margin never routes `automatic`
  (e) `new_session` wins whenever session reuse is ambiguous
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ordaprompt_router.adapter import (  # noqa: E402
    DisabledByPolicy,
    OpenRouterBackend,
    RequestHandle,
    SyntheticBackend,
    sha256_hex,
)
from ordaprompt_router.receipts import ReceiptStore, receipt_prompt_text_absent  # noqa: E402
from ordaprompt_router.router import CalibrationModel, LabelRegistry, RouterConfig, route  # noqa: E402
from ordaprompt_router.schemas import (  # noqa: E402
    CandidateSet,
    ClassificationRequest,
    PrivacyViolationError,
    SchemaError,
)

RESULTS = []


def probe(name, fn):
    try:
        fn()
    except AssertionError as exc:
        RESULTS.append((name, False, str(exc)))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, "%s: %s" % (type(exc).__name__, exc)))
    else:
        RESULTS.append((name, True, ""))


# identity-active calibration -> calibrated value == raw value (threshold control)
IDENTITY = CalibrationModel(model_id="platt-v1", active=True, a_topic=1.0, b_topic=0.0,
                            a_session=1.0, b_session=0.0)
PROMPT_TEXT = "REFUND my invoice for the billing cycle, this is urgent"


def _request(**over):
    base = {
        "schema": "ordapilot.classification_request/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "ts": "2026-09-22T10:00:00Z",
        "prompt_hash": sha256_hex(PROMPT_TEXT),
        "context_hash": None,
        "context_token_estimate": 0,
        "project_id_hash": "sha256:" + "cd" * 32,
        "config_rev": "1.0.0",
        "explicit_directive": None,
    }
    base.update(over)
    return ClassificationRequest.from_dict(base)


def _candidates(topics, sessions):
    return CandidateSet.from_dict({
        "schema": "ordapilot.candidate_set/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "topics": [{"topic_id": t, "label_state": "promoted", "label_age_days": 1, "evidence_count": 1}
                   for t in topics],
        "topic_sentinels": ["novel", "ambiguous"],
        "sessions": sessions,
        "session_sentinels": ["new_session"],
    })


class TableBackend(SyntheticBackend):
    """Fixed raw scores by candidate id, one batch call per surface."""

    name = "synthetic"

    def __init__(self, topic, session=None):
        super().__init__()
        self.topic = dict(topic)
        self.session = dict(session or {})

    def batch_score(self, request_handle, candidates, surface):
        self.batch_calls[surface] = self.batch_calls.get(surface, 0) + 1
        table = self.topic if surface == "topic" else self.session
        ids = (candidates.topic_candidate_ids() if surface == "topic"
               else candidates.session_candidate_ids())
        return [{"candidate": c, "raw": float(table.get(c, 0.5))} for c in ids]

    def propose_taxonomy(self, request_handle, exemplar_hashes):
        return None


def _route(topic_scores, session_scores=None, sessions=None, calibration=IDENTITY):
    config = RouterConfig(thresholds=__import__(
        "ordaprompt_router.router", fromlist=["Thresholds"]).Thresholds(),
        calibration=calibration,
        backend=TableBackend(topic_scores, session_scores or {}),
        label_registry=LabelRegistry())
    return route(_request(), _candidates(["topic-auth", "topic-billing"], sessions or []), config)


# -- (a) -----------------------------------------------------------------
def probe_a():
    backend = OpenRouterBackend(enabled=False)
    try:
        backend.propose_taxonomy(RequestHandle("r", "sha256:" + "ab" * 32, "sha256:" + "cd" * 32, "1.0.0"),
                                 ["sha256:" + "ab" * 32])
    except DisabledByPolicy as exc:
        assert "disabled by policy" in str(exc), str(exc)
        return
    raise AssertionError("disabled adapter did NOT raise DisabledByPolicy")


# -- (b) -----------------------------------------------------------------
def probe_b():
    malformed = {
        "schema": "ordapilot.candidate_set/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "topics": [{"topic_id": "topic-auth", "label_state": "promoted", "label_age_days": 1, "evidence_count": 1}],
        "topic_sentinels": ["novel"],          # missing `ambiguous`
        "sessions": [],
        "session_sentinels": ["new_session"],
    }
    try:
        CandidateSet.from_dict(malformed)
    except SchemaError as exc:
        assert "topic_sentinels" in str(exc), str(exc)
    else:
        raise AssertionError("malformed candidate set accepted (missing sentinel)")

    poisoned = dict(malformed)
    poisoned["topic_sentinels"] = ["novel", "ambiguous"]
    poisoned["topics"] = [{"topic_id": "topic-auth", "label_state": "promoted",
                           "label_age_days": 1, "evidence_count": 1, "free_text": "user was angry"}]
    try:
        CandidateSet.from_dict(poisoned)
    except SchemaError as exc:
        assert "unknown field" in str(exc), str(exc)
    else:
        raise AssertionError("malformed candidate set accepted (unknown field)")


# -- (c) -----------------------------------------------------------------
def probe_c():
    with tempfile.TemporaryDirectory() as tmp:
        store = ReceiptStore(tmp)
        result = _route({"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1})
        receipt = dict(result.receipt)

        bad = dict(receipt)
        bad["rationale"] = "user asked about a refund for the billing cycle"
        try:
            store.append(bad)
        except (SchemaError, PrivacyViolationError):
            pass
        else:
            raise AssertionError("receipt with a free-text field was accepted")

        bad2 = dict(receipt)
        bad2["topic_candidate_ids"] = [PROMPT_TEXT]
        try:
            store.append(bad2)
        except (SchemaError, PrivacyViolationError):
            pass
        else:
            raise AssertionError("receipt with raw prompt text was accepted")

        # a legitimate receipt is still writable, and the raw prompt is absent
        store.append(receipt)
        assert store.count() == 1, "valid receipt was not written"
        assert receipt_prompt_text_absent(receipt, PROMPT_TEXT), "raw prompt text leaked into the receipt"
        with open(store.path, "r", encoding="utf-8") as fh:
            body = fh.read()
        assert "REFUND" not in body and "refund" not in body, "prompt content found in the receipt file"


# -- (d) -----------------------------------------------------------------
def probe_d():
    # d1: strong margin, score below tau_topic
    r1 = _route({"topic-auth": 0.70, "topic-billing": 0.10, "novel": 0.05, "ambiguous": 0.05})
    assert r1.decision.band != "automatic", "sub-threshold score routed automatic"
    assert r1.topic_escalation_code == "below_min_score", r1.topic_escalation_code

    # d2: score above tau_topic, margin below mu_topic
    r2 = _route({"topic-auth": 0.90, "topic-billing": 0.89, "novel": 0.05, "ambiguous": 0.05})
    assert r2.decision.band != "automatic", "sub-threshold margin routed automatic"
    assert r2.topic_escalation_code == "below_margin", r2.topic_escalation_code

    # d3: no calibration at all -> automatic unreachable even with perfect scores
    r3 = _route({"topic-auth": 0.99, "topic-billing": 0.01, "novel": 0.0, "ambiguous": 0.0},
                calibration=CalibrationModel())
    assert r3.decision.band != "automatic", "automatic without calibration"
    assert r3.topic_escalation_code == "calibration_missing", r3.topic_escalation_code


# -- (e) -----------------------------------------------------------------
def probe_e():
    clean = [{"session_id": "sess-a", "project_id_hash": "sha256:" + "cd" * 32,
              "context_token_cost": 400, "idle_days": 1, "contamination_flags": []}]
    topic = {"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1}

    # e1: session similarity below tau_sess_sim -> new_session, never reuse
    r1 = _route(topic, {"sess-a": 0.80, "new_session": 0.10}, sessions=clean)
    assert r1.decision.decision != "session_reuse", "reuse granted below tau_sess_sim"
    assert r1.session_band == "abstain_or_new_session", r1.session_band

    # e2: contaminated high-similarity session -> new_session wins
    contaminated = [{"session_id": "sess-a", "project_id_hash": "sha256:" + "cd" * 32,
                     "context_token_cost": 400, "idle_days": 1,
                     "contamination_flags": ["project_mismatch"]}]
    r2 = _route(topic, {"sess-a": 0.99, "new_session": 0.10}, sessions=contaminated)
    assert r2.decision.decision != "session_reuse", "reuse granted on a contaminated session"
    assert r2.session_escalation_code == "contamination_high", r2.session_escalation_code

    # e3: missing calibration -> new_session by default
    r3 = _route(topic, {"sess-a": 0.99, "new_session": 0.10}, sessions=clean,
                calibration=CalibrationModel())
    assert r3.decision.decision != "session_reuse", "reuse granted without calibration"
    assert r3.decision.decision == "new_session", r3.decision.decision


for label, fn in (("(a) disabled OpenRouter adapter raises DisabledByPolicy", probe_a),
                  ("(b) malformed candidate set rejected (closed schema)", probe_b),
                  ("(c) receipt writer refuses raw prompt / free text", probe_c),
                  ("(d) sub-threshold score or margin never routes automatic", probe_d),
                  ("(e) new_session wins whenever session reuse is ambiguous", probe_e)):
    probe(label, fn)

failed = 0
for name, ok, detail in RESULTS:
    print("%-62s %s%s" % (name, "PASS" if ok else "FAIL", ("  <- " + detail) if detail else ""))
    failed += 0 if ok else 1
print("NEGATIVE-PROBES %s probes=%d failed=%d" % ("PASS" if not failed else "FAIL", len(RESULTS), failed))
sys.exit(0 if not failed else 1)
