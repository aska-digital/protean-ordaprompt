"""Offline evaluation harness -- SYNTHETIC fixtures only, zero network.

What this harness can and cannot prove
--------------------------------------
The fixtures are deterministic synthetic cases scored by the deterministic
:class:`SyntheticBackend`; outcomes are *designed*, not observed.  The harness
therefore validates the **machinery** (band policy, calibration fitting, unlock
gating, receipt privacy, determinism) and the fixture *distribution* it was
handed.  It is NOT evidence of real-world classifier quality: real calibration
requires real routing receipts plus human outcome feedback (leo-arch.md
section 7).  Every report says so.

Run:
    python3 build/eval/harness.py --self-test      # internal checks, exit 0/1
    python3 build/eval/harness.py                  # full run -> results.json/.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import fixturegen as fg  # noqa: E402
from ordaprompt_router.adapter import (  # noqa: E402
    BackendError,
    ClassificationBackend,
    DisabledByPolicy,
    OpenRouterBackend,
    SyntheticBackend,
    assert_hash_only_payload,
    canonicalize_redacted,
    sha256_hex,
)
from ordaprompt_router.receipts import (  # noqa: E402
    ReceiptStore,
    assert_no_free_text,
    receipt_prompt_text_absent,
    store_sha256,
)
from ordaprompt_router.router import (  # noqa: E402
    CalibrationModel,
    LabelRegistry,
    RouterConfig,
    Thresholds,
    _logit,
    _sigmoid,
    contamination_risk,
    route,
)
from ordaprompt_router.schemas import (  # noqa: E402
    CandidateSet,
    ClassificationRequest,
    PrivacyViolationError,
    SchemaError,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
RESULTS_JSON = os.path.join(HERE, "results.json")
RESULTS_MD = os.path.join(HERE, "results.md")
CALIBRATION_JSON = os.path.join(HERE, "calibration.json")
MIN_IN_BAND_HOLDOUT = 50
TARGET_IN_BAND_PRECISION = 0.98

DISCLAIMER = (
    "SYNTHETIC FIXTURES ONLY: outcomes are designed by the fixture generator and scored by the "
    "deterministic hash backend. These numbers validate the routing/calibration machinery, not "
    "real-world classifier quality. Real calibration requires real receipts plus human feedback."
)


# --------------------------------------------------------------------------
# Stub backends for policy unit checks
# --------------------------------------------------------------------------


class TableBackend(ClassificationBackend):
    """Fixed raw scores by candidate id; one batch call per surface."""

    name = "synthetic"

    def __init__(self, topic: Dict[str, float], session: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.topic = dict(topic)
        self.session = dict(session or {})

    def batch_score(self, request_handle, candidates, surface):  # type: ignore[override]
        self.batch_calls[surface] = self.batch_calls.get(surface, 0) + 1
        table = self.topic if surface == "topic" else self.session
        ids = candidates.topic_candidate_ids() if surface == "topic" else candidates.session_candidate_ids()
        return [{"candidate": c, "raw": float(table.get(c, 0.5))} for c in ids]

    def propose_taxonomy(self, request_handle, exemplar_hashes):  # type: ignore[override]
        return None


class FailingBackend(TableBackend):
    def batch_score(self, request_handle, candidates, surface):  # type: ignore[override]
        raise BackendError("simulated backend failure")


# --------------------------------------------------------------------------
# Deterministic ids / helpers
# --------------------------------------------------------------------------


def _det_uuid(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


# --------------------------------------------------------------------------
# Calibration fitting (Platt scaling on the tuning split)
# --------------------------------------------------------------------------


def fit_platt(samples: Sequence[Tuple[float, int]], iterations: int = 6000, lr: float = 0.5, l2: float = 0.05) -> Dict[str, Any]:
    """Fit c = sigma(a*logit(raw) + b) by gradient descent on log-loss.

    ``l2`` regularizes the slope so the fitted map stays smooth (a moderate
    ``a``) instead of collapsing to a step function on perfectly-separable
    synthetic data -- which would erase the margin signal the policy needs.
    """
    if not samples:
        raise ValueError("no calibration samples")
    labels = {int(y) for _, y in samples}
    if len(labels) < 2:
        raise ValueError("calibration samples need both classes, got %s" % (labels,))
    xs = [_logit(s) for s, _ in samples]
    ys = [int(y) for _, y in samples]
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) or 1.0
    zs = [(x - mean) / sd for x in xs]
    a = 0.0
    b = 0.0
    for _ in range(iterations):
        ga = 0.0
        gb = 0.0
        for z, y in zip(zs, ys):
            p = _sigmoid(a * z + b)
            ga += (p - y) * z
            gb += p - y
        ga = ga / len(zs) + l2 * a
        gb = gb / len(zs)
        a -= lr * ga
        b -= lr * gb
    return {
        "a": a / sd,
        "b": b - a * mean / sd,
        "n": len(samples),
        "positives": sum(ys),
        "logit_mean": mean,
        "logit_sd": sd,
    }


def ece_and_brier(samples: Sequence[Tuple[float, int]], bins: int = 10) -> Dict[str, float]:
    if not samples:
        return {"ece": 0.0, "brier": 0.0, "nll": 0.0, "n": 0}
    brier = sum((p - y) ** 2 for p, y in samples) / len(samples)
    nll = 0.0
    for p, y in samples:
        p = min(1.0 - 1e-9, max(1e-9, p))
        nll -= (y * math.log(p) + (1 - y) * math.log(1 - p))
    nll /= len(samples)
    buckets: Dict[int, List[Tuple[float, int]]] = {}
    for p, y in samples:
        buckets.setdefault(min(bins - 1, int(p * bins)), []).append((p, y))
    ece = 0.0
    for members in buckets.values():
        confidence = sum(p for p, _ in members) / len(members)
        accuracy = sum(y for _, y in members) / len(members)
        ece += (len(members) / len(samples)) * abs(confidence - accuracy)
    return {"ece": ece, "brier": brier, "nll": nll, "n": len(samples)}


# --------------------------------------------------------------------------
# Routing a fixture set
# --------------------------------------------------------------------------


def _route_case(case: Dict[str, Any], backend: ClassificationBackend, calibration: CalibrationModel,
                thresholds: Thresholds, registry: LabelRegistry) -> Dict[str, Any]:
    request = ClassificationRequest.from_dict(case["request"])
    candidates = CandidateSet.from_dict(case["candidates"])
    config = RouterConfig(
        thresholds=thresholds,
        calibration=calibration,
        backend=backend,
        label_registry=registry,
    )
    started = time.perf_counter()
    result = route(
        request,
        candidates,
        config,
        include_session=bool(case["include_session"]),
        receipt_id=_det_uuid("receipt", case["case_id"]),
        ts=case["request"]["ts"],
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "case": case,
        "result": result,
        "latency_ms": latency_ms,
        "topic_top": (result.decision.topic_top or {}).get("candidate"),
        "topic_top_calibrated": (result.decision.topic_top or {}).get("calibrated", 0.0),
        "topic_margin": result.decision.topic_margin,
        "session_top": (result.decision.session_top or {}).get("candidate"),
        "session_band": result.session_band,
        "topic_band": result.topic_band,
        "band": result.decision.band,
        "decision": result.decision.decision,
        "escalation_code": result.escalation_code,
        "token_cost_estimate": result.decision.token_cost_estimate,
        "contamination_score": result.decision.contamination_score,
        "transitions": list(result.transitions),
    }


def route_all(cases: Sequence[Dict[str, Any]], calibration: CalibrationModel, thresholds: Optional[Thresholds] = None) -> List[Dict[str, Any]]:
    backend = fg.fixture_backend(cases)
    thresholds = thresholds or Thresholds()
    return [_route_case(case, backend, calibration, thresholds, LabelRegistry()) for case in cases]


def replay_calls(cases: Sequence[Dict[str, Any]], calibration: CalibrationModel) -> Dict[str, int]:
    backend = fg.fixture_backend(cases)
    registry = LabelRegistry()
    thresholds = Thresholds()
    for case in cases:
        _route_case(case, backend, calibration, thresholds, registry)
    return dict(backend.batch_calls)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _topic_truth(row: Dict[str, Any]) -> Optional[str]:
    return row["case"]["truth"]["topic"]


def _session_truth(row: Dict[str, Any]) -> Optional[str]:
    return row["case"]["truth"]["session"]


def compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    labelled = [r for r in rows if _topic_truth(r) is not None]
    correct = [r for r in labelled if r["topic_top"] == _topic_truth(r)]
    auto_topics = [
        r
        for r in rows
        if r["band"] == "automatic" and r["decision"] in ("topic_assign", "session_reuse")
    ]
    auto_correct = [r for r in auto_topics if r["topic_top"] == _topic_truth(r)]
    novel_calls = [r for r in rows if r["topic_top"] == "novel"]
    novel_truth = [r for r in rows if _topic_truth(r) == "novel"]
    novel_tp = [r for r in novel_calls if _topic_truth(r) == "novel"]

    reuse_rows = [r for r in rows if r["session_band"] == "automatic"]
    reuse_clean: List[Dict[str, Any]] = []
    for r in reuse_rows:
        session = CandidateSet.from_dict(r["case"]["candidates"]).session_index().get(r["session_top"])
        if session is None or not session.contamination_flags:
            reuse_clean.append(r)
    clean_targets = [r for r in rows if isinstance(_session_truth(r), str) and _session_truth(r) != "new_session"]
    clean_reused = [r for r in clean_targets if r["session_band"] == "automatic" and r["session_top"] == _session_truth(r)]

    session_labelled = [r for r in rows if _session_truth(r) is not None]
    session_correct = []
    for r in session_labelled:
        truth = _session_truth(r)
        if truth == "new_session":
            session_correct.append(r["session_band"] != "automatic")
        else:
            session_correct.append(r["session_band"] == "automatic" and r["session_top"] == truth)

    bands: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    codes: Dict[str, int] = {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
        if r["escalation_code"]:
            codes[r["escalation_code"]] = codes.get(r["escalation_code"], 0) + 1

    latencies = [r["latency_ms"] for r in rows]
    tokens = sum(r["token_cost_estimate"] for r in rows)
    refused_tokens = 0
    for r in rows:
        candidates = CandidateSet.from_dict(r["case"]["candidates"])
        flags = candidates.session_index().get(r["session_top"] or "", None)
        if flags is not None and flags.contamination_flags:
            refused_tokens += flags.context_token_cost

    def rate(num: int, den: int) -> float:
        return round(num / float(den), 6) if den else 0.0

    return {
        "cases": n,
        "topic": {
            "labelled_cases": len(labelled),
            "accuracy": rate(len(correct), len(labelled)),
            "automatic_assignments": len(auto_topics),
            "automatic_precision": rate(len(auto_correct), len(auto_topics)),
            "novelty_precision": rate(len(novel_tp), len(novel_calls)),
            "novelty_recall": rate(len(novel_tp), len(novel_truth)),
            "novel_calls": len(novel_calls),
        },
        "session": {
            "reuse_granted": len(reuse_rows),
            "reuse_accuracy": rate(
                len([r for r in reuse_rows if r["session_top"] == _session_truth(r)]), len(reuse_rows)
            ),
            "contamination_rate": rate(len(reuse_rows) - len(reuse_clean), len(reuse_rows)),
            "clean_targets": len(clean_targets),
            "fragmentation": rate(len(clean_targets) - len(clean_reused), len(clean_targets)),
            "decision_accuracy": rate(sum(1 for ok in session_correct if ok), len(session_labelled)),
            "labelled_cases": len(session_labelled),
        },
        "bands": bands,
        "decisions": decisions,
        "escalation_codes": codes,
        "abstention_rate": rate(bands.get("abstain_or_new_session", 0), n),
        "automatic_rate": rate(bands.get("automatic", 0), n),
        "latency_ms": {
            "mean": round(sum(latencies) / n, 4) if n else 0.0,
            "p50": round(_percentile(latencies, 0.5), 4),
            "p95": round(_percentile(latencies, 0.95), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "token_cost": {
            "simulated_total_tokens_loaded": tokens,
            "mean_per_decision": round(tokens / float(n), 3) if n else 0.0,
            "reuse_decisions": len(reuse_rows),
            "contaminated_tokens_refused": refused_tokens,
        },
    }


def band_table(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {}
    for r in rows:
        key = "%s/%s" % (r["case"]["topic_kind"], r["case"]["session_kind"])
        entry = table.setdefault(key, {})
        entry[r["band"]] = entry.get(r["band"], 0) + 1
        entry["_n"] = entry.get("_n", 0) + 1
    return table


# --------------------------------------------------------------------------
# Calibration samples
# --------------------------------------------------------------------------

SESSION_CALIBRATION_KINDS = ("clean_reuse", "session_near_tie")


def calibration_samples(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Tuple[float, int]]]:
    """Labeled (raw, correct) pairs for Platt fitting on the TUNING split.

    Topic: only cases with a topic truth (clear/novel -> correct, low-confidence
    -> incorrect).  Sentinel wins (novel, ambiguous) are excluded: the policy's
    novelty/ambiguity rules own those, and mixing them would double-count the
    decision.  A real-topic top with no truth (near-tie / low-confidence) is a
    legitimate NEGATIVE sample: the assignment is not corroborated by anything.

    Session: only reuse-contemplated cases whose top is a real session
    (clean_reuse -> correct, session_near_tie -> incorrect).  Contaminated and
    expensive cases are excluded because their rejection is priced by the
    policy's K/C terms, not by similarity -- including them would double-price
    contamination into the calibrator.
    """
    topic: List[Tuple[float, int]] = []
    session: List[Tuple[float, int]] = []
    for r in rows:
        case = r["case"]
        top = r["topic_top"]
        if top is not None and top not in ("novel", "ambiguous"):
            topic.append(
                (float(case["raw"]["topic"][top]), 1 if top == case["truth"]["topic"] else 0)
            )
        if (
            case["session_kind"] in SESSION_CALIBRATION_KINDS
            and case["truth"]["session"] is not None
            and r["session_top"] is not None
            and r["session_top"] != "new_session"
        ):
            session.append(
                (float(case["raw"]["session"][r["session_top"]]), 1 if r["session_top"] == case["truth"]["session"] else 0)
            )
    return {"topic": topic, "session": session}


def calibrated_samples(rows: Sequence[Dict[str, Any]], calibration: CalibrationModel) -> Dict[str, List[Tuple[float, int]]]:
    """Calibrated (probability, correct) pairs -- same labeled population as
    :func:`calibration_samples`, so ECE/Brier measure the same cases the
    calibrator was fit to predict."""
    out: Dict[str, List[Tuple[float, int]]] = {"topic": [], "session": []}
    for r in rows:
        case = r["case"]
        top = r["topic_top"]
        if top is not None and top not in ("novel", "ambiguous"):
            out["topic"].append(
                (
                    calibration.calibrate("topic", float(case["raw"]["topic"][top])),
                    1 if top == case["truth"]["topic"] else 0,
                )
            )
        if (
            case["session_kind"] in SESSION_CALIBRATION_KINDS
            and case["truth"]["session"] is not None
            and r["session_top"] is not None
            and r["session_top"] != "new_session"
        ):
            stop = r["session_top"]
            out["session"].append(
                (
                    calibration.calibrate("session", float(case["raw"]["session"][stop])),
                    1 if stop == case["truth"]["session"] else 0,
                )
            )
    return out


def would_be_automatic(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    in_band = [r for r in rows if r["band"] == "automatic"]
    correct = 0
    for r in in_band:
        truth_topic = r["case"]["truth"]["topic"]
        if r["topic_top"] != truth_topic:
            continue
        truth_session = r["case"]["truth"]["session"]
        if r["decision"] == "session_reuse" and truth_session is not None and r["session_top"] != truth_session:
            continue
        correct += 1
    return {
        "in_band": len(in_band),
        "correct": correct,
        "precision": round(correct / float(len(in_band)), 6) if in_band else 0.0,
    }


# --------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------


def evaluate(cases: Sequence[Dict[str, Any]], with_full_verification: bool = False) -> Dict[str, Any]:
    started = time.time()
    tuning = [c for c in cases if c["split"] == "tuning"]
    holdout = [c for c in cases if c["split"] == "holdout"]
    thresholds = Thresholds()

    uncalibrated = CalibrationModel()
    tuning_rows_uncal = route_all(tuning, uncalibrated, thresholds)
    holdout_rows_uncal = route_all(holdout, uncalibrated, thresholds)

    samples = calibration_samples(tuning_rows_uncal)
    fit_topic = fit_platt(samples["topic"])
    fit_session = fit_platt(samples["session"])
    simulated = CalibrationModel(
        model_id="platt-v1",
        active=True,
        a_topic=fit_topic["a"],
        b_topic=fit_topic["b"],
        a_session=fit_session["a"],
        b_session=fit_session["b"],
    )

    tuning_rows = route_all(tuning, simulated, thresholds)
    holdout_rows = route_all(holdout, simulated, thresholds)

    holdout_probs = calibrated_samples(holdout_rows, simulated)
    tuning_probs = calibrated_samples(tuning_rows, simulated)
    holdout_topic_metrics = ece_and_brier(holdout_probs["topic"])
    holdout_session_metrics = ece_and_brier(holdout_probs["session"])
    in_band = would_be_automatic(holdout_rows)

    unlock_checks = {
        "holdout_topic_ece_le_0.05": holdout_topic_metrics["ece"] <= 0.05,
        "holdout_topic_brier_le_0.10": holdout_topic_metrics["brier"] <= 0.10,
        "holdout_session_ece_le_0.05": holdout_session_metrics["ece"] <= 0.05,
        "holdout_session_brier_le_0.10": holdout_session_metrics["brier"] <= 0.10,
        "holdout_in_band_ge_%d" % MIN_IN_BAND_HOLDOUT: in_band["in_band"] >= MIN_IN_BAND_HOLDOUT,
        "holdout_in_band_precision_ge_%.2f" % TARGET_IN_BAND_PRECISION: in_band["precision"] >= TARGET_IN_BAND_PRECISION,
    }
    unlocked = all(unlock_checks.values())
    production = simulated if unlocked else uncalibrated
    production_rows = route_all(cases, production, thresholds)

    calls = replay_calls(cases, simulated)
    expected_topic_calls = len(cases)
    # locked #1 is "ONE call per surface, all candidates in that call".  The
    # session surface is only in scope when a live session exists (there is
    # nothing to reuse otherwise), so the expectation counts cases that carry
    # at least one session candidate.
    cases_with_sessions = [c for c in cases if c["candidates"]["sessions"]]
    expected_session_calls = len(cases_with_sessions)

    fixtures_verified = None
    if with_full_verification:
        identical, manifest = fg.verify_fixtures(FIXTURES_DIR)
        fixtures_verified = {"byte_identical": identical, "manifest_sha256": manifest["cases_sha256"]}

    report = {
        "schema": "ordaprompt.eval_results/1",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": DISCLAIMER,
        "fixtures": {
            "cases": len(cases),
            "tuning": len(tuning),
            "holdout": len(holdout),
            "cases_sha256": hashlib.sha256(fg.cases_payload(cases).encode("utf-8")).hexdigest(),
            "kinds": fg.manifest_for(cases, fg.cases_payload(cases))["kind_counts"],
            "soft_target_rate": fg.manifest_for(cases, fg.cases_payload(cases))["soft_target_rate"],
            "byte_verified": fixtures_verified,
        },
        "policy": {
            "thresholds": thresholds.to_dict(),
            "contamination_weights": dict((k, v) for k, v in sorted(
                __import__("ordaprompt_router.schemas", fromlist=["CONTAMINATION_WEIGHTS"]).CONTAMINATION_WEIGHTS.items()
            )),
            "one_batch_call_per_surface": {
                "topic_calls": calls.get("topic", 0),
                "session_calls": calls.get("session", 0),
                "expected_topic_calls": expected_topic_calls,
                "expected_session_calls": expected_session_calls,
                "session_side_skipped_cases": len(cases) - len(cases_with_sessions),
                "note": "one batch call per surface; the session surface is skipped "
                        "when the candidate set has no live session",
                "holds": calls.get("topic", 0) == expected_topic_calls
                and calls.get("session", 0) == expected_session_calls,
            },
        },
        "calibration": {
            "method": "platt scaling on logit(raw), fitted on the TUNING split only",
            "topic": fit_topic,
            "session": {
                **fit_session,
                "fitted_on_kinds": list(SESSION_CALIBRATION_KINDS),
                "note": "session calibrator is fitted only on reuse-contemplated cases; "
                        "contamination and token cost are priced by the policy, not the calibrator",
            },
            "holdout_topic_calibrated": holdout_topic_metrics,
            "holdout_session_calibrated": holdout_session_metrics,
            "tuning_topic_calibrated": ece_and_brier(tuning_probs["topic"]),
            "tuning_session_calibrated": ece_and_brier(tuning_probs["session"]),
            "unlock_checks": unlock_checks,
            "unlocked": unlocked,
            "model_id": production.model_id,
            "production_active": production.active,
        },
        "uncalibrated_baseline": {
            "note": "with no validated calibration the automatic band MUST be unreachable (locked #5)",
            "tuning": compute_metrics(tuning_rows_uncal),
            "holdout": compute_metrics(holdout_rows_uncal),
        },
        "with_calibration_tuning": compute_metrics(tuning_rows),
        "with_calibration_holdout": {
            **compute_metrics(holdout_rows),
            "would_be_automatic": in_band,
        },
        "production_mode": compute_metrics(production_rows),
        "band_table_holdout": band_table(holdout_rows),
        "wall_clock_s": round(time.time() - started, 2),
    }
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("# OrdaPilot classifier router -- offline eval (synthetic fixtures)")
    add("")
    add("> %s" % report["disclaimer"])
    add("")
    add("Generated: %s   Cases: %s (tuning %s / holdout %s)" % (
        report["generated_at"], report["fixtures"]["cases"],
        report["fixtures"]["tuning"], report["fixtures"]["holdout"],
    ))
    add("")
    add("## Holdout metrics (calibrated simulation)")
    hold = report["with_calibration_holdout"]
    add("- topic accuracy (labelled): %.4f over %d cases" % (hold["topic"]["accuracy"], hold["topic"]["labelled_cases"]))
    add("- automatic assignment precision: %.4f over %d automatic decisions" % (
        hold["topic"]["automatic_precision"], hold["topic"]["automatic_assignments"]))
    add("- novelty precision: %.4f (calls %d) / novelty recall: %.4f" % (
        hold["topic"]["novelty_precision"], hold["topic"]["novel_calls"], hold["topic"]["novelty_recall"]))
    add("- session reuse accuracy: %.4f over %d reuse decisions" % (
        hold["session"]["reuse_accuracy"], hold["session"]["reuse_granted"]))
    add("- session contamination rate: %.4f (must be 0)" % hold["session"]["contamination_rate"])
    add("- session fragmentation: %.4f over %d clean-reuse cases" % (
        hold["session"]["fragmentation"], hold["session"]["clean_targets"]))
    add("- session decision accuracy: %.4f over %d labelled cases" % (
        hold["session"]["decision_accuracy"], hold["session"]["labelled_cases"]))
    add("- abstention rate: %.4f ; automatic rate: %.4f" % (hold["abstention_rate"], hold["automatic_rate"]))
    add("- latency per decision (ms): mean %.4f / p50 %.4f / p95 %.4f" % (
        hold["latency_ms"]["mean"], hold["latency_ms"]["p50"], hold["latency_ms"]["p95"]))
    add("- simulated tokens loaded: %d total, %.2f per decision, %d refused for contamination" % (
        hold["token_cost"]["simulated_total_tokens_loaded"],
        hold["token_cost"]["mean_per_decision"],
        hold["token_cost"]["contaminated_tokens_refused"],
    ))
    add("")
    add("## Band distribution")
    for band, count in sorted(report["with_calibration_holdout"]["bands"].items()):
        add("- %s: %d" % (band, count))
    add("")
    add("## Calibration (Platt on tuning split)")
    cal = report["calibration"]
    add("- topic: a=%.4f b=%.4f (n=%d, positives=%d)" % (
        cal["topic"]["a"], cal["topic"]["b"], cal["topic"]["n"], cal["topic"]["positives"]))
    add("- session: a=%.4f b=%.4f (n=%d, positives=%d)" % (
        cal["session"]["a"], cal["session"]["b"], cal["session"]["n"], cal["session"]["positives"]))
    add("- holdout topic ECE %.4f / Brier %.4f ; session ECE %.4f / Brier %.4f" % (
        cal["holdout_topic_calibrated"]["ece"], cal["holdout_topic_calibrated"]["brier"],
        cal["holdout_session_calibrated"]["ece"], cal["holdout_session_calibrated"]["brier"]))
    add("- holdout would-be-automatic: %d in band, precision %.4f" % (
        report["with_calibration_holdout"]["would_be_automatic"]["in_band"],
        report["with_calibration_holdout"]["would_be_automatic"]["precision"]))
    add("- unlock checks:")
    for key, value in sorted(cal["unlock_checks"].items()):
        add("    - %s: %s" % (key, "PASS" if value else "FAIL"))
    add("- unlocked: %s -> production calibration model id: %s" % (cal["unlocked"], cal["model_id"]))
    add("")
    add("## Locked-constraint invariants")
    uncal = report["uncalibrated_baseline"]
    add("- uncalibrated automatic rate (tuning/holdout): %.4f / %.4f (locked #5: must be 0)" % (
        uncal["tuning"]["automatic_rate"], uncal["holdout"]["automatic_rate"]))
    add("- one batch call per surface: %s" % report["policy"]["one_batch_call_per_surface"])
    add("- uncalibrated abstention rate (holdout): %.4f" % uncal["holdout"]["abstention_rate"])
    add("")
    add("## Fixture census")
    for kind, count in sorted(report["fixtures"]["kinds"].items()):
        add("- %s: %d" % (kind, count))
    add("")
    add("Wall clock: %ss" % report["wall_clock_s"])
    add("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="internal checks only; exit 0/1")
    parser.add_argument("--fixtures-dir", default=FIXTURES_DIR)
    parser.add_argument("--cases", type=int, default=fg.DEFAULT_TOTAL_CASES)
    parser.add_argument("--regenerate", action="store_true", help="rebuild fixtures before evaluating")
    parser.add_argument("--out-dir", default=HERE)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return self_test(args.fixtures_dir, args.cases)

    if args.regenerate or not os.path.isfile(os.path.join(args.fixtures_dir, "cases.json")):
        manifest = fg.write_fixtures(args.fixtures_dir, total=args.cases)
        print("fixtures generated: %s" % json.dumps(manifest, sort_keys=True))

    cases, manifest = fg.load_fixtures(args.fixtures_dir)
    print("loaded %d fixtures (%s)" % (len(cases), json.dumps(manifest.get("kind_counts", {}), sort_keys=True)))

    report = evaluate(cases, with_full_verification=True)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    with open(CALIBRATION_JSON, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                CalibrationModel(
                    model_id=report["calibration"]["model_id"],
                    active=report["calibration"]["production_active"],
                    a_topic=report["calibration"]["topic"]["a"],
                    b_topic=report["calibration"]["topic"]["b"],
                    a_session=report["calibration"]["session"]["a"],
                    b_session=report["calibration"]["session"]["b"],
                    unlock_evidence=report["calibration"]["unlock_checks"],
                ).to_dict(),
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )

    hold = report["with_calibration_holdout"]
    print("holdout: topic_acc=%.4f auto_prec=%.4f in_band=%d in_band_prec=%.4f contamination=%.4f abstain=%.4f" % (
        hold["topic"]["accuracy"], hold["topic"]["automatic_precision"],
        hold["would_be_automatic"]["in_band"], hold["would_be_automatic"]["precision"],
        hold["session"]["contamination_rate"], hold["abstention_rate"]))
    print("calibration unlocked=%s production_model=%s" % (
        report["calibration"]["unlocked"], report["calibration"]["model_id"]))
    print("wrote %s, %s, %s" % (os.path.join(args.out_dir, "results.json"), os.path.join(args.out_dir, "results.md"), CALIBRATION_JSON))
    return 0


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _small_cases(total: int = 52) -> List[Dict[str, Any]]:
    return fg.generate_cases(total=total)


def self_test(fixtures_dir: str, cases_total: int) -> int:
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    valid_request = {
        "schema": "ordapilot.classification_request/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "ts": "2026-09-22T10:00:00Z",
        "prompt_hash": "sha256:" + "ab" * 32,
        "context_hash": None,
        "context_token_estimate": 0,
        "project_id_hash": "sha256:" + "cd" * 32,
        "config_rev": "1.0.0",
        "explicit_directive": None,
    }
    valid_candidates = {
        "schema": "ordapilot.candidate_set/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "topics": [{"topic_id": "topic-auth", "label_state": "provisional", "label_age_days": 1, "evidence_count": 1}],
        "topic_sentinels": ["novel", "ambiguous"],
        "sessions": [
            {
                "session_id": "sess-0001",
                "project_id_hash": "sha256:" + "cd" * 32,
                "context_token_cost": 500,
                "idle_days": 2,
                "contamination_flags": [],
            }
        ],
        "session_sentinels": ["new_session"],
    }

    # 1. schemas
    try:
        ClassificationRequest.from_dict(valid_request)
        CandidateSet.from_dict(valid_candidates)
        check("schema.valid_payloads_accepted", True)
    except Exception as exc:  # noqa: BLE001
        check("schema.valid_payloads_accepted", False, repr(exc))

    def rejects(mutator) -> bool:
        payload = json.loads(json.dumps(valid_candidates))
        mutator(payload)
        try:
            CandidateSet.from_dict(payload)
        except SchemaError:
            return True
        except Exception:  # noqa: BLE001
            return False
        return False

    check("schema.unknown_field_rejected", rejects(lambda p: p.update({"extra": 1})))
    check("schema.missing_sentinels_rejected", rejects(lambda p: p.update({"topic_sentinels": ["novel"]})))
    check("schema.unknown_contamination_flag_rejected", rejects(lambda p: p["sessions"][0].update({"contamination_flags": ["vibes"]})))
    check("schema.duplicate_topic_id_rejected", rejects(lambda p: p["topics"].append(dict(p["topics"][0]))))
    check("schema.sentinel_collision_rejected", rejects(lambda p: p["topics"][0].update({"topic_id": "novel"})))
    check("schema.bad_hash_rejected", rejects(lambda p: p["sessions"][0].update({"project_id_hash": "sha256:nope"})))
    check("schema.unknown_request_field_rejected", (lambda: _request_rejects(valid_request))())

    # 2. adapter determinism + one-batch rule
    backend_a = SyntheticBackend()
    backend_b = SyntheticBackend()
    prompt_hash = sha256_hex("identical prompt text")
    check(
        "adapter.deterministic_scores",
        backend_a.pair_score(prompt_hash, "novel", "topic") == backend_b.pair_score(prompt_hash, "novel", "topic"),
    )
    check(
        "adapter.hash_only_of_prompt",
        backend_a.pair_score(prompt_hash, "novel", "topic") == backend_a.pair_score(sha256_hex("identical   prompt text"), "novel", "topic"),
        "canonicalization makes spacing irrelevant",
    )
    candidates = CandidateSet.from_dict(valid_candidates)
    handle = _handle_for(valid_request)
    rows = backend_a.batch_score(handle, candidates, "topic")
    check("adapter.one_batch_call", backend_a.batch_calls["topic"] == 1)
    check(
        "adapter.sentinels_scored_in_batch",
        {r["candidate"] for r in rows} == set(candidates.topic_candidate_ids()),
    )

    gate = OpenRouterBackend(enabled=False)
    try:
        gate.propose_taxonomy(handle, [prompt_hash])
        check("adapter.disabled_openrouter_raises", False, "no exception raised")
    except DisabledByPolicy:
        check("adapter.disabled_openrouter_raises", True)

    enabled = OpenRouterBackend(enabled=True)
    try:
        enabled.batch_score(handle, candidates, "topic")
        check("adapter.enabled_openrouter_refuses_batch_scoring", False, "no exception raised")
    except BackendError:
        check("adapter.enabled_openrouter_refuses_batch_scoring", True)
    try:
        enabled.propose_taxonomy(handle, [prompt_hash])
        check("adapter.no_transport_no_network", False, "no exception raised")
    except BackendError:
        check("adapter.no_transport_no_network", True)
    payload = enabled.build_proposal_payload(handle, [prompt_hash])
    check("adapter.payload_is_hash_only", True)
    try:
        assert_hash_only_payload(payload)
        check("adapter.payload_passes_hash_only_assertion", True)
    except PrivacyViolationError as exc:
        check("adapter.payload_passes_hash_only_assertion", False, repr(exc))
    try:
        assert_hash_only_payload({"content": "the user asked about billing refunds"})
        check("adapter.poisoned_payload_rejected", False, "free text accepted")
    except PrivacyViolationError:
        check("adapter.poisoned_payload_rejected", True)
    check(
        "adapter.payload_has_no_prompt_text",
        "identical prompt text" not in json.dumps(payload),
    )

    # 3. policy unit checks
    for item in _policy_checks():
        if len(item) == 3:
            name, ok, detail = item
        else:
            name, ok = item
            detail = ""
        check(name, ok, detail)

    # 4. labels
    for item in _label_checks():
        check(item[0], item[1], item[2] if len(item) > 2 else "")

    # 5. receipts
    for item in _receipt_checks(valid_request, valid_candidates):
        check(item[0], item[1], item[2] if len(item) > 2 else "")

    # 6. cli
    for item in _cli_checks():
        check(item[0], item[1], item[2] if len(item) > 2 else "")

    # 7. fixtures determinism + manifest hash
    small = _small_cases(52)
    payload_small = fg.cases_payload(small)
    regenerated = fg.cases_payload(_small_cases(52))
    check("fixtures.regeneration_is_byte_identical", regenerated == payload_small)
    manifest_path = os.path.join(fixtures_dir, "manifest.json")
    cases_path = os.path.join(fixtures_dir, "cases.json")
    if not (os.path.isfile(manifest_path) and os.path.isfile(cases_path)):
        # self-test is self-contained: materialize a small fixture set on disk
        fg.write_fixtures(fixtures_dir, total=52)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(cases_path, "r", encoding="utf-8") as fh:
        body = fh.read()
    check(
        "fixtures.manifest_hash_matches_cases_file",
        hashlib.sha256(body.encode("utf-8")).hexdigest() == manifest.get("cases_sha256"),
    )
    on_disk = json.loads(body)
    # `total` is an input to the generator (it sets the tuning/holdout boundary),
    # so the prefix must be regenerated with the SAME total as the on-disk file.
    prefix = [fg.build_case(i, len(on_disk)) for i in range(min(52, len(on_disk)))]
    check(
        "fixtures.on_disk_prefix_matches_regeneration",
        on_disk[: len(prefix)] == prefix,
        "regenerated with total=%d" % len(on_disk),
    )

    # 8. calibration + determinism on a small slice
    subset = fg.generate_cases(total=26)
    cal = CalibrationModel(model_id="platt-v1", active=True, a_topic=2.0, b_topic=-1.0, a_session=2.0, b_session=-1.0)
    first = [_strip_timing(r) for r in route_all(subset, cal)]
    second = [_strip_timing(r) for r in route_all(subset, cal)]
    check("harness.deterministic_routing", first == second)
    uncal = [_strip_timing(r) for r in route_all(subset, CalibrationModel())]
    check(
        "harness.uncalibrated_never_automatic",
        all(r["band"] != "automatic" for r in uncal),
        "locked #5: no calibration -> automatic unreachable",
    )
    fit = fit_platt([(0.9, 1), (0.85, 1), (0.2, 0), (0.3, 0), (0.88, 1), (0.25, 0)])
    check("harness.platt_fit_separates", fit["a"] > 0 and fit["b"] < 0, json.dumps({k: round(v, 3) for k, v in fit.items() if isinstance(v, float)}))
    metrics = compute_metrics([dict(r) for r in route_all(subset, cal)])
    check("harness.metrics_present", "accuracy" in metrics["topic"] and "fragmentation" in metrics["session"])
    bands = {r["band"] for r in route_all(subset, cal)}
    check("harness.band_values_are_closed_enum", bands <= {"automatic", "fallback_escalate", "abstain_or_new_session"})
    receipt_ok = True
    for r in route_all(subset, cal):
        try:
            assert_no_free_text(r["result"].receipt)
        except Exception:  # noqa: BLE001
            receipt_ok = False
    check("harness.receipts_pass_privacy_assertion", receipt_ok)

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print("%-58s %s%s" % (name, "PASS" if ok else "FAIL", ("  <- " + detail) if detail else ""))
    print("SELFTEST %s checks=%d failed=%d" % ("PASS" if not failed else "FAIL", len(checks), len(failed)))
    return 0 if not failed else 1


def _strip_timing(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fully comparable view of a routed row (timing fields removed)."""
    receipt = dict(row["result"].receipt)
    receipt.pop("backend_latency_ms", None)
    return {
        "case_id": row["case"]["case_id"],
        "band": row["band"],
        "decision": row["decision"],
        "topic_top": row["topic_top"],
        "session_top": row["session_top"],
        "session_band": row["session_band"],
        "escalation_code": row["escalation_code"],
        "token_cost_estimate": row["token_cost_estimate"],
        "decision_dict": row["result"].decision.to_dict(),
        "receipt": receipt,
    }


def _handle_for(request: Dict[str, Any]):
    from ordaprompt_router.adapter import RequestHandle

    return RequestHandle(
        request_id=request["request_id"],
        prompt_hash=request["prompt_hash"],
        project_id_hash=request["project_id_hash"],
        config_rev=request["config_rev"],
        context_hash=request["context_hash"],
    )


def _request_rejects(valid_request: Dict[str, Any]) -> bool:
    payload = json.loads(json.dumps(valid_request))
    payload["directive"] = "direct:foo"
    try:
        ClassificationRequest.from_dict(payload)
    except SchemaError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _policy_checks() -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    # identity active calibrator: c == raw exactly, so thresholds are tested in
    # raw-score units (a=1, b=0 -> sigma(logit(s)) == s)
    cal = CalibrationModel(model_id="platt-v1", active=True, a_topic=1.0, b_topic=0.0, a_session=1.0, b_session=0.0)
    thresholds = Thresholds()
    request = ClassificationRequest.from_dict(
        {
            "schema": "ordapilot.classification_request/1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "ts": "2026-09-22T10:00:00Z",
            "prompt_hash": "sha256:" + "ab" * 32,
            "context_hash": None,
            "context_token_estimate": 0,
            "project_id_hash": "sha256:" + "cd" * 32,
            "config_rev": "1.0.0",
            "explicit_directive": None,
        }
    )

    def candidates(topics, sessions):
        return CandidateSet.from_dict(
            {
                "schema": "ordapilot.candidate_set/1",
                "request_id": request.request_id,
                "topics": [
                    {"topic_id": t, "label_state": "promoted", "label_age_days": 1, "evidence_count": 1}
                    for t in topics
                ],
                "topic_sentinels": ["novel", "ambiguous"],
                "sessions": sessions,
                "session_sentinels": ["new_session"],
            }
        )

    def run(topic_scores, session_scores=None, calibration=cal, sessions=None, legacy=False):
        backend = TableBackend(topic_scores, session_scores or {})
        config = RouterConfig(
            thresholds=thresholds, calibration=calibration, backend=backend,
            label_registry=LabelRegistry(), legacy_fallback=legacy,
        )
        return route(request, candidates(["topic-auth", "topic-billing"], sessions or []), config)

    # automatic requires BOTH min calibrated score and margin
    res = run({"topic-auth": 0.9, "topic-billing": 0.89, "novel": 0.1, "ambiguous": 0.1})
    out.append(("policy.automatic_needs_margin", res.decision.band == "fallback_escalate" and res.topic_escalation_code == "below_margin"))
    res = run({"topic-auth": 0.7, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1})
    out.append(("policy.automatic_needs_min_score", res.decision.band == "fallback_escalate" and res.topic_escalation_code == "below_min_score"))
    res = run({"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1})
    out.append(("policy.automatic_when_both_pass", res.decision.band == "automatic" and res.decision.decision == "topic_assign"))

    # calibration gate
    res = run({"topic-auth": 0.99, "topic-billing": 0.1, "novel": 0.05, "ambiguous": 0.05}, calibration=CalibrationModel())
    out.append(("policy.calibration_gate_blocks_automatic", res.decision.band != "automatic" and res.topic_escalation_code == "calibration_missing"))

    # ambiguous wins -> fallback
    res = run({"topic-auth": 0.2, "topic-billing": 0.1, "novel": 0.1, "ambiguous": 0.95})
    out.append(("policy.ambiguous_top_escalates", res.decision.band == "fallback_escalate" and res.topic_escalation_code == "ambiguous_won"))

    # low confidence everywhere -> abstain
    res = run({"topic-auth": 0.4, "topic-billing": 0.35, "novel": 0.3, "ambiguous": 0.2})
    out.append(("policy.low_confidence_abstains", res.decision.band == "abstain_or_new_session"))

    # novel win with the taxonomy gate unavailable -> degrade to fallback
    res = run({"topic-auth": 0.2, "topic-billing": 0.1, "novel": 0.95, "ambiguous": 0.05})
    out.append(("policy.novel_degrades_without_gate", res.decision.band == "fallback_escalate" and res.topic_escalation_code == "taxonomy_rejected"))

    # contaminated session forces new_session even at maximum similarity
    contaminated = [{"session_id": "sess-x", "project_id_hash": "sha256:" + "cd" * 32,
                     "context_token_cost": 100, "idle_days": 1,
                     "contamination_flags": ["project_mismatch"]}]
    res = run(
        {"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1},
        {"sess-x": 0.99, "new_session": 0.1},
        sessions=contaminated,
    )
    out.append(("policy.contamination_forces_new_session", res.session_band == "abstain_or_new_session" and res.session_escalation_code == "contamination_high"))

    # clean high-similarity cheap session -> reuse
    clean = [{"session_id": "sess-x", "project_id_hash": "sha256:" + "cd" * 32,
              "context_token_cost": 100, "idle_days": 1, "contamination_flags": []}]
    res = run(
        {"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1},
        {"sess-x": 0.99, "new_session": 0.1},
        sessions=clean,
    )
    out.append(("policy.clean_session_reused", res.session_band == "automatic" and res.decision.decision == "session_reuse"))

    # token cost is priced: same similarity, fat context loses the margin
    fat = [{"session_id": "sess-x", "project_id_hash": "sha256:" + "cd" * 32,
            "context_token_cost": 8000, "idle_days": 1, "contamination_flags": []}]
    res = run(
        {"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1},
        {"sess-x": 0.99, "new_session": 0.1},
        sessions=fat,
    )
    out.append(("policy.token_cost_reduces_utility",
                res.decision.session_scores[0]["utility"] < 1.0 - 0.30 + 1e-9
                and res.session_band == "automatic",
                "utility=%.4f" % res.decision.session_scores[0]["utility"]))

    # tie on the session side -> new_session (strict policy)
    res = run(
        {"topic-auth": 0.9, "topic-billing": 0.2, "novel": 0.1, "ambiguous": 0.1},
        {"sess-x": 0.60, "new_session": 0.60},
        sessions=clean,
    )
    out.append(("policy.session_tie_new_session", res.session_band != "automatic"))

    # explicit directive short-circuit
    directive = ClassificationRequest.from_dict(dict(
        _request_dict(), explicit_directive="direct:topic-auth"
    ))
    backend = TableBackend({"topic-auth": 0.99, "topic-billing": 0.1, "novel": 0.1, "ambiguous": 0.1})
    config = RouterConfig(thresholds=thresholds, calibration=cal, backend=backend, label_registry=LabelRegistry())
    res = route(directive, candidates(["topic-auth"], []), config)
    out.append(("policy.explicit_directive_short_circuits", res.decision.decision == "escalate"
                and res.escalation_code == "explicit_directive"
                and backend.batch_calls["topic"] == 0))

    # legacy rollback
    res = run({"topic-auth": 0.99, "topic-billing": 0.1, "novel": 0.1, "ambiguous": 0.1}, legacy=True)
    out.append(("policy.legacy_rollback_escalates", res.decision.decision == "escalate" and res.escalation_code == "legacy_mode"))

    # backend failure -> fail-escalate, never fail-open
    failing = RouterConfig(thresholds=thresholds, calibration=cal, backend=FailingBackend({}), label_registry=LabelRegistry())
    res = route(request, candidates(["topic-auth"], []), failing)
    out.append(("policy.backend_error_fail_escalate", res.decision.band == "fallback_escalate"
                and res.decision.decision == "escalate" and res.topic_escalation_code == "backend_error"))
    return out


def _request_dict() -> Dict[str, Any]:
    return {
        "schema": "ordapilot.classification_request/1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "ts": "2026-09-22T10:00:00Z",
        "prompt_hash": "sha256:" + "ab" * 32,
        "context_hash": None,
        "context_token_estimate": 0,
        "project_id_hash": "sha256:" + "cd" * 32,
        "config_rev": "1.0.0",
        "explicit_directive": None,
    }


def _label_checks() -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    registry = LabelRegistry()
    registry.create_provisional("topic-new")
    t1 = registry.register_corroboration("topic-new", "sess-1", "2026-09-01", 0.90, 0.85)
    t2 = registry.register_corroboration("topic-new", "sess-2", "2026-09-02", 0.91, 0.85)
    t3 = registry.register_corroboration("topic-new", "sess-3", "2026-09-03", 0.92, 0.85)
    out.append(("labels.promotes_after_k3_two_sessions_two_days",
                t1 is None and t2 is None and t3 == "topic-new:promoted" and registry.state_of("topic-new") == "promoted"))

    same_day = LabelRegistry()
    same_day.create_provisional("topic-same")
    results = [
        same_day.register_corroboration("topic-same", "sess-1", "2026-09-01", 0.90, 0.85),
        same_day.register_corroboration("topic-same", "sess-2", "2026-09-01", 0.90, 0.85),
        same_day.register_corroboration("topic-same", "sess-3", "2026-09-01", 0.90, 0.85),
    ]
    out.append(("labels.single_day_does_not_promote", all(r is None for r in results) and same_day.state_of("topic-same") == "provisional"))

    low = LabelRegistry()
    low.create_provisional("topic-low")
    low.register_corroboration("topic-low", "sess-1", "2026-09-01", 0.50, 0.85)
    out.append(("labels.below_tau_not_corroborating", len(low.labels["topic-low"].corroborations) == 0))

    override = LabelRegistry()
    override.create_provisional("topic-new")
    for day, sess in (("2026-09-01", "sess-1"), ("2026-09-02", "sess-2"), ("2026-09-03", "sess-3")):
        override.register_corroboration("topic-new", sess, day, 0.90, 0.85)
    code = override.record_human_override("topic-new")
    out.append(("labels.human_override_demotes", code == "topic-new:demoted" and override.state_of("topic-new") == "provisional"))

    contradict = LabelRegistry()
    contradict.create_provisional("topic-c")
    for day, sess in (("2026-09-01", "sess-1"), ("2026-09-02", "sess-2"), ("2026-09-03", "sess-3")):
        contradict.register_corroboration("topic-c", sess, day, 0.90, 0.85)
    d1 = contradict.record_contradiction("topic-c")
    d2 = contradict.record_contradiction("topic-c")
    out.append(("labels.two_contradictions_demote", d1 is None and d2 == "topic-c:demoted"))

    arch = LabelRegistry()
    code = arch.record_human_override("topic-zero")
    out.append(("labels.override_without_evidence_archives", code == "topic-zero:archived"))
    return out


def _receipt_checks(valid_request: Dict[str, Any], valid_candidates: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    import tempfile

    out: List[Tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = ReceiptStore(tmp)
        request = ClassificationRequest.from_dict(valid_request)
        candidates = CandidateSet.from_dict(valid_candidates)
        config = RouterConfig(calibration=CalibrationModel(), backend=TableBackend(
            {"topic-auth": 0.9, "novel": 0.1, "ambiguous": 0.1},
            {"sess-0001": 0.9, "new_session": 0.1},
        ), label_registry=LabelRegistry())
        result = route(request, candidates, config)
        store.append(result.receipt)
        out.append(("receipts.valid_receipt_appended", store.count() == 1))
        out.append(("receipts.no_prompt_text_persisted",
                    receipt_prompt_text_absent(result.receipt, "identical prompt text")))

        poisoned = dict(result.receipt)
        poisoned["rationale"] = "the user asked about billing refunds"
        try:
            store.append(poisoned)
            out.append(("receipts.unknown_field_rejected", False, "accepted"))
        except SchemaError:
            out.append(("receipts.unknown_field_rejected", True))

        free_text = dict(result.receipt)
        free_text["topic_candidate_ids"] = ["the user asked about billing refunds"]
        try:
            store.append(free_text)
            out.append(("receipts.free_text_rejected", False, "accepted"))
        except (SchemaError, PrivacyViolationError):
            out.append(("receipts.free_text_rejected", True))

        snippet = dict(result.receipt)
        snippet["label_state_touched"] = ["topic-auth:provisional-ish waffle"]
        try:
            store.append(snippet)
            out.append(("receipts.prose_in_transition_code_rejected", False, "accepted"))
        except (SchemaError, PrivacyViolationError):
            out.append(("receipts.prose_in_transition_code_rejected", True))

        ok, err = store.append_safe(poisoned, result.receipt["receipt_id"], result.receipt["ts"])
        out.append(("receipts.append_safe_logs_reject_row",
                    (not ok) and len(store.read_rejects()) == 1
                    and store.read_rejects()[0]["escalation_code"] == "schema_reject"))
        out.append(("receipts.store_append_only_count_stable", store.count() == 1))

        with open(store.path, "r", encoding="utf-8") as handle:
            body = handle.read()
        out.append(("receipts.store_file_is_jsonl_hashes_only",
                    "prompt" not in body.replace("prompt_hash", "") and body.count("\n") == 1))
        out.append(("receipts.store_sha256_available", len(store_sha256(store.path)) == 64))
    return out


def _cli_checks() -> List[Tuple[str, bool, str]]:
    import contextlib
    import io
    import tempfile

    from ordaprompt_router import cli

    out: List[Tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        request_path = os.path.join(tmp, "req.json")
        candidates_path = os.path.join(tmp, "cand.json")
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump(_request_dict(), handle)
        with open(candidates_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": "ordapilot.candidate_set/1",
                    "request_id": "11111111-1111-4111-8111-111111111111",
                    "topics": [{"topic_id": "topic-auth", "label_state": "promoted", "label_age_days": 2, "evidence_count": 2}],
                    "topic_sentinels": ["novel", "ambiguous"],
                    "sessions": [
                        {"session_id": "sess-0001", "project_id_hash": "sha256:" + "cd" * 32,
                         "context_token_cost": 500, "idle_days": 2, "contamination_flags": []}
                    ],
                    "session_sentinels": ["new_session"],
                },
                handle,
            )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["classify", "--request", request_path, "--candidates", candidates_path,
                             "--receipts-dir", tmp])
        try:
            payload = json.loads(buffer.getvalue())
            parsed = "routing_decision" in payload and "routing_receipt" in payload
        except ValueError:
            parsed = False
        out.append(("cli.classify_exit_0_and_json", code == 0 and parsed))

        bad_path = os.path.join(tmp, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            handle.write('{"schema":"ordapilot.classification_request/1"}')
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["classify", "--request", bad_path])
        out.append(("cli.malformed_request_exit_2", code == 2))

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["route-session", "--request", request_path,
                             "--topics", "topic-auth", "--sessions", "sess-0001"])
        out.append(("cli.route_session_shorthand_topics_exit_0", code == 0))
    return out


if __name__ == "__main__":
    sys.exit(main())
