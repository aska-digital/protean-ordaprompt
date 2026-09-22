"""Deterministic SYNTHETIC fixture generator for the offline eval harness.

Fixtures are searched, not invented: for each case the generator scans prompt
nonces until the *deterministic* :class:`SyntheticBackend` hash scorer produces
the score structure the case needs (clear winner, near tie, novel win,
ambiguous win, low confidence, contaminated / expensive / near-tie session).
The achieved raw scores are recorded in the fixture and replayed verbatim by
the harness, so the eval never depends on re-running the search.

HARD requirements (semantic, must hold) decide the fixture's ground truth.
SOFT targets (best effort, recorded in ``soft_met``) only shape how clean the
case is; they never change the truth label.

Everything here is pure stdlib, offline and byte-reproducible.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ordaprompt_router.adapter import SyntheticBackend, sha256_hex
from ordaprompt_router.schemas import CandidateSet, ClassificationRequest

GENERATOR_VERSION = "fixturegen-1.0.0"
DEFAULT_TOTAL_CASES = 400
DEFAULT_BUDGET = 40000

TOPIC_POOL = (
    "topic-auth",
    "topic-billing",
    "topic-ci-pipeline",
    "topic-db-migration",
    "topic-docs-site",
    "topic-infra-net",
    "topic-payments",
    "topic-search-index",
)

TOPICS_PER_CASE = 5

#: (topic kind, session kind) plan; repeated to fill the requested case count.
#: clear_topic/clean_reuse dominates so that >=50 genuinely-automatic cases
#: land in the holdout split (leo-arch.md section 7 unlock gate).
CASE_PLAN: Tuple[Tuple[str, str], ...] = (
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "clean_reuse"),
    ("clear_topic", "no_sessions"),
    ("clear_topic", "no_sessions"),
    ("near_tie", "clean_reuse"),
    ("near_tie", "clean_reuse"),
    ("near_tie", "no_sessions"),
    ("novel", "clean_reuse"),
    ("novel", "clean_reuse"),
    ("novel", "no_sessions"),
    ("ambiguous", "clean_reuse"),
    ("ambiguous", "clean_reuse"),
    ("low_confidence", "clean_reuse"),
    ("clear_topic", "contaminated"),
    ("clear_topic", "expensive_context"),
    ("clear_topic", "session_near_tie"),
    ("clear_topic", "new_session_argmax"),
)

TUNING_FRACTION = 2.0 / 3.0
REQUEST_ID_NAMESPACE = "0f0f0f0f-0000-4000-8000-"


class FixtureSearchError(RuntimeError):
    """Raised when no prompt nonce satisfies a case's hard requirements."""


# --------------------------------------------------------------------------
# Small deterministic helpers
# --------------------------------------------------------------------------


def _request_id(index: int) -> str:
    return REQUEST_ID_NAMESPACE + "%012d" % index


def _prompt_text(case_id: str, nonce: int) -> str:
    return "synthetic fixture prompt %s nonce %d" % (case_id, nonce)


def _topic_ids_for(index: int) -> List[str]:
    start = (index * 3) % len(TOPIC_POOL)
    return [TOPIC_POOL[(start + offset) % len(TOPIC_POOL)] for offset in range(TOPICS_PER_CASE)]


def _session_specs(kind: str, case_id: str) -> List[Dict[str, Any]]:
    a = "sess-%s-a" % case_id
    b = "sess-%s-b" % case_id
    if kind == "no_sessions":
        return []
    if kind == "clean_reuse":
        return [
            {"session_id": a, "context_token_cost": 400, "idle_days": 1, "contamination_flags": []},
            {"session_id": b, "context_token_cost": 300, "idle_days": 4, "contamination_flags": []},
        ]
    if kind == "contaminated":
        return [
            {"session_id": a, "context_token_cost": 400, "idle_days": 1, "contamination_flags": ["project_mismatch"]},
            {"session_id": b, "context_token_cost": 300, "idle_days": 4, "contamination_flags": []},
        ]
    if kind == "expensive_context":
        return [
            {"session_id": a, "context_token_cost": 40000, "idle_days": 1, "contamination_flags": ["stale_context"]},
            {"session_id": b, "context_token_cost": 300, "idle_days": 4, "contamination_flags": []},
        ]
    if kind in ("session_near_tie", "new_session_argmax"):
        return [
            {"session_id": a, "context_token_cost": 400, "idle_days": 1, "contamination_flags": []},
            {"session_id": b, "context_token_cost": 300, "idle_days": 2, "contamination_flags": []},
        ]
    raise ValueError("unknown session kind %r" % (kind,))


def _ranked(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _margin(scores: Dict[str, float], target: str) -> float:
    ranked = _ranked(scores)
    others = [value for candidate, value in ranked if candidate != target]
    return scores[target] - max(others) if others else 1.0


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------


def _topic_predicate(kind: str, index: int, scores: Dict[str, float], topic_ids: List[str]) -> Tuple[bool, bool, Optional[str]]:
    """Return (hard_ok, soft_ok, truth_topic)."""
    ranked = _ranked(scores)
    top = ranked[0][0]
    second = ranked[1][0]
    target = topic_ids[index % len(topic_ids)]
    ambiguous_rank = [c for c, _ in ranked].index("ambiguous")
    ambiguous_not_top2 = ambiguous_rank >= 2

    if kind == "clear_topic":
        hard = (
            top == target
            and _margin(scores, target) >= 0.10
            and ambiguous_not_top2
            and scores[target] >= 0.86
        )
        soft = hard and scores[target] >= 0.92
        return hard, soft, target

    if kind == "near_tie":
        gap = ranked[0][1] - ranked[1][1]
        hard = (
            gap <= 0.01
            and 0.55 <= ranked[1][1]
            and ranked[0][1] <= 0.80
            and ambiguous_not_top2
        )
        return hard, hard, None

    if kind == "novel":
        hard = (
            top == "novel"
            and _margin(scores, "novel") >= 0.10
            and scores["novel"] >= 0.86
            and ambiguous_not_top2
        )
        soft = hard and scores["novel"] >= 0.92
        return hard, soft, "novel"

    if kind == "ambiguous":
        hard = top == "ambiguous" and scores["ambiguous"] >= 0.72
        return hard, hard, None

    if kind == "low_confidence":
        hard = max(scores.values()) <= 0.56
        return hard, hard, None

    raise ValueError("unknown topic kind %r" % (kind,))


def _session_predicate(
    kind: str, scores: Dict[str, float], session_specs: List[Dict[str, Any]]
) -> Tuple[bool, bool, Optional[str]]:
    """Return (hard_ok, soft_ok, truth_session)."""
    if kind == "no_sessions":
        return True, True, "new_session"
    ids = [spec["session_id"] for spec in session_specs]
    a, b = ids[0], ids[1]
    sa, sb = scores[a], scores[b]

    if kind == "clean_reuse":
        hard = sa > sb and (sa - sb) >= 0.10 and sb <= 0.50 and sa >= 0.85
        soft = hard and sa >= 0.90
        return hard, soft, a

    if kind == "contaminated":
        # a highly similar but flagged session: K >= 0.5 must force new_session
        hard = sa > sb and (sa - sb) >= 0.10 and sb <= 0.50 and sa >= 0.85
        soft = hard and sa >= 0.90
        return hard, soft, "new_session"

    if kind == "expensive_context":
        hard = sa > sb and (sa - sb) >= 0.10 and sb <= 0.50 and sa >= 0.85
        soft = hard and sa >= 0.90
        # policy-priced case: whether reuse survives the token-cost term depends
        # on the fitted calibrator, so no session truth is asserted here
        return hard, soft, None

    if kind == "session_near_tie":
        gap = abs(sa - sb)
        hard = gap <= 0.02 and 0.55 <= sa <= 0.75 and 0.55 <= sb <= 0.75
        return hard, hard, "new_session"

    if kind == "new_session_argmax":
        hard = sa <= 0.45 and sb <= 0.45
        return hard, hard, "new_session"

    raise ValueError("unknown session kind %r" % (kind,))


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------


def build_case(index: int, total: int, budget: int = DEFAULT_BUDGET) -> Dict[str, Any]:
    topic_kind, session_kind = CASE_PLAN[index % len(CASE_PLAN)]
    case_id = "case-%04d" % index
    topic_ids = _topic_ids_for(index)
    session_specs = _session_specs(session_kind, case_id)
    session_ids = [spec["session_id"] for spec in session_specs]
    backend = SyntheticBackend()

    split = "tuning" if (index / float(total)) < TUNING_FRACTION else "holdout"
    request_id = _request_id(index)

    for nonce in range(budget):
        prompt_text = _prompt_text(case_id, nonce)
        prompt_hash = sha256_hex(prompt_text)
        topic_scores = {t: backend.pair_score(prompt_hash, t, "topic") for t in topic_ids + ["novel", "ambiguous"]}
        t_hard, t_soft, truth_topic = _topic_predicate(topic_kind, index, topic_scores, topic_ids)
        if not t_hard:
            # early rejection: session scoring only runs on topic-hard nonces
            continue
        session_scores = {
            s: backend.pair_score(prompt_hash, s, "session") for s in session_ids + ["new_session"]
        }
        s_hard, s_soft, truth_session = _session_predicate(session_kind, session_scores, session_specs)
        if not s_hard:
            continue
        # first nonce satisfying the HARD (semantic) requirements wins; the
        # soft flag is a post-hoc diagnostic, not a search target
        return _assemble(
            index=index,
            case_id=case_id,
            split=split,
            topic_kind=topic_kind,
            session_kind=session_kind,
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            request_id=request_id,
            topic_ids=topic_ids,
            session_specs=session_specs,
            topic_scores=topic_scores,
            session_scores=session_scores,
            truth_topic=truth_topic,
            truth_session=truth_session,
            soft_met=bool(t_soft and s_soft),
            nonce=nonce,
        )
    raise FixtureSearchError(
        "%s (%s/%s): no nonce satisfied the hard requirements within %d trials"
        % (case_id, topic_kind, session_kind, budget)
    )


def _assemble(
    index: int,
    case_id: str,
    split: str,
    topic_kind: str,
    session_kind: str,
    prompt_text: str,
    prompt_hash: str,
    request_id: str,
    topic_ids: List[str],
    session_specs: List[Dict[str, Any]],
    topic_scores: Dict[str, float],
    session_scores: Dict[str, float],
    truth_topic: Optional[str],
    truth_session: Optional[str],
    soft_met: bool,
    nonce: int,
) -> Dict[str, Any]:
    project_hash = sha256_hex("project-%d" % (index % 7))
    day = "2026-09-%02d" % (1 + (index % 28))
    ts = "%sT%02d:00:00Z" % (day, index % 24)
    request = ClassificationRequest(
        request_id=request_id,
        ts=ts,
        prompt_hash=prompt_hash,
        project_id_hash=project_hash,
        config_rev="1.0.0",
        context_hash=sha256_hex("context-%s" % case_id),
        context_token_estimate=1200,
        explicit_directive=None,
    )
    candidates = CandidateSet.from_dict(
        {
            "schema": "ordapilot.candidate_set/1",
            "request_id": request_id,
            "topics": [
                {"topic_id": t, "label_state": "promoted", "label_age_days": 5 + i, "evidence_count": 3 + i}
                for i, t in enumerate(topic_ids)
            ],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": [
                {
                    "session_id": spec["session_id"],
                    "project_id_hash": project_hash,
                    "context_token_cost": spec["context_token_cost"],
                    "idle_days": spec["idle_days"],
                    "contamination_flags": list(spec["contamination_flags"]),
                }
                for spec in session_specs
            ],
            "session_sentinels": ["new_session"],
        }
    )
    return {
        "case_id": case_id,
        "split": split,
        "topic_kind": topic_kind,
        "session_kind": session_kind,
        # the router itself gates the session side on the presence of live
        # sessions; the fixture always routes with session comparison enabled
        "include_session": True,
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "nonce": nonce,
        "soft_met": soft_met,
        "truth": {"topic": truth_topic, "session": truth_session},
        "raw": {"topic": topic_scores, "session": session_scores},
        "request": request.to_dict(),
        "candidates": candidates.to_dict(),
    }


# --------------------------------------------------------------------------
# Generation / IO
# --------------------------------------------------------------------------


def generate_cases(total: int = DEFAULT_TOTAL_CASES, budget: int = DEFAULT_BUDGET) -> List[Dict[str, Any]]:
    return [build_case(index, total, budget=budget) for index in range(total)]


def cases_payload(cases: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(list(cases), sort_keys=True, indent=1)


def manifest_for(cases: Sequence[Dict[str, Any]], payload: str) -> Dict[str, Any]:
    import hashlib

    split_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    soft = 0
    for case in cases:
        split_counts[case["split"]] = split_counts.get(case["split"], 0) + 1
        key = "%s/%s" % (case["topic_kind"], case["session_kind"])
        kind_counts[key] = kind_counts.get(key, 0) + 1
        soft += 1 if case["soft_met"] else 0
    return {
        "generator_version": GENERATOR_VERSION,
        "cases": len(cases),
        "split_counts": split_counts,
        "kind_counts": kind_counts,
        "soft_targets_met": soft,
        "soft_target_rate": round(soft / float(len(cases)), 4) if cases else 0.0,
        "cases_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def write_fixtures(directory: str, total: int = DEFAULT_TOTAL_CASES, budget: int = DEFAULT_BUDGET) -> Dict[str, Any]:
    cases = generate_cases(total=total, budget=budget)
    payload = cases_payload(cases)
    manifest = manifest_for(cases, payload)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cases.json"), "w", encoding="utf-8") as handle:
        handle.write(payload)
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def load_fixtures(directory: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(os.path.join(directory, "cases.json"), "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    manifest_path = os.path.join(directory, "manifest.json")
    manifest: Dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    return cases, manifest


def fixture_backend(cases: Sequence[Dict[str, Any]]) -> SyntheticBackend:
    """Replay backend: recorded raw scores verbatim, hash scorer elsewhere."""
    table: Dict[Tuple[str, str], Dict[str, float]] = {}
    for case in cases:
        for surface in ("topic", "session"):
            table[(case["prompt_hash"], surface)] = dict(case["raw"][surface])
    return SyntheticBackend(fixture_table=table)


def verify_fixtures(directory: str, total: int = DEFAULT_TOTAL_CASES, budget: int = DEFAULT_BUDGET) -> Tuple[bool, Dict[str, Any]]:
    """Regenerate in memory and prove the on-disk fixtures are byte-identical."""
    cases = generate_cases(total=total, budget=budget)
    payload = cases_payload(cases)
    manifest = manifest_for(cases, payload)
    with open(os.path.join(directory, "cases.json"), "r", encoding="utf-8") as handle:
        on_disk = handle.read()
    return on_disk == payload, manifest
