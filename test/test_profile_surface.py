"""Offline tests for the `profile` routing surface (phase 2, D8).

Every test here runs with an INJECTED transport (`MockTransport`) or with the local
deterministic adapter plus a fixture table.  **No socket is opened, no third-party
service is called, no real credential is used, and no environment secret is read**:
the environment key used below is a literal non-secret placeholder.  A test that
needed a network call would fail loudly (the injected transport raises when it is
called more often than the test allows, and the registry never builds the stdlib
transport here).

Run: python3 -m unittest discover -s test -p 'test_*.py' -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ordaprompt_router import (  # noqa: E402
    PROFILE_SENTINELS,
    SURFACES,
    BackendError,
    CandidateSet,
    ClassificationRequest,
    OpenRouterBackend,
    PrivacyViolationError,
    ProfileCandidate,
    ProfileMetadataViolation,
    ProviderConfigError,
    ProviderContractError,
    ProviderError,
    ProviderRegistry,
    RequestHandle,
    RouterConfig,
    SyntheticBackend,
    Transport,
    TransportRequest,
    TransportResponse,
    assert_profile_payload_whitelisted,
    route,
    sha256_hex,
)
from ordaprompt_router.adapter import ClassificationBackend  # noqa: E402
from ordaprompt_router.providers import (  # noqa: E402
    PROFILE_REQUEST_FIELDS,
    OpenAICompatibleBackend,
    ProviderRow,
    parse_provider_row,
    validate_batch_reply,
)
from ordaprompt_router.receipts import ReceiptStore, assert_no_free_text  # noqa: E402
from ordaprompt_router.router import CalibrationModel  # noqa: E402
from ordaprompt_router.schemas import (  # noqa: E402
    ESCALATION_CODES,
    RECEIPT_FIELDS,
    RoutingDecision,
    SchemaError,
    validate_receipt,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

PROMPT_TEXT = "REFUND my invoice for the billing cycle, this is urgent"
#include a marker string that must never reach the wire
PROFILE_SECRET_MARKER = "MARKER-profile-description-must-not-leave"
REQUEST_ID = "11111111-1111-4111-8111-111111111111"
CONFIG_REV = "1.0.0"
PROJECT_HASH = "sha256:" + "cd" * 32
KEY_ENV = "ORDAPROMPT_TEST_PROFILE_KEY"
KEY_VALUE = "«redacted:sk-…»"

TOPIC_IDS = ["topic-auth", "topic-billing", "novel", "ambiguous"]
SESSION_IDS = ["sess-a", "new_session"]
PROFILE_IDS = ["profile-alpha", "profile-beta"]
PROFILE_BALLOT = PROFILE_IDS + list(PROFILE_SENTINELS)  # sentinel always last
TOPIC_SCORE_FIELDS = ("candidate", "raw")
SESSION_SCORE_FIELDS = ("candidate", "raw")


def make_request():
    return ClassificationRequest.from_dict(
        {
            "schema": "ordapilot.classification_request/1",
            "request_id": REQUEST_ID,
            "ts": "2026-09-22T10:00:00Z",
            "prompt_hash": sha256_hex(PROMPT_TEXT),
            "context_hash": None,
            "context_token_estimate": 0,
            "project_id_hash": PROJECT_HASH,
            "config_rev": CONFIG_REV,
            "explicit_directive": None,
        }
    )


def profile_block(ids=None, **over):
    rows = []
    for index, profile_id in enumerate(ids if ids is not None else PROFILE_IDS):
        row: Dict[str, Any] = {
            "profile_id": profile_id,
            "scope": "project",
            "privacy_class": "private",
            "labels": ["label-%d" % index],
        }
        row.update(over)
        rows.append(row)
    return rows


def make_candidates(with_sessions: bool = False, with_profiles: bool = True,
                    profile_ids=None, with_topics: bool = True):
    sessions = (
        [
            {
                "session_id": "sess-a",
                "project_id_hash": PROJECT_HASH,
                "context_token_cost": 400,
                "idle_days": 1,
                "contamination_flags": [],
            }
        ]
        if with_sessions
        else []
    )
    document: Dict[str, Any] = {
        "schema": "ordapilot.candidate_set/1",
        "request_id": REQUEST_ID,
        "topics": (
            [
                {"topic_id": "topic-auth", "label_state": "promoted", "label_age_days": 1, "evidence_count": 2},
                {"topic_id": "topic-billing", "label_state": "promoted", "label_age_days": 2, "evidence_count": 1},
            ]
            if with_topics
            else []
        ),
        "topic_sentinels": ["novel", "ambiguous"],
        "sessions": sessions,
        "session_sentinels": ["new_session"],
    }
    if with_profiles:
        document["profiles"] = profile_block(profile_ids)
        document["profile_sentinels"] = list(PROFILE_SENTINELS)
    return CandidateSet.from_dict(document)


HANDLE = RequestHandle(
    request_id=REQUEST_ID,
    prompt_hash=sha256_hex(PROMPT_TEXT),
    project_id_hash=PROJECT_HASH,
    config_rev=CONFIG_REV,
    context_hash=None,
)


class MockTransport:
    """Records requests; never opens a socket (D5)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: List[TransportRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("transport was called more times than the test allows")
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return cast(TransportResponse, item(request))
        return cast(TransportResponse, item)


def content_body(content: str, status: int = 200) -> TransportResponse:
    return TransportResponse(
        status=status,
        headers={"content-type": "application/json"},
        body=json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"),
    )


def scores_body(scores) -> TransportResponse:
    return content_body(json.dumps({"scores": scores}))


def good_scores(ids=None):
    return [{"candidate": candidate, "raw": 0.50} for candidate in (ids if ids is not None else PROFILE_BALLOT)]


def provider_row(**over):
    row = {
        "id": "jev",
        "kind": "openai_compatible",
        "base_url": "https://provider.example.com/v1",
        "model": "jev-classifier-1",
        "api_key_env": KEY_ENV,
    }
    row.update(over)
    return row


def providers_config(rows=None, **over):
    config = {"schema_version": 1, "providers": [provider_row()] if rows is None else list(rows)}
    config.update(over)
    return config


def make_backend(transport, **row_over) -> OpenAICompatibleBackend:
    row = parse_provider_row(provider_row(**row_over), 0)
    return OpenAICompatibleBackend(row, api_key=KEY_VALUE, transport=transport)


def payload_json(transport: MockTransport, index: int = 0) -> Dict[str, Any]:
    return json.loads(transport.requests[index].body.decode("utf-8"))


def user_message(transport: MockTransport, index: int = 0) -> Dict[str, Any]:
    return payload_json(transport, index)["messages"][1]


class KeyEnvTestCase(unittest.TestCase):
    """Sets/clears the placeholder env key without leaking it between tests."""

    def set_key(self, name: str = KEY_ENV, value: str = KEY_VALUE) -> None:
        previous = os.environ.get(name)
        os.environ[name] = value
        self.addCleanup(self._restore, name, previous)

    def clear_key(self, name: str = KEY_ENV) -> None:
        previous = os.environ.get(name)
        os.environ.pop(name, None)
        self.addCleanup(self._restore, name, previous)

    @staticmethod
    def _restore(name: str, previous) -> None:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def fixture_backend(table) -> SyntheticBackend:
    """Deterministic offline backend with an explicit score table (no network)."""
    return SyntheticBackend(fixture_table=table)


IDENTITY_CALIBRATION = CalibrationModel(model_id="platt-v1", active=True)


class FullFixtureTestCase(unittest.TestCase):
    """Shared fixture tables that make every surface's outcome deterministic."""

    PROMPT_HASH = sha256_hex(PROMPT_TEXT)

    def table(self, topic_top: float = 0.99, sentinel: float = 0.99, alpha: float = 0.30,
              beta: float = 0.20, session_real: float = 0.99):
        return {
            (self.PROMPT_HASH, "topic"): {
                "topic-auth": topic_top,
                "topic-billing": 0.10,
                "novel": 0.05,
                "ambiguous": 0.05,
            },
            (self.PROMPT_HASH, "session"): {"sess-a": session_real, "new_session": 0.10},
            (self.PROMPT_HASH, "profile"): {
                "profile-alpha": alpha,
                "profile-beta": beta,
                "no_suitable_profile": sentinel,
            },
        }

    def config(self, **table_over):
        return RouterConfig(
            backend=fixture_backend(self.table(**table_over)),
            calibration=IDENTITY_CALIBRATION,
        )


# --------------------------------------------------------------------------
# A. The closed surface domain
# --------------------------------------------------------------------------


class TestSurfaceDomain(unittest.TestCase):
    def test_closed_surface_domain_is_topic_session_profile(self):
        self.assertEqual(tuple(SURFACES), ("topic", "session", "profile"))

    def test_batch_calls_is_seeded_for_every_surface(self):
        backend = SyntheticBackend()
        self.assertEqual(backend.batch_calls, {"topic": 0, "session": 0, "profile": 0})
        self.assertEqual(CandidateSet.candidate_ids_for.__doc__ is not None, True)

    def test_unknown_surface_is_refused_by_the_local_adapter(self):
        backend = SyntheticBackend()
        for surface in ("modality", "Profile", "profile ", "", "topic2"):
            with self.assertRaises(BackendError) as ctx:
                backend.batch_score(HANDLE, make_candidates(), surface)
            self.assertIn("unknown surface", str(ctx.exception))

    def test_unknown_surface_is_refused_by_the_provider_adapter(self):
        backend = make_backend(MockTransport([]))
        for surface in ("modality", "Profile", "profiles", " topic"):
            with self.assertRaises(ProviderContractError) as ctx:
                backend.batch_score(HANDLE, make_candidates(), surface)
            self.assertEqual(ctx.exception.code, "unknown_surface")
        self.assertEqual(backend.transport_calls, 0, "an unknown surface must not reach the wire")

    def test_openrouter_never_scores_the_profile_surface(self):
        backend = OpenRouterBackend(enabled=True)
        with self.assertRaises(BackendError):
            backend.batch_score(HANDLE, make_candidates(), "profile")


# --------------------------------------------------------------------------
# B. Happy path: ONE call, all eligible profiles, exact row shape, caller order
# --------------------------------------------------------------------------


class TestProfileSurfaceHappyPath(KeyEnvTestCase):
    def test_one_post_scores_the_whole_ballot_in_the_caller_order(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        rows = backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(transport.call_count, 1, "ONE call per surface")
        self.assertEqual(backend.batch_calls, {"topic": 0, "session": 0, "profile": 1})
        self.assertEqual(transport.requests[0].url, "https://provider.example.com/v1/chat/completions")
        self.assertEqual([row["candidate"] for row in rows], PROFILE_BALLOT)
        for row in rows:
            self.assertEqual(set(row.keys()), {"candidate", "raw"})
            self.assertIsInstance(row["raw"], float)
            self.assertTrue(0.0 <= row["raw"] <= 1.0)

    def test_payload_carries_the_exact_ballot_and_nothing_more(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(), "profile")
        message = user_message(transport)
        self.assertEqual(message["surface"], "profile")
        self.assertEqual(message["candidates"], PROFILE_BALLOT)
        block = message["profile_candidates"]
        self.assertEqual([row["profile_id"] for row in block], PROFILE_IDS)
        for row in block:
            self.assertEqual(set(row.keys()), set(PROFILE_REQUEST_FIELDS))

    def test_sentinel_is_always_in_the_profile_candidate_set(self):
        candidates = make_candidates()
        self.assertEqual(candidates.profile_candidate_ids(), PROFILE_BALLOT)
        self.assertIn("no_suitable_profile", candidates.profile_candidate_ids())
        # declared explicitly or not, the parsed set always carries the sentinel
        self.assertEqual(tuple(candidates.profile_sentinels), PROFILE_SENTINELS)

    def test_reordered_reply_is_a_hard_error_on_the_profile_surface(self):
        shuffled = list(reversed(good_scores()))
        transport = MockTransport([scores_body(shuffled)])
        backend = make_backend(transport)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, "candidate_order_mismatch")

    def test_caller_declared_order_is_preserved_end_to_end(self):
        transport = MockTransport([scores_body(good_scores(["profile-beta", "profile-alpha", "no_suitable_profile"]))])
        backend = make_backend(transport)
        candidates = make_candidates(profile_ids=["profile-beta", "profile-alpha"])
        rows = backend.batch_score(HANDLE, candidates, "profile")
        self.assertEqual([row["candidate"] for row in rows], ["profile-beta", "profile-alpha", "no_suitable_profile"])
        self.assertEqual(
            [row["profile_id"] for row in user_message(transport)["profile_candidates"]],
            ["profile-beta", "profile-alpha"],
        )

    def test_profile_batch_is_isolated_from_the_other_surfaces(self):
        transport = MockTransport(
            [
                scores_body(good_scores(TOPIC_IDS)),
                scores_body(good_scores(SESSION_IDS)),
                scores_body(good_scores()),
            ]
        )
        backend = make_backend(transport)
        candidates = make_candidates(with_sessions=True)
        backend.batch_score(HANDLE, candidates, "topic")
        backend.batch_score(HANDLE, candidates, "session")
        backend.batch_score(HANDLE, candidates, "profile")
        bodies = [user_message(transport, i) for i in range(3)]
        self.assertEqual(bodies[0]["candidates"], TOPIC_IDS)
        self.assertEqual(bodies[1]["candidates"], SESSION_IDS)
        self.assertEqual(bodies[2]["candidates"], PROFILE_BALLOT)
        # a profile id may not appear in a topic/session ballot and vice versa
        for banned in PROFILE_BALLOT:
            self.assertNotIn(banned, bodies[0]["candidates"])
            self.assertNotIn(banned, bodies[1]["candidates"])
        for banned in TOPIC_IDS + SESSION_IDS:
            self.assertNotIn(banned, bodies[2]["candidates"])
        # only the profile surface carries a profile metadata block
        self.assertNotIn("profile_candidates", bodies[0])
        self.assertNotIn("profile_candidates", bodies[1])
        self.assertEqual(backend.batch_calls, {"topic": 1, "session": 1, "profile": 1})

    def test_profile_candidate_carries_no_metadata_beyond_the_whitelist(self):
        candidates = make_candidates()
        for profile in candidates.profiles:
            self.assertEqual(
                set(profile.to_dict().keys()), {"profile_id", "scope", "privacy_class", "labels"}
            )


# --------------------------------------------------------------------------
# C. Candidate-set preservation: no widening, no reorder, no add, no drop
# --------------------------------------------------------------------------


class TestCandidateSetPreservation(KeyEnvTestCase):
    def assert_contract_reject(self, reply_items, code: str) -> ProviderContractError:
        backend = make_backend(MockTransport(list(reply_items)), max_retries=0)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, code, str(ctx.exception))
        return ctx.exception

    def test_missing_candidate_is_a_hard_error(self):
        rows = good_scores(["profile-alpha", "no_suitable_profile"])
        self.assert_contract_reject([scores_body(rows)], "candidate_set_mismatch")

    def test_extra_candidate_is_a_hard_error(self):
        rows = good_scores() + [{"candidate": "profile-gamma", "raw": 0.5}]
        self.assert_contract_reject([scores_body(rows)], "candidate_set_mismatch")

    def test_unknown_candidate_id_is_a_hard_error(self):
        rows = good_scores(["profile-alpha", "profile-beta", "topic-auth"])
        self.assert_contract_reject([scores_body(rows)], "candidate_set_mismatch")

    def test_duplicate_candidate_id_is_a_hard_error(self):
        rows = good_scores() + [{"candidate": "profile-alpha", "raw": 0.5}]
        self.assert_contract_reject([scores_body(rows)], "duplicate_candidate_id")

    def test_dropped_sentinel_is_a_hard_error(self):
        rows = good_scores(PROFILE_IDS)
        self.assert_contract_reject([scores_body(rows)], "candidate_set_mismatch")

    def test_extra_score_row_field_is_a_hard_error(self):
        rows = [{"candidate": "profile-alpha", "raw": 0.5, "why": "gut"}, {"candidate": "profile-beta", "raw": 0.5},
                {"candidate": "no_suitable_profile", "raw": 0.5}]
        self.assert_contract_reject([scores_body(rows)], "score_row_unknown_field")

    def test_metadata_that_does_not_match_the_ballot_is_refused_before_the_wire(self):
        transport = MockTransport([])
        backend = make_backend(transport)
        candidates = make_candidates()
        with self.assertRaises(ProviderContractError) as ctx:
            backend.build_batch_payload(
                HANDLE, ["profile-alpha", "no_suitable_profile"], "profile",
                [ProfileCandidate(profile_id="profile-delta")],
            )
        self.assertEqual(ctx.exception.code, "candidate_set_mismatch")
        self.assertEqual(transport.call_count, 0)

    def test_candidate_ids_come_from_one_surface_only(self):
        candidates = make_candidates(with_sessions=True)
        self.assertEqual(candidates.candidate_ids_for("topic"), TOPIC_IDS)
        self.assertEqual(candidates.candidate_ids_for("session"), SESSION_IDS)
        self.assertEqual(candidates.candidate_ids_for("profile"), PROFILE_BALLOT)
        with self.assertRaises(ValueError):
            candidates.candidate_ids_for("modality")

    def test_duplicate_profile_id_in_the_candidate_set_is_rejected(self):
        document = {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": [],
            "session_sentinels": ["new_session"],
            "profiles": profile_block(["profile-alpha", "profile-alpha"]),
        }
        with self.assertRaises(SchemaError):
            CandidateSet.from_dict(document)

    def test_sentinel_id_may_not_be_declared_as_a_profile(self):
        document = {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": [],
            "session_sentinels": ["new_session"],
            "profiles": profile_block(["no_suitable_profile"]),
        }
        with self.assertRaises(SchemaError):
            CandidateSet.from_dict(document)

    def test_unknown_candidate_set_field_still_fails_hard(self):
        document = {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": [],
            "session_sentinels": ["new_session"],
            "profile_allowlist": ["profile-alpha"],
        }
        with self.assertRaises(SchemaError):
            CandidateSet.from_dict(document)

    def test_unknown_profile_field_is_rejected_by_the_closed_schema(self):
        for extra in ({"display_name": "Ada"}, {"privacy_note": "sensitive"}, {"eligibility": "all"}):
            document = {
                "schema": "ordapilot.candidate_set/1",
                "request_id": REQUEST_ID,
                "topics": [],
                "topic_sentinels": ["novel", "ambiguous"],
                "sessions": [],
                "session_sentinels": ["new_session"],
                "profiles": [dict(profile_block(["profile-alpha"])[0], **extra)],
            }
            with self.assertRaises(SchemaError):
                CandidateSet.from_dict(document)

    def test_free_text_label_is_rejected_by_the_closed_schema(self):
        document = {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": [],
            "session_sentinels": ["new_session"],
            "profiles": [dict(profile_block(["profile-alpha"])[0], labels=[PROFILE_SECRET_MARKER])],
        }
        with self.assertRaises(SchemaError):
            CandidateSet.from_dict(document)


# --------------------------------------------------------------------------
# D. Abstention is reachable
# --------------------------------------------------------------------------


class TestAbstention(FullFixtureTestCase):
    def test_explicit_abstain_winner_forces_the_abstain_band(self):
        result = route(
            make_request(), make_candidates(with_sessions=True), self.config(sentinel=0.99),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.profile_band, "abstain_or_new_session")
        self.assertEqual(result.profile_escalation_code, "profile_abstain_argmax")
        self.assertEqual(result.decision.band, "abstain_or_new_session")
        self.assertEqual(result.decision.decision, "new_session")
        self.assertEqual(result.profile_scores[-1]["candidate"], "no_suitable_profile")

    def test_profile_abstain_overrides_a_session_reuse(self):
        # control: without the profile surface the very same fixture reuses the session
        control = route(
            make_request(), make_candidates(with_sessions=True, with_profiles=False), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(control.session_band, "automatic")
        self.assertEqual(control.decision.decision, "session_reuse")

        profiled = route(
            make_request(), make_candidates(with_sessions=True), self.config(sentinel=0.99),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(profiled.decision.band, "abstain_or_new_session")
        self.assertEqual(profiled.decision.decision, "new_session")
        self.assertNotEqual(profiled.decision.decision, "session_reuse")

    def test_margin_gate_is_not_met_so_nothing_becomes_automatic(self):
        result = route(
            make_request(), make_candidates(),
            self.config(sentinel=0.10, alpha=0.88, beta=0.86),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.profile_band, "fallback_escalate")
        self.assertEqual(result.profile_escalation_code, "below_margin")
        self.assertNotEqual(result.decision.band, "automatic")

    def test_low_profile_scores_abstain(self):
        result = route(
            make_request(), make_candidates(),
            self.config(sentinel=0.20, alpha=0.30, beta=0.10),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.profile_band, "abstain_or_new_session")
        self.assertIn(result.profile_escalation_code, ("below_min_score", "calibration_missing"))

    def test_backend_failure_on_the_profile_surface_abstains_and_emits_no_scores(self):
        backend = make_backend(
            MockTransport([
                ProviderError("backend_error", "boom"),
                ProviderError("backend_error", "boom"),
            ]), max_retries=0
        )
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.profile_band, "abstain_or_new_session")
        self.assertEqual(result.profile_escalation_code, "backend_error")
        self.assertEqual(result.decision.band, "abstain_or_new_session")
        self.assertNotIn("profile_scores", result.receipt)
        validate_receipt(result.receipt)

    def test_abstain_code_is_in_the_closed_escalation_domain(self):
        self.assertIn("profile_abstain_argmax", ESCALATION_CODES)


# --------------------------------------------------------------------------
# E. Calibration gate
# --------------------------------------------------------------------------


class TestCalibrationGate(FullFixtureTestCase):
    def test_automatic_is_unreachable_without_calibration(self):
        # raw 0.99 on every surface, sentinel last: still nothing automatic
        backend = fixture_backend(self.table(sentinel=0.01, alpha=0.99, beta=0.01))
        result = route(
            make_request(), make_candidates(with_sessions=True),
            RouterConfig(backend=backend),  # default calibration: model_id "none", inactive
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.receipt["calibration_model_id"], "none")
        self.assertEqual(result.profile_band, "fallback_escalate")
        self.assertEqual(result.profile_escalation_code, "calibration_missing")
        self.assertEqual(result.decision.band, "abstain_or_new_session")
        self.assertNotEqual(result.decision.band, "automatic")

    def test_raw_profile_scores_reach_the_receipt_unchanged(self):
        result = route(
            make_request(), make_candidates(), self.config(sentinel=0.05, alpha=0.40, beta=0.35),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        by_id = {row["candidate"]: row for row in result.receipt["profile_scores"]}
        self.assertEqual(by_id["profile-alpha"]["raw"], 0.40)
        # identity calibration (active, a=1, b=0) leaves the ranking signal alone
        self.assertAlmostEqual(by_id["profile-alpha"]["calibrated"], 0.40, places=9)
        self.assertEqual(by_id["no_suitable_profile"]["raw"], 0.05)

    def test_profile_surface_reuses_the_generic_thresholds_only(self):
        config = RouterConfig()
        result = route(
            make_request(), make_candidates(), self.config(sentinel=0.05, alpha=0.95, beta=0.10),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        # the threshold block is the same closed set for every surface: no tau_profile
        self.assertEqual(set(result.receipt["thresholds_used"].keys()), set(config.thresholds.to_dict().keys()))
        self.assertEqual(result.profile_band, "automatic")  # permitted, gated by calibration
        self.assertEqual(result.decision.band, "automatic")

    def test_uncalibrated_surface_cannot_promote_an_automatic_band(self):
        backend = fixture_backend(self.table(topic_top=0.10, sentinel=0.01, alpha=0.99))
        result = route(
            make_request(), make_candidates(),
            RouterConfig(backend=backend, calibration=CalibrationModel(model_id="none", active=False)),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertNotEqual(result.decision.band, "automatic")
        self.assertIn(result.profile_escalation_code, ("calibration_missing", "below_min_score"))
        self.assertFalse(CalibrationModel(model_id="none").active)


# --------------------------------------------------------------------------
# F. Privacy choke point
# --------------------------------------------------------------------------


class TestPrivacyChokePoint(KeyEnvTestCase):
    def test_no_raw_prompt_text_reaches_the_profile_payload(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(), "profile")
        body = transport.requests[0].body.decode("utf-8")
        self.assertIn(sha256_hex(PROMPT_TEXT), body)
        for needle in (PROMPT_TEXT, "REFUND", "invoice", "urgent"):
            self.assertNotIn(needle, body)

    def test_no_non_whitelisted_profile_metadata_reaches_the_payload(self):
        candidates = make_candidates()
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, candidates, "profile")
        message = user_message(transport)
        self.assertEqual(set(message["profile_candidates"][0].keys()), set(PROFILE_REQUEST_FIELDS))
        body = transport.requests[0].body.decode("utf-8")
        for banned in ("display_name", "description", "eligibility", "email", "owner", PROFILE_SECRET_MARKER):
            self.assertNotIn(banned, body)

    def test_a_hostile_profile_object_cannot_smuggle_an_extra_field(self):
        class HostileProfile:
            def to_dict(self):
                return {
                    "profile_id": "profile-alpha",
                    "scope": "project",
                    "privacy_class": "private",
                    "labels": [],
                    "description": PROFILE_SECRET_MARKER,
                }

        transport = MockTransport([])
        backend = make_backend(transport)
        with self.assertRaises(ProfileMetadataViolation) as ctx:
            backend.build_batch_payload(HANDLE, ["profile-alpha", "no_suitable_profile"], "profile", [HostileProfile()])
        self.assertIn("non-whitelisted", str(ctx.exception))
        self.assertEqual(transport.call_count, 0)

    def test_a_hostile_mapping_cannot_smuggle_an_extra_field(self):
        backend = make_backend(MockTransport([]))
        with self.assertRaises(ProfileMetadataViolation):
            backend.build_batch_payload(
                HANDLE, ["profile-alpha", "no_suitable_profile"], "profile",
                [{"profile_id": "profile-alpha", "scope": "project", "privacy_class": "private",
                  "labels": [], "privacy_note": PROFILE_SECRET_MARKER}],
            )

    def test_out_of_domain_scope_or_privacy_class_is_refused(self):
        backend = make_backend(MockTransport([]))
        for field, value in (("scope", "everywhere"), ("privacy_class", "topsecret")):
            row = {"profile_id": "profile-alpha", "scope": "project", "privacy_class": "private", "labels": []}
            row[field] = value
            with self.assertRaises(ProfileMetadataViolation):
                backend.build_batch_payload(
                    HANDLE, ["profile-alpha", "no_suitable_profile"], "profile", [row]
                )

    def test_free_text_label_is_refused_at_the_choke_point(self):
        backend = make_backend(MockTransport([]))
        with self.assertRaises(ProfileMetadataViolation):
            backend.build_batch_payload(
                HANDLE, ["profile-alpha", "no_suitable_profile"], "profile",
                [{"profile_id": "profile-alpha", "scope": "project", "privacy_class": "private",
                  "labels": [PROFILE_SECRET_MARKER]}],
            )

    def test_payload_assertion_rejects_a_smuggled_profile_key_anywhere(self):
        payload = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "ordaprompt-batch-score-v1"},
                {"role": "user", "content": "batch-score", "profile_display_name": "Ada"},
            ],
        }
        with self.assertRaises(ProfileMetadataViolation):
            assert_profile_payload_whitelisted(payload)

    def test_payload_assertion_rejects_an_incomplete_profile_row(self):
        payload = {
            "profile_candidates": [{"profile_id": "profile-alpha", "scope": "project"}],
        }
        with self.assertRaises(ProfileMetadataViolation):
            assert_profile_payload_whitelisted(payload)

    def test_payload_assertion_accepts_the_real_whitelisted_shape(self):
        backend = make_backend(MockTransport([scores_body(good_scores())]))
        backend.batch_score(HANDLE, make_candidates(), "profile")
        assert_profile_payload_whitelisted(backend.last_payload)

    def test_topic_and_session_payloads_carry_no_profile_metadata_at_all(self):
        transport = MockTransport([scores_body(good_scores(TOPIC_IDS))])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(with_sessions=True), "topic")
        body = transport.requests[0].body.decode("utf-8")
        self.assertNotIn("profile", body)

    def test_diagnostics_never_leak_key_or_profile_text(self):
        backend = make_backend(MockTransport([scores_body(good_scores())]))
        backend.batch_score(HANDLE, make_candidates(), "profile")
        text = repr(backend) + json.dumps(backend.last_payload, sort_keys=True)
        self.assertNotIn(KEY_VALUE, text)
        self.assertNotIn(KEY_VALUE[:6], text)
        self.assertNotIn(PROFILE_SECRET_MARKER, text)


# --------------------------------------------------------------------------
# G. Fail closed (D3) on the profile surface
# --------------------------------------------------------------------------


class TestProfileFailClosed(KeyEnvTestCase):
    def test_unset_api_key_env_disables_the_provider(self):
        self.clear_key()
        registry = ProviderRegistry.from_dict(providers_config())
        capability = registry.capabilities()[0]
        self.assertFalse(capability["api_key_present"])
        self.assertEqual(capability["disabled_reason"], "api_key_env_unset")
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(MockTransport([]))
        self.assertEqual(ctx.exception.code, "api_key_env_unset")

    def test_missing_key_never_falls_back_and_opens_no_socket(self):
        self.clear_key()
        rows = [provider_row(), provider_row(id="backup", base_url="https://backup.example.com/v1")]
        registry = ProviderRegistry.from_dict(
            providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        )
        transport = MockTransport([])
        with self.assertRaises(ProviderConfigError):
            registry.select_backend(transport)
        self.assertEqual(transport.call_count, 0)

    def test_malformed_replies_fail_closed(self):
        cases = [
            (TransportResponse(200, {}, b"<html>nope</html>"), "reply_not_json"),
            (TransportResponse(200, {}, b"[1,2,3]"), "reply_not_object"),
            (TransportResponse(200, {}, b'{"id": "x"}'), "reply_choices_invalid"),
            (TransportResponse(200, {}, b'{"choices": []}'), "reply_choices_invalid"),
            (content_body("plain prose, not json"), "reply_content_not_json"),
            (content_body('{"label": "profile-alpha"}'), "reply_scores_missing"),
            (content_body('{"scores": {"profile-alpha": 0.5}}'), "reply_scores_not_list"),
            (content_body('{"scores": ["profile-alpha"]}'), "score_row_not_object"),
        ]
        for reply, code in cases:
            backend = make_backend(MockTransport([reply]), max_retries=0)
            with self.assertRaises(ProviderContractError) as ctx:
                backend.batch_score(HANDLE, make_candidates(), "profile")
            self.assertEqual(ctx.exception.code, code, (code, str(ctx.exception)))

    def test_bad_score_values_fail_closed(self):
        def rows_with(candidate: str, raw: Any):
            rows = good_scores()
            rows = [r for r in rows if r["candidate"] != candidate]
            rows.append({"candidate": candidate, "raw": raw})
            return rows

        for candidate, raw, code in (
            ("profile-alpha", "0.9", "score_not_numeric"),
            ("profile-alpha", True, "score_not_numeric"),
            ("profile-alpha", float("nan"), "score_not_finite"),
            ("profile-alpha", float("inf"), "score_not_finite"),
            ("profile-alpha", -0.0001, "score_out_of_range"),
            ("profile-alpha", 1.0001, "score_out_of_range"),
        ):
            backend = make_backend(MockTransport([scores_body(rows_with(candidate, raw))]), max_retries=0)
            with self.assertRaises(ProviderContractError) as ctx:
                backend.batch_score(HANDLE, make_candidates(), "profile")
            self.assertEqual(ctx.exception.code, code, (code, str(ctx.exception)))

        invalid_rows = good_scores()
        invalid_rows[-1] = {"candidate": "profile alpha", "raw": 0.5}
        backend = make_backend(MockTransport([scores_body(invalid_rows)]), max_retries=0)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, "score_candidate_invalid")

    def test_nan_token_in_the_body_is_rejected(self):
        rows = [
            {"candidate": "profile-alpha", "raw": 0.5},
            {"candidate": "profile-beta", "raw": float("nan")},
            {"candidate": "no_suitable_profile", "raw": 0.5},
        ]
        body = json.dumps({"scores": rows})
        self.assertIn("NaN", body)
        backend = make_backend(MockTransport([content_body(body)]), max_retries=0)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, "score_not_finite")

    def test_too_many_profiles_is_refused_before_the_wire(self):
        transport = MockTransport([])
        backend = make_backend(transport, batch_max_candidates=2)
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, "batch_too_large")
        self.assertEqual(transport.call_count, 0)

    def test_contract_violation_is_not_retried(self):
        transport = MockTransport([content_body("not json at all")])
        backend = make_backend(transport, max_retries=3)
        with self.assertRaises(ProviderContractError):
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(transport.call_count, 1)

    def test_refused_host_never_reaches_the_transport_for_the_profile_surface(self):
        row = ProviderRow(
            id="evil", kind="openai_compatible",
            base_url="http://metadata.example.internal/v1", model="m", api_key_env=KEY_ENV,
        )
        transport = MockTransport([])
        backend = OpenAICompatibleBackend(row, api_key=KEY_VALUE, transport=transport)
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "profile")
        self.assertEqual(ctx.exception.code, "url_scheme_refused")
        self.assertEqual(transport.call_count, 0)

    def test_validate_batch_reply_is_reusable_for_profiles(self):
        document = {"choices": [{"message": {"content": json.dumps({"scores": good_scores()})}}]}
        rows = validate_batch_reply(document, PROFILE_BALLOT, provider_id="jev")
        self.assertEqual([row["candidate"] for row in rows], PROFILE_BALLOT)
        self.assertEqual(len(rows), len(PROFILE_BALLOT))


# --------------------------------------------------------------------------
# H. Router wiring, receipt shape and round-trip
# --------------------------------------------------------------------------


class TestProfileRouterIntegration(FullFixtureTestCase):
    def test_receipt_omits_the_profile_block_when_no_profiles_were_declared(self):
        result = route(
            make_request(), make_candidates(with_profiles=False), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        for field in ("profile_candidate_ids", "profile_scores", "profile_top", "profile_margin"):
            self.assertNotIn(field, result.receipt)
        self.assertIsNone(result.profile_band)
        self.assertEqual(result.decision.profile_candidate_ids, [])
        validate_receipt(result.receipt)

    def test_profile_block_is_emitted_and_valid_when_the_surface_ran(self):
        result = route(
            make_request(), make_candidates(), self.config(sentinel=0.05, alpha=0.40, beta=0.30),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.receipt["profile_candidate_ids"], PROFILE_BALLOT)
        self.assertEqual(
            [row["candidate"] for row in result.receipt["profile_scores"]], PROFILE_BALLOT
        )
        self.assertEqual(result.receipt["profile_top"]["candidate"], "profile-alpha")
        self.assertAlmostEqual(result.receipt["profile_margin"], 0.10, places=9)
        self.assertEqual(set(result.receipt.keys()) - set(RECEIPT_FIELDS), set())
        validate_receipt(result.receipt)
        assert_no_free_text(result.receipt)

    def test_profile_receipt_round_trips_through_the_store(self):
        result = route(
            make_request(), make_candidates(), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(tmp)
            store.append(result.receipt)
            stored = store.read_all()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["profile_candidate_ids"], PROFILE_BALLOT)
            validate_receipt(stored[0])
            assert_no_free_text(stored[0])

    def test_sealed_receipt_domain_still_rejects_an_invented_profile_field(self):
        result = route(
            make_request(), make_candidates(), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        tampered = dict(result.receipt)
        tampered["profile_scores"][0]["note"] = "free text"
        with self.assertRaises(SchemaError):
            validate_receipt(tampered)

    def test_decision_document_round_trips_with_the_profile_block(self):
        result = route(
            make_request(), make_candidates(), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        document = result.decision.to_dict()
        self.assertIn("profile_scores", document)
        parsed = RoutingDecision.from_dict(document)
        self.assertEqual(parsed.profile_candidate_ids, PROFILE_BALLOT)
        self.assertEqual(parsed.profile_margin, result.decision.profile_margin)

    def test_profile_score_rows_have_exactly_two_keys(self):
        result = route(
            make_request(), make_candidates(), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        for row in result.decision.profile_scores:
            self.assertEqual(set(row.keys()), {"candidate", "raw", "calibrated"})

    def test_profile_surface_can_be_omitted_by_omitting_profile_candidates(self):
        result = route(
            make_request(), make_candidates(with_profiles=False), self.config(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertIsNone(result.profile_band)
        self.assertNotIn("profile_scores", result.receipt)

    def test_cli_local_profile_run_is_offline_and_records_the_surface(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "classify",
                    "--request", str(ROOT / "eval" / "demo" / "request.json"),
                    "--topics", "topic-auth,topic-billing",
                    "--profiles", "profile-alpha,profile-beta",
                    "--receipts-dir", str(Path(tmp) / "receipts"),
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(proc.stdout)["routing_receipt"]
            self.assertEqual(receipt["profile_candidate_ids"], PROFILE_BALLOT)
            self.assertEqual(receipt["backend_used"], "synthetic")
            for field in ("profile_scores", "profile_top", "profile_margin"):
                self.assertIn(field, receipt)
            validate_receipt(receipt)
            assert_no_free_text(receipt)

    def test_cli_rejects_a_profile_id_outside_the_closed_domain(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable, "-m", "ordaprompt_router.cli", "classify",
                "--request", str(ROOT / "eval" / "demo" / "request.json"),
                "--profiles", "Profile Alpha",
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("schema_reject", proc.stderr)


# --------------------------------------------------------------------------
# I. providers.json schema: no profile-related row field is required (item 8)
# --------------------------------------------------------------------------


class TestProviderRowSchemaUnchanged(unittest.TestCase):
    """Item 8 decision: NO providers.json row change is needed.

    Eligibility and privacy class are properties of the CANDIDATES, which are
    computed upstream and travel in the candidate set; they are not properties of the
    transport row.  A row therefore gains nothing, and any profile-ish row field stays
    a hard `unknown_field` error rather than a new, silently-defaulted knob.
    """

    def test_profile_related_row_fields_are_still_unknown_fields(self):
        for field in ("profile_allowlist", "default_profile", "profile_privacy_class", "profiles"):
            row = provider_row()
            row[field] = "profile-alpha"
            with self.assertRaises(ProviderConfigError) as ctx:
                ProviderRegistry.from_dict(providers_config(rows=[row]))
            self.assertEqual(ctx.exception.code, "unknown_field")

    def test_row_field_set_is_unchanged_by_the_profile_surface(self):
        from ordaprompt_router.providers import ROW_FIELDS

        self.assertEqual(
            tuple(ROW_FIELDS),
            (
                "id", "kind", "base_url", "model", "api_key_env", "allow_private_network",
                "timeout_s", "max_retries", "max_tokens", "batch_max_candidates",
                "supports_taxonomy_proposal",
            ),
        )


if __name__ == "__main__":
    unittest.main()
