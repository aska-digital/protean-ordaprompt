"""Offline tests for the provider layer (locked decisions D1-D7).

Every test in this file uses a MOCKED transport: no socket is ever opened, no
third-party service is contacted, and no real credential is used.  The environment
key used here is a literal non-secret placeholder value.

Run: python3 -m unittest discover -s test -p 'test_*.py' -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ordaprompt_router import (  # noqa: E402
    BackendError,
    CandidateSet,
    ClassificationRequest,
    OpenAICompatibleBackend,
    ProviderChainBackend,
    ProviderConfigError,
    ProviderContractError,
    ProviderRegistry,
    ProviderTransportError,
    RequestHandle,
    RouterConfig,
    SyntheticBackend,
    TransportRequest,
    TransportResponse,
    route,
    sha256_hex,
)
from ordaprompt_router.providers import (  # noqa: E402
    NoRedirectHandler,
    is_loopback_or_private_host,
    parse_provider_row,
    stdlib_transport,
    validate_batch_reply,
)
from ordaprompt_router.receipts import ReceiptStore, assert_no_free_text  # noqa: E402
from ordaprompt_router.router import CalibrationModel  # noqa: E402
from ordaprompt_router.schemas import RECEIPT_FIELDS, validate_receipt  # noqa: E402

PROMPT_TEXT = "REFUND my invoice for the billing cycle, this is urgent"
REQUEST_ID = "11111111-1111-4111-8111-111111111111"
CONFIG_REV = "1.0.0"
PROJECT_HASH = "sha256:" + "cd" * 32
KEY_ENV = "ORDAPROMPT_TEST_PROVIDER_KEY"
KEY_VALUE = "sk-unit-test-placeholder-value-0000"
TOPIC_IDS = ["topic-auth", "topic-billing", "novel", "ambiguous"]
SESSION_IDS = ["sess-a", "new_session"]

#: the receipt field set of release 1.0.0 (24 fields, D2: unchanged by default)
BASELINE_RECEIPT_FIELDS = {
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
}


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


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


def make_candidates(with_sessions: bool = False) -> CandidateSet:
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
    return CandidateSet.from_dict(
        {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [
                {"topic_id": "topic-auth", "label_state": "promoted", "label_age_days": 1, "evidence_count": 2},
                {"topic_id": "topic-billing", "label_state": "promoted", "label_age_days": 2, "evidence_count": 1},
            ],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": sessions,
            "session_sentinels": ["new_session"],
        }
    )


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
        self.requests = []

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
    return [{"candidate": candidate, "raw": 0.50} for candidate in (ids if ids is not None else TOPIC_IDS)]


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


def auth_headers(requests) -> list:
    return [request.headers.get("Authorization") for request in requests]


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


# --------------------------------------------------------------------------
# D2 -- local-only path is unchanged when no providers file is supplied
# --------------------------------------------------------------------------


class TestLocalOnlyPathUnchanged(KeyEnvTestCase):
    def test_default_config_backend_is_the_local_deterministic_adapter(self):
        config = RouterConfig()
        backend = config.backend
        self.assertIsInstance(backend, SyntheticBackend)
        self.assertEqual(getattr(backend, "name", None), "synthetic")
        self.assertIsNone(getattr(backend, "last_fallback_from", None))

    def test_receipt_field_set_is_the_baseline_24_without_providers(self):
        result = route(make_request(), make_candidates(with_sessions=True), RouterConfig())
        self.assertEqual(set(result.receipt.keys()), BASELINE_RECEIPT_FIELDS)
        # the only additive receipt field is the optional fallback marker (D3); it is
        # allowed by the schema but never EMITTED unless a declared chain fires
        self.assertEqual(set(RECEIPT_FIELDS), BASELINE_RECEIPT_FIELDS | {"fallback_from"})
        self.assertNotIn("fallback_from", result.receipt)
        self.assertEqual(result.receipt["backend_used"], "synthetic")
        validate_receipt(result.receipt)
        assert_no_free_text(result.receipt)

    def test_default_route_is_byte_identical_across_runs(self):
        first = route(
            make_request(), make_candidates(with_sessions=True), RouterConfig(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        second = route(
            make_request(), make_candidates(with_sessions=True), RouterConfig(),
            receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(
            json.dumps(first.receipt, sort_keys=True, separators=(",", ":")),
            json.dumps(second.receipt, sort_keys=True, separators=(",", ":")),
        )

    def test_cli_without_providers_flag_is_local_only(self):
        demo = ROOT / "eval" / "demo"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "route-session",
                    "--request", str(demo / "request.json"),
                    "--candidates", str(demo / "candidates.json"),
                    "--receipts-dir", str(Path(tmp) / "receipts"),
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(proc.stdout)["routing_receipt"]
            self.assertEqual(set(receipt.keys()), BASELINE_RECEIPT_FIELDS)
            self.assertEqual(receipt["backend_used"], "synthetic")
            self.assertEqual(proc.stderr, "")

    def test_cli_exposes_the_providers_flag(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ordaprompt_router.cli", "classify", "--help"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--providers", proc.stdout)


# --------------------------------------------------------------------------
# D3 -- providers.json loader validation (fail closed)
# --------------------------------------------------------------------------


class TestProviderConfigValidation(KeyEnvTestCase):
    def test_valid_config_is_accepted(self):
        self.set_key()
        registry = ProviderRegistry.from_dict(providers_config())
        self.assertEqual(registry.order, ["jev"])
        self.assertIsNone(registry.default_provider)
        capabilities = registry.capabilities()
        self.assertEqual(len(capabilities), 1)
        self.assertEqual(capabilities[0]["id"], "jev")
        self.assertEqual(capabilities[0]["kind"], "openai_compatible")
        self.assertTrue(capabilities[0]["requires_network"])
        self.assertTrue(capabilities[0]["api_key_present"])
        self.assertTrue(capabilities[0]["enabled"])
        self.assertIsNone(capabilities[0]["disabled_reason"])

    def test_unknown_provider_id_rejected(self):
        self.set_key()
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(default_provider="missing"))
        self.assertEqual(ctx.exception.code, "unknown_provider_id")

    def test_duplicate_provider_id_rejected(self):
        self.set_key()
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(
                providers_config(rows=[provider_row(), provider_row()], default_provider="jev")
            )
        self.assertEqual(ctx.exception.code, "duplicate_provider_id")

    def test_unparsable_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text("{not json at all", encoding="utf-8")
            with self.assertRaises(ProviderConfigError) as ctx:
                ProviderRegistry.from_file(str(path))
            self.assertEqual(ctx.exception.code, "config_unparsable_json")

    def test_missing_file_rejected_without_implicit_default(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_file("/nonexistent/providers.json")
        self.assertEqual(ctx.exception.code, "config_unreadable")

    def test_unknown_config_field_rejected(self):
        self.set_key()
        config = providers_config()
        config["providers_path"] = "/etc/elsewhere.json"
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(config)
        self.assertEqual(ctx.exception.code, "unknown_field")

    def test_unknown_row_field_rejected(self):
        self.set_key()
        row = provider_row(api_key="sk-not-allowed-here")
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[row]))
        self.assertEqual(ctx.exception.code, "unknown_field")
        self.assertIn("api_key", str(ctx.exception))

    def test_schema_version_must_be_one(self):
        self.set_key()
        config = providers_config()
        config["schema_version"] = 2
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(config)
        self.assertEqual(ctx.exception.code, "schema_version_unsupported")

    def test_schema_version_must_be_an_integer(self):
        self.set_key()
        config = providers_config()
        config["schema_version"] = "1"
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(config)
        self.assertEqual(ctx.exception.code, "schema_version_invalid")

    def test_ambiguous_selection_without_default_provider(self):
        self.set_key()
        rows = [provider_row(), provider_row(id="backup", base_url="https://backup.example.com/v1")]
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=rows))
        self.assertEqual(ctx.exception.code, "provider_selection_ambiguous")

    def test_allow_fallback_requires_default_provider(self):
        self.set_key()
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(allow_fallback=True))
        self.assertEqual(ctx.exception.code, "default_provider_required")

    def test_local_row_rejects_base_url(self):
        row = {"id": "local-a", "kind": "local", "base_url": "https://example.com"}
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[row]))
        self.assertEqual(ctx.exception.code, "field_invalid")

    def test_kind_must_be_known(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(kind="anthropic")]))
        self.assertEqual(ctx.exception.code, "kind_unknown")

    def test_local_row_never_proposes_taxonomy(self):
        row = {"id": "local-a", "kind": "local", "supports_taxonomy_proposal": True}
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[row]))
        self.assertEqual(ctx.exception.code, "taxonomy_unsupported_for_kind")

    def test_provider_id_domain_is_closed(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(id="Jev Provider")]))
        self.assertEqual(ctx.exception.code, "field_invalid")

    def test_bounded_numeric_fields(self):
        with self.assertRaises(ProviderConfigError):
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(max_tokens=0)]))
        with self.assertRaises(ProviderConfigError):
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(max_retries=99)]))
        with self.assertRaises(ProviderConfigError):
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(timeout_s=True)]))
        with self.assertRaises(ProviderConfigError):
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(batch_max_candidates=10 ** 6)]))

    def test_local_row_selects_the_deterministic_adapter(self):
        registry = ProviderRegistry.from_dict(providers_config(rows=[{"id": "local-a", "kind": "local"}]))
        backend = registry.select_backend()
        self.assertIsInstance(backend, SyntheticBackend)


# --------------------------------------------------------------------------
# D1 -- credentials: names only, never values
# --------------------------------------------------------------------------


class TestCredentialHandling(KeyEnvTestCase):
    def test_missing_api_key_env_value_disables_that_provider(self):
        self.clear_key()
        registry = ProviderRegistry.from_dict(providers_config())
        capability = registry.capabilities()[0]
        self.assertFalse(capability["api_key_present"])
        self.assertFalse(capability["enabled"])
        self.assertEqual(capability["disabled_reason"], "api_key_env_unset")
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(MockTransport([]))
        self.assertEqual(ctx.exception.code, "api_key_env_unset")

    def test_missing_key_never_silently_falls_back_to_another_provider(self):
        self.clear_key()
        rows = [provider_row(), provider_row(id="backup", base_url="https://backup.example.com/v1")]
        registry = ProviderRegistry.from_dict(
            providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        )
        transport = MockTransport([])
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(transport)
        self.assertEqual(ctx.exception.code, "api_key_env_unset")
        self.assertEqual(transport.call_count, 0, "a disabled provider must not open a socket")

    def test_api_key_env_must_be_a_declared_name(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[provider_row(api_key_env="sk-literal-value")]))
        self.assertEqual(ctx.exception.code, "field_invalid")

    def test_api_key_env_field_is_required_for_network_rows(self):
        row = provider_row()
        row.pop("api_key_env")
        with self.assertRaises(ProviderConfigError) as ctx:
            ProviderRegistry.from_dict(providers_config(rows=[row]))
        self.assertEqual(ctx.exception.code, "missing_field")

    def test_capabilities_never_contains_key_material(self):
        self.set_key()
        registry = ProviderRegistry.from_dict(providers_config())
        blob = json.dumps(registry.capabilities(), sort_keys=True)
        self.assertNotIn(KEY_VALUE, blob)
        self.assertNotIn(KEY_VALUE[:6], blob)
        self.assertNotIn(str(len(KEY_VALUE)), blob)
        backend = registry.select_backend(MockTransport([scores_body(good_scores())]))
        self.assertNotIn(KEY_VALUE, repr(backend))
        self.assertNotIn(KEY_VALUE[:6], repr(backend))

    def test_no_key_material_in_error_messages(self):
        self.set_key()
        transport = MockTransport([TransportResponse(401, {}, b"unauthorized")])
        backend = make_backend(transport, max_retries=0)
        with self.assertRaises(ProviderTransportError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertIsInstance(ctx.exception, BackendError)
        message = str(ctx.exception)
        self.assertNotIn(KEY_VALUE, message)
        self.assertNotIn(KEY_VALUE[:6], message)
        self.assertNotIn("Authorization", message)
        self.assertNotIn("Bearer", message)
        # the header itself is still sent (that is the whole point of the credential)
        self.assertEqual(auth_headers(transport.requests), ["Bearer " + KEY_VALUE])

    def test_no_key_material_in_receipt_for_a_provider_backed_route(self):
        self.set_key()
        transport = MockTransport([scores_body(good_scores()), scores_body(good_scores(SESSION_IDS))])
        backend = make_backend(transport)
        result = route(
            make_request(), make_candidates(with_sessions=True),
            RouterConfig(backend=backend), receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        blob = json.dumps(result.receipt, sort_keys=True)
        self.assertNotIn(KEY_VALUE, blob)
        self.assertNotIn(KEY_VALUE[:6], blob)
        validate_receipt(result.receipt)
        assert_no_free_text(result.receipt)

    def test_no_key_material_in_stdout_or_receipts_directory(self):
        self.set_key()
        demo_request = ROOT / "eval" / "demo" / "request.json"
        with tempfile.TemporaryDirectory() as tmp:
            providers_path = Path(tmp) / "providers.json"
            # a local row: the whole CLI path runs offline with the flag supplied
            providers_path.write_text(
                json.dumps(providers_config(rows=[{"id": "local-a", "kind": "local"}])), encoding="utf-8"
            )
            receipts_dir = Path(tmp) / "receipts"
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "classify",
                    "--request", str(demo_request), "--topics", "topic-auth,topic-billing",
                    "--providers", str(providers_path), "--receipts-dir", str(receipts_dir),
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
                env=dict(os.environ, **{KEY_ENV: KEY_VALUE}),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn(KEY_VALUE, proc.stdout)
            self.assertNotIn(KEY_VALUE, proc.stderr)
            receipts_file = receipts_dir / "receipts.jsonl"
            self.assertTrue(receipts_file.is_file())
            self.assertNotIn(KEY_VALUE, receipts_file.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# D3/D6 -- strict reply contract: every malformed shape is its own case
# --------------------------------------------------------------------------


class TestBatchReplyContract(KeyEnvTestCase):
    def assert_contract_reject(self, reply_items, code: str) -> ProviderContractError:
        backend = make_backend(MockTransport(list(reply_items)), max_retries=0)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, code, str(ctx.exception))
        self.assertIsInstance(ctx.exception, BackendError)
        return ctx.exception

    def test_reply_body_not_json(self):
        self.assert_contract_reject([TransportResponse(200, {}, b"<html>nope</html>")], "reply_not_json")

    def test_reply_body_not_text(self):
        self.assert_contract_reject([TransportResponse(200, {}, cast(bytes, 12345))], "reply_not_json")

    def test_reply_not_object(self):
        self.assert_contract_reject([TransportResponse(200, {}, b"[1, 2, 3]")], "reply_not_object")

    def test_reply_choices_missing(self):
        self.assert_contract_reject([TransportResponse(200, {}, b'{"id": "x"}')], "reply_choices_invalid")

    def test_reply_choices_empty(self):
        self.assert_contract_reject([TransportResponse(200, {}, b'{"choices": []}')], "reply_choices_invalid")

    def test_reply_choices_not_a_list(self):
        self.assert_contract_reject([TransportResponse(200, {}, b'{"choices": {}}')], "reply_choices_invalid")

    def test_reply_choice_not_object(self):
        self.assert_contract_reject([TransportResponse(200, {}, b'{"choices": ["nope"]}')], "reply_choice_invalid")

    def test_reply_message_missing(self):
        self.assert_contract_reject([TransportResponse(200, {}, b'{"choices": [{}]}')], "reply_message_invalid")

    def test_reply_content_missing(self):
        self.assert_contract_reject(
            [TransportResponse(200, {}, b'{"choices": [{"message": {}}]}')], "reply_content_missing"
        )

    def test_reply_content_not_a_string(self):
        self.assert_contract_reject(
            [TransportResponse(200, {}, b'{"choices": [{"message": {"content": 42}}]}')],
            "reply_content_missing",
        )

    def test_reply_content_not_json(self):
        self.assert_contract_reject([content_body("sorry, I cannot help with that")], "reply_content_not_json")

    def test_reply_content_not_object(self):
        self.assert_contract_reject([content_body("[1, 2, 3]")], "reply_content_not_object")

    def test_reply_scores_missing(self):
        self.assert_contract_reject([content_body('{"label": "topic-auth"}')], "reply_scores_missing")

    def test_reply_unknown_field(self):
        payload = json.dumps({"scores": good_scores(), "rationale": "because"})
        self.assert_contract_reject([content_body(payload)], "reply_unknown_field")

    def test_reply_scores_not_a_list(self):
        self.assert_contract_reject([content_body('{"scores": {"topic-auth": 0.5}}')], "reply_scores_not_list")

    def test_score_row_not_object(self):
        self.assert_contract_reject([content_body('{"scores": ["topic-auth"]}')], "score_row_not_object")

    def test_score_row_unknown_field(self):
        rows = [{"candidate": "topic-auth", "raw": 0.5, "why": "gut"}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_row_unknown_field")

    def test_score_row_missing_raw(self):
        self.assert_contract_reject(
            [content_body('{"scores": [{"candidate": "topic-auth"}]}')], "score_not_numeric"
        )

    def test_score_candidate_outside_the_id_domain(self):
        rows = [{"candidate": "topic auth", "raw": 0.5}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_candidate_invalid")

    def test_score_non_numeric(self):
        rows = [{"candidate": "topic-auth", "raw": "0.9"}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_not_numeric")

    def test_score_bool_as_number(self):
        rows = [{"candidate": "topic-auth", "raw": True}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_not_numeric")

    def test_score_nan_rejected(self):
        rows = [{"candidate": "topic-auth", "raw": 0.5}, {"candidate": "topic-billing", "raw": 0.5},
                {"candidate": "novel", "raw": 0.5}, {"candidate": "ambiguous", "raw": float("nan")}]
        body = json.dumps({"scores": rows})  # json.dumps emits a bare NaN token
        self.assertIn("NaN", body)
        self.assert_contract_reject([content_body(body)], "score_not_finite")

    def test_score_infinity_rejected(self):
        rows = [{"candidate": "topic-auth", "raw": 0.5}, {"candidate": "topic-billing", "raw": 0.5},
                {"candidate": "novel", "raw": float("inf")}, {"candidate": "ambiguous", "raw": 0.5}]
        body = json.dumps({"scores": rows})
        self.assertIn("Infinity", body)
        self.assert_contract_reject([content_body(body)], "score_not_finite")

    def test_score_above_range_rejected(self):
        rows = [{"candidate": "topic-auth", "raw": 1.0001}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_out_of_range")

    def test_score_below_range_rejected(self):
        rows = [{"candidate": "topic-auth", "raw": -0.0001}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "score_out_of_range")

    def test_missing_candidate_id_rejected(self):
        rows = [{"candidate": c, "raw": 0.5} for c in TOPIC_IDS if c != "novel"]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "candidate_set_mismatch")

    def test_extra_candidate_id_rejected(self):
        rows = good_scores() + [{"candidate": "topic-payroll", "raw": 0.5}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "candidate_set_mismatch")

    def test_unknown_candidate_id_rejected(self):
        rows = good_scores() + [{"candidate": "sess-a", "raw": 0.5}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "candidate_set_mismatch")

    def test_duplicate_candidate_id_rejected(self):
        rows = good_scores() + [{"candidate": "topic-auth", "raw": 0.5}]
        self.assert_contract_reject([content_body(json.dumps({"scores": rows}))], "duplicate_candidate_id")

    def test_unknown_surface_rejected(self):
        backend = make_backend(MockTransport([]))
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "modality")
        self.assertEqual(ctx.exception.code, "unknown_surface")

    def test_valid_reply_is_normalised_to_candidate_order(self):
        shuffled = list(reversed(good_scores()))
        backend = make_backend(MockTransport([scores_body(shuffled)]))
        rows = backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual([row["candidate"] for row in rows], TOPIC_IDS)
        self.assertEqual(rows[0], {"candidate": "topic-auth", "raw": 0.5})

    def test_validate_batch_reply_is_directly_usable(self):
        document = {"choices": [{"message": {"content": json.dumps({"scores": good_scores()})}}]}
        rows = validate_batch_reply(document, TOPIC_IDS, provider_id="jev")
        self.assertEqual(len(rows), len(TOPIC_IDS))

    def test_boundaries_zero_and_one_are_accepted(self):
        rows = [
            {"candidate": "topic-auth", "raw": 1},
            {"candidate": "topic-billing", "raw": 0},
            {"candidate": "novel", "raw": 1.0},
            {"candidate": "ambiguous", "raw": 0.0},
        ]
        backend = make_backend(MockTransport([scores_body(rows)]))
        scores = backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual([row["raw"] for row in scores], [1.0, 0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# D4/D5 -- transport boundary: SSRF refusals, redirects, injection seam
# --------------------------------------------------------------------------


class TestTransportBoundary(KeyEnvTestCase):
    def test_redirect_handler_refuses_every_redirect(self):
        handler = NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example.com/")
        )
        self.assertIsNone(
            handler.redirect_request(None, None, 301, "Moved", {}, "http://127.0.0.1:1/")
        )

    def test_redirect_status_is_refused_and_not_followed(self):
        transport = MockTransport(
            [TransportResponse(302, {"location": "https://evil.example.com/chat/completions"}, b"")]
        )
        backend = make_backend(transport, max_retries=0)
        with self.assertRaises(ProviderTransportError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "redirect_refused")
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(
            [request.url for request in transport.requests],
            ["https://provider.example.com/v1/chat/completions"],
        )

    def test_http_plain_non_private_host_refused_by_the_loader(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            parse_provider_row(provider_row(base_url="http://provider.example.com/v1"), 0)
        self.assertEqual(ctx.exception.code, "url_scheme_refused")

    def test_http_loopback_requires_the_explicit_opt_in(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            parse_provider_row(provider_row(base_url="http://127.0.0.1:8080/v1"), 0)
        self.assertEqual(ctx.exception.code, "url_scheme_refused")

    def test_http_private_host_allowed_with_opt_in(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(
            transport, base_url="http://127.0.0.1:8080/v1", allow_private_network=True
        )
        rows = backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(len(rows), len(TOPIC_IDS))
        self.assertEqual(transport.requests[0].url, "http://127.0.0.1:8080/v1/chat/completions")

    def test_http_public_host_refused_even_with_opt_in(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            parse_provider_row(
                provider_row(base_url="http://provider.example.com/v1", allow_private_network=True), 0
            )
        self.assertEqual(ctx.exception.code, "private_network_not_allowed")

    def test_non_http_schemes_refused(self):
        for base_url in ("file:///etc/passwd", "ftp://provider.example.com/v1", "//provider.example.com/v1"):
            with self.assertRaises(ProviderConfigError) as ctx:
                parse_provider_row(provider_row(base_url=base_url), 0)
            self.assertIn(ctx.exception.code, {"url_scheme_refused", "base_url_invalid"})

    def test_url_with_embedded_credentials_refused(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            parse_provider_row(provider_row(base_url="https://user:pass@provider.example.com/v1"), 0)
        self.assertEqual(ctx.exception.code, "url_userinfo_refused")

    def test_url_with_query_or_fragment_refused(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            parse_provider_row(provider_row(base_url="https://provider.example.com/v1?key=1"), 0)
        self.assertEqual(ctx.exception.code, "base_url_invalid")

    def test_refused_host_is_blocked_before_any_transport_call(self):
        # a crafted row that bypasses the loader (the loader would already refuse it)
        from ordaprompt_router.providers import OpenAICompatibleBackend, ProviderRow

        row = ProviderRow(
            id="evil",
            kind="openai_compatible",
            base_url="http://metadata.example.internal/v1",
            model="m",
            api_key_env=KEY_ENV,
        )
        transport = MockTransport([scores_body(good_scores())])
        backend = OpenAICompatibleBackend(row, api_key=KEY_VALUE, transport=transport)
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "url_scheme_refused")
        self.assertEqual(transport.call_count, 0, "a refused host must not reach the transport")

    def test_request_url_is_built_from_the_row_only(self):
        hostile = json.dumps({"scores": good_scores(), "next_url": "https://evil.example.com/chat/completions"})
        transport = MockTransport([content_body(hostile), scores_body(good_scores())])
        backend = make_backend(transport, max_retries=0)
        with self.assertRaises(ProviderContractError):
            backend.batch_score(HANDLE, make_candidates(), "topic")
        backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(
            [request.url for request in transport.requests],
            ["https://provider.example.com/v1/chat/completions"] * 2,
        )

    def test_no_env_var_can_move_the_request_url(self):
        self.set_key()
        os.environ["ORDAPROMPT_PROVIDER_BASE_URL"] = "https://evil.example.com/v1"
        self.addCleanup(os.environ.pop, "ORDAPROMPT_PROVIDER_BASE_URL", None)
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(transport.requests[0].url, "https://provider.example.com/v1/chat/completions")

    def test_payload_is_hash_and_id_only(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(), "topic")
        body = transport.requests[0].body.decode("utf-8")
        self.assertIn(sha256_hex(PROMPT_TEXT), body)
        self.assertNotIn(PROMPT_TEXT, body)
        self.assertNotIn("REFUND", body)
        self.assertNotIn("invoice", body)
        payload = json.loads(body)
        self.assertEqual(payload["model"], "jev-classifier-1")
        self.assertEqual(payload["messages"][1]["candidates"], TOPIC_IDS)
        self.assertEqual(len(payload["messages"]), 2)

    def test_transport_is_injectable_and_stdlib_is_the_default(self):
        row = parse_provider_row(provider_row(), 0)
        injected = OpenAICompatibleBackend(row, api_key=KEY_VALUE, transport=MockTransport([]))
        self.assertIsNotNone(injected._transport)
        default = OpenAICompatibleBackend(row, api_key=KEY_VALUE)
        self.assertIsNone(default._transport)
        self.assertTrue(callable(stdlib_transport))

    def test_loopback_and_private_host_classifier(self):
        for host in ("localhost", "127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5", "::1", "db.localhost"):
            self.assertTrue(is_loopback_or_private_host(host), host)
        for host in ("provider.example.com", "8.8.8.8", "172.32.0.1", "", "evil.local"):
            self.assertFalse(is_loopback_or_private_host(host), host)


# --------------------------------------------------------------------------
# D3/D6 -- batch semantics: ONE call per surface, bounded retries
# --------------------------------------------------------------------------


class TestBatchSemantics(KeyEnvTestCase):
    def test_one_transport_post_per_surface(self):
        transport = MockTransport([scores_body(good_scores()), scores_body(good_scores(SESSION_IDS))])
        backend = make_backend(transport)
        backend.batch_score(HANDLE, make_candidates(with_sessions=True), "topic")
        self.assertEqual(backend.batch_calls, {"topic": 1, "session": 0})
        self.assertEqual(transport.call_count, 1)
        backend.batch_score(HANDLE, make_candidates(with_sessions=True), "session")
        self.assertEqual(backend.batch_calls, {"topic": 1, "session": 1})
        self.assertEqual(transport.call_count, 2)
        for request in transport.requests:
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url, "https://provider.example.com/v1/chat/completions")
        self.assertEqual(backend.transport_calls, 2)

    def test_batch_too_large_is_refused_before_the_network(self):
        transport = MockTransport([])
        backend = make_backend(transport, batch_max_candidates=2)
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "batch_too_large")
        self.assertEqual(transport.call_count, 0)

    def test_transport_failure_retries_at_most_the_declared_budget(self):
        transport = MockTransport(
            [
                ProviderTransportError("transport_failed", "connection refused"),
                ProviderTransportError("transport_failed", "connection refused"),
            ]
        )
        backend = make_backend(transport, max_retries=1)
        with self.assertRaises(ProviderTransportError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "transport_failed")
        self.assertEqual(transport.call_count, 2)

    def test_no_retry_without_budget(self):
        transport = MockTransport([ProviderTransportError("transport_failed", "connection refused")])
        backend = make_backend(transport, max_retries=0)
        with self.assertRaises(ProviderTransportError):
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(transport.call_count, 1)

    def test_schema_error_is_never_retried(self):
        transport = MockTransport([content_body("not json at all")])
        backend = make_backend(transport, max_retries=3)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "reply_content_not_json")
        self.assertEqual(transport.call_count, 1, "a contract violation must not be retried")

    def test_server_error_is_retried_within_budget(self):
        transport = MockTransport([TransportResponse(503, {}, b""), scores_body(good_scores())])
        backend = make_backend(transport, max_retries=1)
        rows = backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(len(rows), len(TOPIC_IDS))
        self.assertEqual(transport.call_count, 2)

    def test_client_error_is_not_retried(self):
        transport = MockTransport([TransportResponse(400, {}, b"bad request"), scores_body(good_scores())])
        backend = make_backend(transport, max_retries=1)
        with self.assertRaises(ProviderTransportError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "http_status")
        self.assertEqual(transport.call_count, 1)


# --------------------------------------------------------------------------
# D6 -- raw scores flow through calibration unchanged
# --------------------------------------------------------------------------


class TestScoreIntegrity(KeyEnvTestCase):
    def test_raw_scores_reach_the_receipt_unchanged(self):
        values = [0.91, 0.12, 0.05, 0.07]
        rows = [{"candidate": c, "raw": v} for c, v in zip(TOPIC_IDS, values)]
        backend = make_backend(MockTransport([scores_body(rows)]))
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        by_id = {row["candidate"]: row for row in result.receipt["topic_scores"]}
        for candidate, value in zip(TOPIC_IDS, values):
            self.assertEqual(by_id[candidate]["raw"], value)
            self.assertEqual(by_id[candidate]["calibrated"], value)
        self.assertEqual(result.receipt["backend_used"], "openjev")

    def test_identity_calibration_keeps_provider_scores(self):
        rows = [{"candidate": c, "raw": v} for c, v in zip(TOPIC_IDS, [0.9, 0.1, 0.05, 0.05])]
        backend = make_backend(MockTransport([scores_body(rows)]))
        calibration = CalibrationModel(model_id="platt-v1", active=True)
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend, calibration=calibration),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        by_id = {row["candidate"]: row for row in result.receipt["topic_scores"]}
        self.assertAlmostEqual(by_id["topic-auth"]["calibrated"], 0.9, places=9)
        self.assertEqual(result.receipt["calibration_model_id"], "platt-v1")

    def test_provider_backend_never_routes_automatic_without_calibration(self):
        rows = [{"candidate": c, "raw": v} for c, v in zip(TOPIC_IDS, [0.99, 0.01, 0.0, 0.0])]
        backend = make_backend(MockTransport([scores_body(rows)]))
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertNotEqual(result.decision.band, "automatic")
        self.assertEqual(result.topic_escalation_code, "calibration_missing")


# --------------------------------------------------------------------------
# D3 -- declared fallback chain, never silent
# --------------------------------------------------------------------------


class TestFallbackChain(KeyEnvTestCase):
    def chain_registry(self):
        rows = [
            provider_row(),
            provider_row(id="backup", base_url="https://backup.example.com/v1", model="backup-1"),
        ]
        return ProviderRegistry.from_dict(
            providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        )

    def test_failure_without_a_declared_chain_abstains(self):
        backend = make_backend(
            MockTransport([ProviderTransportError("transport_failed", "refused")]), max_retries=0
        )
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.decision.band, "fallback_escalate")
        self.assertEqual(result.decision.decision, "escalate")
        self.assertEqual(result.escalation_code, "backend_error")
        self.assertNotIn("fallback_from", result.receipt)

    def test_declared_chain_falls_back_and_records_fallback_from(self):
        self.set_key()
        failed = ProviderTransportError("transport_failed", "connection refused")
        transport = MockTransport([failed, failed, scores_body(good_scores())])
        registry = self.chain_registry()
        backend = cast(ProviderChainBackend, registry.select_backend(transport))
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        self.assertEqual(result.receipt["fallback_from"], "jev")
        self.assertEqual(result.receipt["backend_used"], "openjev")
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(result.decision.topic_scores[0]["candidate"], "topic-auth")
        validate_receipt(result.receipt)
        assert_no_free_text(result.receipt)

    def test_fallback_receipt_round_trips_through_the_store(self):
        self.set_key()
        failed = ProviderTransportError("transport_failed", "connection refused")
        transport = MockTransport([failed, failed, scores_body(good_scores())])
        backend = cast(ProviderChainBackend, self.chain_registry().select_backend(transport))
        result = route(
            make_request(), make_candidates(), RouterConfig(backend=backend),
            include_session=False, receipt_id=REQUEST_ID, ts="2026-09-22T10:00:00.000Z",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(tmp)
            store.append(result.receipt)
            stored = store.read_all()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["fallback_from"], "jev")
            validate_receipt(stored[0])

    def test_fallback_events_are_recorded(self):
        self.set_key()
        failed = ProviderTransportError("transport_failed", "connection refused")
        transport = MockTransport([failed, failed, scores_body(good_scores())])
        backend = cast(ProviderChainBackend, self.chain_registry().select_backend(transport))
        backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(backend.last_fallback_from, "jev")
        self.assertEqual(
            backend.fallback_events,
            [{"surface": "topic", "from": "jev", "to": "backup", "code": "transport_failed"}],
        )

    def test_contract_violation_never_falls_back(self):
        self.set_key()
        bad = scores_body([{"candidate": "topic-auth", "raw": 0.5}])  # incomplete candidate set
        backup_ok = scores_body(good_scores())
        transport = MockTransport([bad, backup_ok])
        backend = cast(ProviderChainBackend, self.chain_registry().select_backend(transport))
        with self.assertRaises(ProviderContractError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(ctx.exception.code, "candidate_set_mismatch")
        self.assertEqual(transport.call_count, 1, "an untrustworthy reply must not be papered over")
        self.assertIsNone(backend.last_fallback_from)

    def test_chain_abstains_when_every_member_is_unavailable(self):
        self.set_key()
        rows = [
            provider_row(max_retries=0),
            provider_row(id="backup", base_url="https://backup.example.com/v1", model="backup-1", max_retries=0),
        ]
        registry = ProviderRegistry.from_dict(
            providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        )
        failed = ProviderTransportError("transport_failed", "connection refused")
        transport = MockTransport([failed, failed])
        backend = cast(ProviderChainBackend, registry.select_backend(transport))
        with self.assertRaises(ProviderTransportError):
            backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(backend.last_fallback_from, "jev")


# --------------------------------------------------------------------------
# D3 -- taxonomy proposals are refused unless a row enables them
# --------------------------------------------------------------------------


class TestTaxonomyProposal(KeyEnvTestCase):
    def test_proposal_is_refused_by_default(self):
        backend = make_backend(MockTransport([]))
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)])
        self.assertEqual(ctx.exception.code, "taxonomy_proposal_not_enabled")

    def test_chain_refuses_when_no_member_enables_proposals(self):
        self.set_key()
        rows = [
            provider_row(),
            provider_row(id="backup", base_url="https://backup.example.com/v1", model="backup-1"),
        ]
        registry = ProviderRegistry.from_dict(
            providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        )
        backend = registry.select_backend(MockTransport([]))
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)])
        self.assertEqual(ctx.exception.code, "taxonomy_proposal_not_enabled")

    def test_proposal_is_allowed_when_the_row_enables_it(self):
        payload = json.dumps({"slug": "topic-refunds", "parent_slug": None, "confidence": 0.95})
        transport = MockTransport([content_body(payload)])
        backend = make_backend(transport, supports_taxonomy_proposal=True)
        proposal = backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)])
        self.assertEqual(proposal, {"slug": "topic-refunds", "parent_slug": None, "confidence": 0.95})
        self.assertEqual(transport.call_count, 1)

    def test_low_confidence_proposal_is_refused(self):
        payload = json.dumps({"slug": "topic-refunds", "parent_slug": None, "confidence": 0.2})
        backend = make_backend(MockTransport([content_body(payload)]), supports_taxonomy_proposal=True)
        self.assertIsNone(backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)]))

    def test_malformed_proposal_slug_is_refused(self):
        payload = json.dumps({"slug": "Topic Refunds!", "parent_slug": None, "confidence": 0.99})
        backend = make_backend(MockTransport([content_body(payload)]), supports_taxonomy_proposal=True)
        self.assertIsNone(backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)]))

    def test_proposal_unknown_field_is_a_hard_error(self):
        payload = json.dumps({"slug": "topic-refunds", "confidence": 0.99, "reasoning": "free text"})
        backend = make_backend(MockTransport([content_body(payload)]), supports_taxonomy_proposal=True)
        with self.assertRaises(ProviderContractError) as ctx:
            backend.propose_taxonomy(HANDLE, [sha256_hex(PROMPT_TEXT)])
        self.assertEqual(ctx.exception.code, "proposal_unknown_field")


# --------------------------------------------------------------------------
# CLI wiring (D2): the flag is the only entry point; bad files fail closed
# --------------------------------------------------------------------------


class TestCliProviderWiring(KeyEnvTestCase):
    def run_cli(self, args, providers_document=None, extra_env=None):
        demo = ROOT / "eval" / "demo"
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable, "-m", "ordaprompt_router.cli", "classify",
                "--request", str(demo / "request.json"),
                "--candidates", str(demo / "candidates.json"),
            ]
            if providers_document is not None:
                path = Path(tmp) / "providers.json"
                if isinstance(providers_document, str):
                    path.write_text(providers_document, encoding="utf-8")
                else:
                    path.write_text(json.dumps(providers_document), encoding="utf-8")
                command += ["--providers", str(path)]
            command += list(args)
            env = dict(os.environ)
            env.pop(KEY_ENV, None)
            if extra_env:
                env.update(extra_env)
            return subprocess.run(
                command, cwd=str(ROOT), capture_output=True, text=True, timeout=60, env=env
            )

    def test_local_provider_row_runs_fully_offline(self):
        proc = self.run_cli([], providers_document=providers_config(rows=[{"id": "local-a", "kind": "local"}]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        receipt = json.loads(proc.stdout)["routing_receipt"]
        self.assertEqual(receipt["backend_used"], "synthetic")
        self.assertNotIn("fallback_from", receipt)
        validate_receipt(receipt)

    def test_unparsable_providers_file_fails_closed(self):
        proc = self.run_cli([], providers_document="{broken")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertIn("provider_reject", proc.stderr)
        self.assertIn("config_unparsable_json", proc.stderr)

    def test_unknown_default_provider_fails_closed(self):
        proc = self.run_cli([], providers_document=providers_config(default_provider="nope"))
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertIn("unknown_provider_id", proc.stderr)

    def test_unset_api_key_env_fails_closed_without_fallback(self):
        rows = [
            provider_row(),
            provider_row(id="backup", base_url="https://backup.example.com/v1", model="backup-1"),
        ]
        document = providers_config(rows=rows, default_provider="jev", allow_fallback=True)
        proc = self.run_cli([], providers_document=document)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertIn("api_key_env_unset", proc.stderr)

    def test_missing_providers_file_fails_closed(self):
        demo = ROOT / "eval" / "demo"
        proc = subprocess.run(
            [
                sys.executable, "-m", "ordaprompt_router.cli", "classify",
                "--request", str(demo / "request.json"), "--topics", "topic-auth",
                "--providers", "/nonexistent/providers.json",
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("config_unreadable", proc.stderr)

    def test_refused_host_fails_closed_at_config_time(self):
        document = providers_config(rows=[provider_row(base_url="http://provider.example.com/v1")])
        proc = self.run_cli([], providers_document=document)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("url_scheme_refused", proc.stderr)


# --------------------------------------------------------------------------
# D7 -- wording: "Jev-style" only, no bundled model, no key in the docs
# --------------------------------------------------------------------------


class TestWordingAndDocs(unittest.TestCase):
    def read(self, name: str) -> str:
        return (ROOT / name).read_text(encoding="utf-8")

    def test_readme_documents_provider_configuration(self):
        readme = self.read("README.md")
        self.assertIn("Configuring a provider", readme)
        self.assertIn("--providers", readme)
        self.assertIn("api_key_env", readme)
        self.assertIn("Jev-style", readme)
        self.assertIn("fallback_from", readme)

    def test_readme_carries_no_key_material(self):
        readme = self.read("README.md")
        self.assertNotIn("sk-", readme)
        self.assertNotIn("Bearer ", readme)
        # no positive claim of a bundled model or a tested live integration (D7)
        self.assertNotIn("bundles a Jev model", readme)
        self.assertNotIn("tested against a live Jev", readme)
        self.assertIn("does not claim", readme)
        self.assertIn("bundles no model", readme)

    def test_manifest_keeps_the_empty_config_schema_and_new_version(self):
        plugin = self.read("plugin.yaml")
        self.assertIn("config_schema: {}", plugin)
        self.assertIn("version: 1.1.0", plugin)
        self.assertNotIn("sk-", plugin)

    def test_jev_row_is_a_plain_openai_compatible_row(self):
        transport = MockTransport([scores_body(good_scores())])
        backend = make_backend(transport, id="jev")
        backend.batch_score(HANDLE, make_candidates(), "topic")
        payload = json.loads(transport.requests[0].body.decode("utf-8"))
        self.assertEqual(payload["model"], "jev-classifier-1")
        self.assertEqual(transport.requests[0].url, "https://provider.example.com/v1/chat/completions")
        self.assertEqual(backend.name, "openjev")
        # no bundled model artifact is shipped for the jev row
        self.assertEqual(list(ROOT.rglob("*jev*.bin")), [])

    def test_package_version_is_bumped(self):
        import ordaprompt_router

        self.assertEqual(ordaprompt_router.__version__, "1.1.0")


if __name__ == "__main__":
    unittest.main()
