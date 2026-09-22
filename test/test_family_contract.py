"""Offline tests for the LOCKED provider-family contract (leo-adapter-architecture.md).

Scope of this file: the parts of the sealed architecture that the provider layer newly
implements at this head -- the closed five-family taxonomy and its fail-closed refusal of the
designed-but-unimplemented families, the per-family batch limits (L25), the data-class and
surface eligibility filters (section 2.3 / L21), capability discovery (L14), and the
calibration-state consistency gate plus fitted-key binding (section 6.4 / L23 / L16).

Every test here is offline: the transports are injected fakes, no socket is ever opened, no
third-party service is contacted and no real credential is used (the environment value below
is a literal non-secret placeholder).  The `SocketGuard` transport exists so that a test which
accidentally reaches the network FAILS instead of dialling out.

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
from typing import Any, Dict, List, Optional, cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ordaprompt_router import (  # noqa: E402
    FAMILY_BATCH_LIMITS,
    IMPLEMENTED_KINDS,
    PROVIDER_KINDS,
    ROW_DATA_CLASSES,
    SURFACES,
    CandidateSet,
    ClassificationRequest,
    OpenAICompatibleBackend,
    ProviderConfigError,
    ProviderRegistry,
    ProviderRow,
    RequestHandle,
    RouterConfig,
    SyntheticBackend,
    TransportRequest,
    TransportResponse,
    local_backend_key,
    route,
    sha256_hex,
)
from ordaprompt_router.providers import (  # noqa: E402
    DESIGNED_UNIMPLEMENTED_KINDS,
    FAMILY_BATCH_LIMITS as _FAMILY_LIMITS,
    is_fallback_trigger,
    parse_provider_row,
)
from ordaprompt_router.router import CalibrationModel  # noqa: E402

PROMPT_TEXT = "REFUND my invoice for the billing cycle, this is urgent"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"
CONFIG_REV = "1.0.0"
PROJECT_HASH = "sha256:" + "ef" * 32
KEY_ENV = "ORDAPROMPT_FAMILY_TEST_KEY"
KEY_VALUE = "sk-family-00112233445566778899aabbccddeeff"
TOPIC_IDS = ["topic-auth", "topic-billing", "novel", "ambiguous"]
SESSION_IDS = ["sess-a", "new_session"]

#: the four family selectors the locked section-8.1 row schema adds
FAMILY_ROW_FIELDS = ("transform", "endpoint_mode", "data_class", "surfaces")


# --------------------------------------------------------------------------
# fixtures / fake transport
# --------------------------------------------------------------------------


def providers_config(rows: List[Dict[str, Any]], **over: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = {"schema_version": 1, "providers": rows}
    config.update(over)
    return config


def network_row(**over: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": "jev",
        "kind": "openai_compatible",
        "base_url": "https://provider.example.com/v1",
        "model": "family-model-1",
        "api_key_env": KEY_ENV,
    }
    row.update(over)
    return row


def local_row(**over: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": "local-default", "kind": "local"}
    row.update(over)
    return row


class SocketGuard:
    """Transport that FAILS the test if it is ever invoked (S-inv 7 / L5)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.calls += 1
        raise AssertionError("a test attempted a network call: %s %s" % (request.method, request.url))


class RecordingTransport:
    """Records requests, replays queued responses; never opens a socket (D5)."""

    def __init__(self, replies: Optional[List[Any]] = None) -> None:
        self.replies = list(replies or [])
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
        return cast(TransportResponse, item)


def scores_body(ids: List[str], key: str = "scores") -> TransportResponse:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {key: [{"candidate": cid, "raw": 0.5} for cid in ids]}
                    )
                }
            }
        ]
    }
    return TransportResponse(status=200, headers={}, body=json.dumps(payload).encode("utf-8"))


def make_request() -> ClassificationRequest:
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


def make_candidates(with_sessions: bool = True) -> CandidateSet:
    return CandidateSet.from_dict(
        {
            "schema": "ordapilot.candidate_set/1",
            "request_id": REQUEST_ID,
            "topics": [
                {"topic_id": "topic-auth", "label_state": "promoted", "label_age_days": 1, "evidence_count": 2},
                {"topic_id": "topic-billing", "label_state": "promoted", "label_age_days": 2, "evidence_count": 1},
            ],
            "topic_sentinels": ["novel", "ambiguous"],
            "sessions": (
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
            ),
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


def load_row(row: Dict[str, Any], index: int = 0) -> ProviderRow:
    return parse_provider_row(row, index)


class KeyEnvTestCase(unittest.TestCase):
    """Base class that owns the placeholder credential in the environment."""

    def setUp(self) -> None:
        self._saved = os.environ.get(KEY_ENV)
        os.environ[KEY_ENV] = KEY_VALUE

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(KEY_ENV, None)
        else:
            os.environ[KEY_ENV] = self._saved


# --------------------------------------------------------------------------
# 1. The closed family taxonomy (section 2.1) and the unimplemented families
# --------------------------------------------------------------------------


class TestClosedFamilyTaxonomy(unittest.TestCase):
    def test_kind_enum_is_the_locked_five_families(self):
        self.assertEqual(
            tuple(PROVIDER_KINDS),
            ("local", "openai_compatible", "jev_decision", "nanojev_batch", "laya_local"),
        )

    def test_only_two_families_have_a_real_adapter_at_this_head(self):
        self.assertEqual(tuple(IMPLEMENTED_KINDS), ("local", "openai_compatible"))
        self.assertEqual(
            tuple(DESIGNED_UNIMPLEMENTED_KINDS),
            ("jev_decision", "nanojev_batch", "laya_local"),
        )
        self.assertEqual(
            set(IMPLEMENTED_KINDS) | set(DESIGNED_UNIMPLEMENTED_KINDS), set(PROVIDER_KINDS)
        )
        self.assertFalse(set(IMPLEMENTED_KINDS) & set(DESIGNED_UNIMPLEMENTED_KINDS))

    def test_unknown_kind_is_a_hard_config_error(self):
        for token in ("openai-compat", "openai", "jev", ""):
            with self.assertRaises(ProviderConfigError) as ctx:
                ProviderRegistry.from_dict(providers_config(rows=[network_row(kind=token)]))
            self.assertIn(ctx.exception.code, ("kind_unknown", "field_invalid"))

    def test_jev_decision_requires_a_transform_before_it_is_refused(self):
        # the shaped request/response mapping: a jev_decision row without a declared
        # transform can never be silently assumed to speak /chat/completions
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(
                {
                    "id": "jev-native",
                    "kind": "jev_decision",
                    "base_url": "https://classifier.example.com/v1",
                    "model": "typesafe-systemone",
                    "api_key_env": KEY_ENV,
                }
            )
        self.assertEqual(ctx.exception.code, "transform_required")

        with self.assertRaises(ProviderConfigError) as ctx2:
            load_row(
                {
                    "id": "jev-native",
                    "kind": "jev_decision",
                    "base_url": "https://classifier.example.com/v1",
                    "model": "typesafe-systemone",
                    "api_key_env": KEY_ENV,
                    "transform": "sj-choice-batch-v1",
                }
            )
        self.assertEqual(ctx2.exception.code, "kind_unimplemented")

    def test_nanojev_batch_is_unimplemented(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(
                {
                    "id": "nanojev",
                    "kind": "nanojev_batch",
                    "base_url": "https://nanojev.example.com",
                    "model": "nanojev-1",
                    "api_key_env": KEY_ENV,
                }
            )
        self.assertEqual(ctx.exception.code, "kind_unimplemented")

    def test_laya_local_requires_endpoint_mode_then_is_unimplemented(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row({"id": "laya", "kind": "laya_local", "model": "laya-1", "transform": "laya-choice-v1"})
        self.assertEqual(ctx.exception.code, "field_invalid")

        for mode in ("in_process", "loopback_sidecar"):
            with self.assertRaises(ProviderConfigError) as ctx2:
                load_row(
                    {
                        "id": "laya",
                        "kind": "laya_local",
                        "model": "laya-1",
                        "transform": "laya-choice-v1",
                        "endpoint_mode": mode,
                    }
                )
            self.assertEqual(ctx2.exception.code, "kind_unimplemented")

    def test_transform_is_not_allowed_on_chat_completions_kinds(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(network_row(transform="sj-choice-batch-v1"))
        self.assertEqual(ctx.exception.code, "field_invalid")
        with self.assertRaises(ProviderConfigError) as ctx2:
            load_row(local_row(transform="laya-choice-v1"))
        self.assertEqual(ctx2.exception.code, "field_invalid")

    def test_unknown_transform_label_is_refused(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(
                {
                    "id": "jev-native",
                    "kind": "jev_decision",
                    "base_url": "https://classifier.example.com/v1",
                    "model": "typesafe-systemone",
                    "api_key_env": KEY_ENV,
                    "transform": "chat-completions-v2",
                }
            )
        self.assertEqual(ctx.exception.code, "field_invalid")

    def test_endpoint_mode_is_only_meaningful_for_laya_local(self):
        for kind in ("local", "openai_compatible"):
            with self.assertRaises(ProviderConfigError) as ctx:
                load_row(network_row(kind=kind, endpoint_mode="in_process"))
            self.assertEqual(ctx.exception.code, "field_invalid")

    def test_unimplemented_family_rows_never_enter_a_registry_or_a_chain(self):
        for kind, extra in (
            ("jev_decision", {"transform": "sj-choice-batch-v1"}),
            ("nanojev_batch", {}),
        ):
            with self.assertRaises(ProviderConfigError) as ctx:
                ProviderRegistry.from_dict(
                    providers_config(rows=[network_row(kind=kind, **extra)])
                )
            self.assertEqual(ctx.exception.code, "kind_unimplemented")

    def test_unimplemented_family_error_is_never_a_fallback_trigger(self):
        error = ProviderConfigError("kind_unimplemented", "designed, not shipped")
        self.assertFalse(is_fallback_trigger(error), "configuration failures never fall back")


# --------------------------------------------------------------------------
# 2. Per-family batch limits bind at load (L25)
# --------------------------------------------------------------------------


class TestFamilyBatchLimits(unittest.TestCase):
    def test_family_limits_are_the_documented_caps(self):
        self.assertEqual(
            _FAMILY_LIMITS,
            {
                "local": 256,
                "openai_compatible": 256,
                "jev_decision": 50,
                "nanojev_batch": 255,
                "laya_local": 20,
            },
        )

    def test_row_above_its_family_limit_is_field_invalid(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(network_row(batch_max_candidates=257))
        self.assertEqual(ctx.exception.code, "field_invalid")
        self.assertIn("batch_max_candidates", str(ctx.exception))
        # 257 is 1 above the family cap, so the family rule itself is what rejects it
        with self.assertRaises(ProviderConfigError) as ctx2:
            load_row(network_row(batch_max_candidates=200, kind="jev_decision", transform="sj-choice-batch-v1"))
        self.assertEqual(ctx2.exception.code, "field_invalid")

    def test_family_limit_binds_before_the_unimplemented_refusal(self):
        # the limit check is a schema rule, so it fires first and names the cap
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(
                {
                    "id": "laya",
                    "kind": "laya_local",
                    "model": "laya-1",
                    "transform": "laya-choice-v1",
                    "endpoint_mode": "in_process",
                    "batch_max_candidates": 21,
                }
            )
        self.assertEqual(ctx.exception.code, "field_invalid")
        self.assertIn("family limit", str(ctx.exception))

    def test_boundary_values_are_accepted(self):
        row = load_row(network_row(batch_max_candidates=256))
        self.assertEqual(row.batch_max_candidates, 256)
        self.assertEqual(row.batch_limit, 256)
        local = load_row(local_row(batch_max_candidates=1))
        self.assertEqual(local.batch_max_candidates, 1)

    def test_omitted_cap_defaults_inside_the_family_limit(self):
        # the generic default (64) still applies to the implemented families ...
        self.assertEqual(load_row(network_row()).batch_max_candidates, 64)
        # ... and for a family whose cap is lower, the cap is applied BEFORE the default, so
        # an omitted field never manufactures a spurious field_invalid
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(
                {
                    "id": "laya",
                    "kind": "laya_local",
                    "model": "laya-1",
                    "transform": "laya-choice-v1",
                    "endpoint_mode": "in_process",
                }
            )
        self.assertEqual(ctx.exception.code, "kind_unimplemented")


# --------------------------------------------------------------------------
# 3. Data-class eligibility (section 2.3 / L21)
# --------------------------------------------------------------------------


class TestDataClassEligibility(KeyEnvTestCase):
    def test_local_rows_default_to_private_and_network_rows_to_public(self):
        self.assertEqual(load_row(local_row()).data_class, "private")
        self.assertEqual(load_row(network_row()).data_class, "public")
        self.assertTrue(load_row(local_row()).is_in_process)
        self.assertFalse(load_row(network_row()).is_in_process)

    def test_private_is_refused_on_a_network_row(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(network_row(data_class="private"))
        self.assertEqual(ctx.exception.code, "field_invalid")
        self.assertIn("private", str(ctx.exception))

    def test_internal_override_is_allowed_for_a_self_hosted_row(self):
        row = load_row(network_row(data_class="internal"))
        self.assertEqual(row.data_class, "internal")
        self.assertTrue(row.requires_network)

    def test_data_class_domain_is_closed(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            load_row(network_row(data_class="secret"))
        self.assertEqual(ctx.exception.code, "field_invalid")

    def test_private_request_never_sees_a_public_row(self):
        registry = ProviderRegistry.from_dict(
            providers_config(
                rows=[network_row(data_class="internal"), local_row()],
                default_provider="local-default",
            )
        )
        self.assertTrue(registry.rows_for("topic", "public"))
        self.assertEqual(registry.rows_for("topic", "private"), ["local-default"])
        self.assertEqual(registry.rows_for("topic", "restricted"), [])
        self.assertEqual(registry.rows_for("topic", "internal"), ["jev", "local-default"])

    def test_named_head_below_the_request_class_is_a_hard_error(self):
        registry = ProviderRegistry.from_dict(
            providers_config(rows=[network_row(data_class="internal")])
        )
        guard = SocketGuard()
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(transport=guard, request_class="private")
        self.assertEqual(ctx.exception.code, "data_class_refused")
        self.assertEqual(guard.calls, 0, "a refused request must never open a socket")

    def test_restricted_request_can_never_be_served_by_any_row(self):
        registry = ProviderRegistry.from_dict(providers_config(rows=[local_row()]))
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(request_class="restricted")
        self.assertEqual(ctx.exception.code, "data_class_refused")

    def test_unknown_request_class_fails_closed(self):
        row = load_row(network_row())
        self.assertFalse(row.class_allows("top-secret"))
        self.assertEqual(row.class_allows("public"), True)

    def test_private_request_selects_the_in_process_row(self):
        registry = ProviderRegistry.from_dict(providers_config(rows=[local_row()]))
        backend = registry.select_backend(request_class="private")
        self.assertIsInstance(backend, SyntheticBackend)

    def test_public_request_still_selects_the_network_row(self):
        registry = ProviderRegistry.from_dict(providers_config(rows=[network_row()]))
        backend = registry.select_backend(transport=SocketGuard(), request_class="public")
        self.assertIsInstance(backend, OpenAICompatibleBackend)


# --------------------------------------------------------------------------
# 4. Surface declarations (section 2.3) -- a row that does not list a surface
#    is never offered it, and the refusal happens before any socket
# --------------------------------------------------------------------------


class TestSurfaceEligibility(KeyEnvTestCase):
    def test_surfaces_list_is_closed_nonempty_and_duplicate_free(self):
        for bad in (["topic", "bogus"], [], ["topic", "topic"], "topic", [1, 2]):
            with self.assertRaises(ProviderConfigError) as ctx:
                load_row(network_row(surfaces=bad))
            self.assertEqual(ctx.exception.code, "field_invalid")

    def test_default_surface_set_is_all_three_and_order_is_preserved(self):
        self.assertEqual(tuple(load_row(network_row()).surfaces), tuple(SURFACES))
        self.assertEqual(load_row(network_row(surfaces=["session", "topic"])).surfaces, ("session", "topic"))

    def test_row_without_the_profile_surface_is_never_a_profile_candidate(self):
        registry = ProviderRegistry.from_dict(
            providers_config(rows=[network_row(surfaces=["topic", "session"])])
        )
        self.assertEqual(registry.rows_for("profile"), [])
        self.assertEqual(registry.rows_for("topic"), ["jev"])

    def test_undeclared_surface_is_refused_before_any_socket(self):
        registry = ProviderRegistry.from_dict(providers_config(rows=[network_row(surfaces=["topic"])]))
        guard = SocketGuard()
        backend = registry.select_backend(transport=guard)
        with self.assertRaises(ProviderConfigError) as ctx:
            backend.batch_score(HANDLE, make_candidates(), "session")
        self.assertEqual(ctx.exception.code, "surface_unavailable")
        self.assertEqual(guard.calls, 0, "the refusal must precede the transport")
        self.assertEqual(backend.batch_calls.get("session", 0), 0)

    def test_declared_surface_still_scores_normally(self):
        registry = ProviderRegistry.from_dict(
            providers_config(rows=[network_row(surfaces=["topic", "session"])])
        )
        transport = RecordingTransport([scores_body(TOPIC_IDS)])
        backend = registry.select_backend(transport=transport)
        rows = backend.batch_score(HANDLE, make_candidates(), "topic")
        self.assertEqual([r["candidate"] for r in rows], TOPIC_IDS)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(backend.batch_calls["topic"], 1)

    def test_surface_filter_applies_to_a_fallback_chain_head(self):
        registry = ProviderRegistry.from_dict(
            providers_config(
                rows=[network_row(surfaces=["topic"]), network_row(id="backup", base_url="https://b.example.com/v1")],
                default_provider="jev",
                allow_fallback=True,
            )
        )
        with self.assertRaises(ProviderConfigError) as ctx:
            registry.select_backend(transport=SocketGuard(), surface="profile")
        self.assertEqual(ctx.exception.code, "surface_unavailable")


# --------------------------------------------------------------------------
# 5. Capability discovery: booleans and non-secret labels only (L14)
# --------------------------------------------------------------------------


class TestCapabilityDiscovery(KeyEnvTestCase):
    def test_capability_row_exposes_the_declared_labels(self):
        registry = ProviderRegistry.from_dict(
            providers_config(rows=[network_row(data_class="internal", surfaces=["topic", "session"])])
        )
        capability = registry.capabilities()[0]
        self.assertEqual(capability["id"], "jev")
        self.assertEqual(capability["kind"], "openai_compatible")
        self.assertTrue(capability["implemented"])
        self.assertTrue(capability["requires_network"])
        self.assertTrue(capability["supports_batch_scores"])
        self.assertEqual(capability["data_class"], "internal")
        self.assertEqual(capability["surfaces"], ["topic", "session"])
        self.assertIsNone(capability["transform"])
        self.assertIsNone(capability["endpoint_mode"])
        self.assertFalse(capability["profile_duty_eligible"])
        self.assertEqual(capability["api_key_env"], KEY_ENV)
        self.assertIsInstance(capability["api_key_present"], bool)
        self.assertNotIn("api_key_value", capability)

    def test_capabilities_never_contain_key_material(self):
        import hashlib

        registry = ProviderRegistry.from_dict(providers_config(rows=[network_row()]))
        blob = json.dumps(registry.capabilities(), sort_keys=True)
        self.assertNotIn(KEY_VALUE, blob)
        self.assertNotIn(KEY_VALUE[:6], blob)
        self.assertNotIn(KEY_VALUE[-6:], blob)
        self.assertNotIn(str(len(KEY_VALUE)), blob)
        self.assertNotIn(hashlib.sha256(KEY_VALUE.encode("utf-8")).hexdigest(), blob)
        self.assertNotIn("Authorization", blob)

    def test_unimplemented_family_capability_is_never_dressed_up_as_available(self):
        row = ProviderRow(id="laya", kind="laya_local", model="laya-1", transform="laya-choice-v1", data_class="private")
        capability = row.to_capability(api_key_present=False, enabled=True, disabled_reason=None)
        self.assertFalse(capability["implemented"])
        self.assertFalse(capability["supports_batch_scores"])
        self.assertEqual(capability["transform"], "laya-choice-v1")

    def test_profile_duty_eligibility_follows_the_surface_declaration(self):
        with_profile = load_row(network_row(surfaces=["topic", "session", "profile"]))
        cap = with_profile.to_capability(True, True, None)
        self.assertTrue(cap["profile_duty_eligible"])


# --------------------------------------------------------------------------
# 6. Calibration-state consistency gate (section 6.4 / L23) -- the shaka-02
#    check-G counterexample, as a regression test
# --------------------------------------------------------------------------


class TestCalibrationConsistencyGate(unittest.TestCase):
    def test_unfitted_model_is_never_active(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            CalibrationModel(model_id="none", active=True)
        self.assertEqual(ctx.exception.code, "calibration_state_invalid")

    def test_unfitted_model_is_never_active_through_the_load_path(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            CalibrationModel.from_dict({"model_id": "none", "active": True, "a_topic": 1.0})
        self.assertEqual(ctx.exception.code, "calibration_state_invalid")

    def test_fitted_model_that_is_inactive_is_refused(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            CalibrationModel(model_id="platt-v1", active=False)
        self.assertEqual(ctx.exception.code, "calibration_state_invalid")

    def test_consistent_pairs_are_accepted(self):
        self.assertFalse(CalibrationModel().active)
        self.assertEqual(CalibrationModel().model_id, "none")
        active = CalibrationModel(model_id="platt-v1", active=True)
        self.assertTrue(active.active)
        round_tripped = CalibrationModel.from_dict(
            {"model_id": "platt-v1", "active": True, "a_topic": 2.0, "b_topic": -1.0}
        )
        self.assertEqual(round_tripped.model_id, "platt-v1")

    def test_shipped_calibration_artifact_is_consistent(self):
        path = ROOT / "eval" / "calibration.json"
        with open(path, "r", encoding="utf-8") as handle:
            model = CalibrationModel.from_dict(json.load(handle))
        self.assertEqual(model.model_id, "none")
        self.assertFalse(model.active)

    def test_no_boolean_alone_unlocks_the_automatic_band(self):
        result = route(
            make_request(),
            make_candidates(),
            RouterConfig(calibration=CalibrationModel(), backend=SyntheticBackend()),
            include_session=True,
        )
        self.assertNotEqual(result.decision.band, "automatic")
        self.assertEqual(result.receipt["calibration_model_id"], "none")


class TestCalibrationKeyBinding(unittest.TestCase):
    def test_active_calibration_must_declare_its_fitted_key(self):
        model = CalibrationModel(model_id="platt-v1", active=True)
        with self.assertRaises(ProviderConfigError) as ctx:
            model.check_provider_key(load_row(network_row()))
        self.assertEqual(ctx.exception.code, "calibration_key_mismatch")

    def test_key_mismatch_is_refused(self):
        model = CalibrationModel(
            model_id="platt-v1", active=True, kind="openai_compatible", model="other-model-9"
        )
        with self.assertRaises(ProviderConfigError) as ctx:
            model.check_provider_key(load_row(network_row()))
        self.assertEqual(ctx.exception.code, "calibration_key_mismatch")

    def test_matching_key_is_accepted(self):
        model = CalibrationModel(
            model_id="platt-v1",
            active=True,
            kind="openai_compatible",
            model="family-model-1",
            transform=None,
        )
        model.check_provider_key(load_row(network_row()))  # must not raise

    def test_inactive_calibration_is_never_key_checked(self):
        CalibrationModel().check_provider_key(load_row(network_row()))
        CalibrationModel().check_provider_key({"kind": "anything", "model": "else"})

    def test_local_backend_key_is_the_deterministic_scorer(self):
        self.assertEqual(local_backend_key(), {"kind": "local", "model": "local-deterministic", "transform": None})
        bound = CalibrationModel(
            model_id="platt-v1", active=True, kind="local", model="local-deterministic"
        )
        bound.check_provider_key(local_backend_key())  # must not raise
        with self.assertRaises(ProviderConfigError) as ctx:
            CalibrationModel(
                model_id="platt-v1", active=True, kind="openai_compatible", model="hosted-1"
            ).check_provider_key(local_backend_key())
        self.assertEqual(ctx.exception.code, "calibration_key_mismatch")


# --------------------------------------------------------------------------
# 7. CLI wiring: the gates fire before anything is routed or written
# --------------------------------------------------------------------------


class TestCliGates(unittest.TestCase):
    def _write_inputs(self, tmp: str) -> Dict[str, str]:
        request_path = os.path.join(tmp, "request.json")
        candidates_path = os.path.join(tmp, "candidates.json")
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump(make_request().to_dict(), handle)
        with open(candidates_path, "w", encoding="utf-8") as handle:
            json.dump(make_candidates().to_dict(), handle)
        return {"request": request_path, "candidates": candidates_path}

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "ordaprompt_router.cli"] + args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_cli_refuses_an_active_calibration_bound_to_another_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp)
            calibration = os.path.join(tmp, "calibration.json")
            with open(calibration, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model_id": "platt-v1",
                        "active": True,
                        "kind": "openai_compatible",
                        "model": "hosted-1",
                    },
                    handle,
                )
            receipts = os.path.join(tmp, "receipts")
            proc = self._run(
                [
                    "route-session",
                    "--request",
                    paths["request"],
                    "--candidates",
                    paths["candidates"],
                    "--calibration",
                    calibration,
                    "--receipts-dir",
                    receipts,
                ]
            )
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("calibration_key_mismatch", proc.stderr)
            self.assertFalse(os.path.exists(receipts) and os.listdir(receipts))

    def test_cli_refuses_an_inconsistent_calibration_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp)
            calibration = os.path.join(tmp, "calibration.json")
            with open(calibration, "w", encoding="utf-8") as handle:
                json.dump({"model_id": "none", "active": True}, handle)
            proc = self._run(
                [
                    "classify",
                    "--request",
                    paths["request"],
                    "--candidates",
                    paths["candidates"],
                    "--calibration",
                    calibration,
                ]
            )
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("calibration_state_invalid", proc.stderr)

    def test_cli_accepts_a_calibration_bound_to_the_local_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp)
            calibration = os.path.join(tmp, "calibration.json")
            with open(calibration, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model_id": "platt-v1",
                        "active": True,
                        "kind": "local",
                        "model": "local-deterministic",
                        "a_topic": 1.7773,
                        "b_topic": -1.8304,
                        "a_session": 1.8240,
                        "b_session": -0.6516,
                    },
                    handle,
                )
            proc = self._run(
                [
                    "classify",
                    "--request",
                    paths["request"],
                    "--candidates",
                    paths["candidates"],
                    "--calibration",
                    calibration,
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["routing_receipt"]["calibration_model_id"], "platt-v1")

    def test_cli_help_documents_the_data_class_flag(self):
        proc = self._run(["classify", "--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--data-class", proc.stdout)
        self.assertIn("--providers", proc.stdout)

    def test_cli_refuses_a_providers_row_for_an_unimplemented_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp)
            providers = os.path.join(tmp, "providers.json")
            with open(providers, "w", encoding="utf-8") as handle:
                json.dump(
                    providers_config(
                        rows=[
                            {
                                "id": "laya",
                                "kind": "laya_local",
                                "model": "laya-1",
                                "transform": "laya-choice-v1",
                                "endpoint_mode": "in_process",
                            }
                        ]
                    ),
                    handle,
                )
            proc = self._run(
                [
                    "classify",
                    "--request",
                    paths["request"],
                    "--candidates",
                    paths["candidates"],
                    "--providers",
                    providers,
                ]
            )
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("kind_unimplemented", proc.stderr)

    def test_cli_refuses_a_private_request_with_only_public_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp)
            providers = os.path.join(tmp, "providers.json")
            with open(providers, "w", encoding="utf-8") as handle:
                json.dump(providers_config(rows=[network_row()]), handle)
            os.environ[KEY_ENV] = KEY_VALUE
            try:
                proc = self._run(
                    [
                        "classify",
                        "--request",
                        paths["request"],
                        "--candidates",
                        paths["candidates"],
                        "--providers",
                        providers,
                        "--data-class",
                        "private",
                    ]
                )
            finally:
                os.environ.pop(KEY_ENV, None)
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("data_class_refused", proc.stderr)


# --------------------------------------------------------------------------
# 8. Documentation wording (L7 / L22): Jev-STYLE, never a live Jev integration
# --------------------------------------------------------------------------


class TestReadmeWording(unittest.TestCase):
    def setUp(self) -> None:
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_readme_says_jev_style_architecture(self):
        self.assertIn("Jev-style", self.readme)

    def test_readme_states_the_unimplemented_bindings(self):
        lowered = self.readme.lower()
        self.assertIn("simple-jev", lowered)
        self.assertIn("not implemented", lowered)
        self.assertIn("typesafe", lowered)
        self.assertIn("laya", lowered)
        self.assertIn("profiles.json", lowered)

    def test_readme_never_claims_a_bundled_or_live_jev_model(self):
        lowered = self.readme.lower()
        for forbidden in (
            "bundles a jev model",
            "ships a jev model",
            "calls a jev model",
            "uses a jev model",
            "tested against live jev",
            "verified against live jev",
        ):
            self.assertNotIn(forbidden, lowered)
        # ... and the disclaimer is stated in words, not merely implied
        self.assertIn("no live jev, simple-jev, nanojev or laya integration was tested", lowered)

    def test_readme_documents_the_five_families_and_the_two_implemented_kinds(self):
        for family in PROVIDER_KINDS:
            self.assertIn(family, self.readme)
        for field in FAMILY_ROW_FIELDS:
            self.assertIn(field, self.readme)
        for data_class in ROW_DATA_CLASSES:
            self.assertIn(data_class, self.readme)


if __name__ == "__main__":
    unittest.main()
