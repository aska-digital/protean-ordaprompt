"""OrdaPilot classifier router -- bounded slice of the LOCKED leo-arch.md spec.

Package layout
    schemas.py   closed JSON schemas (section 1)
    adapter.py   ClassificationBackend + SyntheticBackend + gated OpenRouter (section 2)
    router.py    batch comparison, three-band policy, session utility, labels (sections 3, 5, 6)
    receipts.py  append-only, hashes-only receipt store (sections 1.4, 4)
    providers.py operator-supplied provider registry + OpenAI-compatible adapter
    cli.py       `python3 -m ordaprompt_router.cli classify|route-session`

Stdlib only.  No network unless an operator supplies a providers file with
`--providers` and selects a provider row that requires network.  No Hermes core or
profile-config changes.
"""

from __future__ import annotations

from .adapter import (
    BackendError,
    ClassificationBackend,
    DisabledByPolicy,
    OpenRouterBackend,
    RequestHandle,
    SyntheticBackend,
    canonicalize_redacted,
    sha256_hex,
)
from .providers import (
    OpenAICompatibleBackend,
    ProfileMetadataViolation,
    ProviderChainBackend,
    ProviderConfigError,
    ProviderContractError,
    ProviderError,
    ProviderRegistry,
    ProviderRow,
    ProviderTransportError,
    Transport,
    TransportRequest,
    TransportResponse,
    assert_profile_payload_whitelisted,
    stdlib_transport,
)
from .receipts import ReceiptStore, assert_no_free_text
from .router import (
    CalibrationModel,
    LabelRegistry,
    RouterConfig,
    RouteResult,
    Thresholds,
    route,
)
from .schemas import (
    PROFILE_SENTINELS,
    SURFACES,
    CandidateSet,
    ClassificationRequest,
    PrivacyViolationError,
    ProfileCandidate,
    RoutingDecision,
    SchemaError,
)

__all__ = [
    "BackendError",
    "CalibrationModel",
    "CandidateSet",
    "ClassificationBackend",
    "ClassificationRequest",
    "DisabledByPolicy",
    "LabelRegistry",
    "OpenAICompatibleBackend",
    "OpenRouterBackend",
    "PrivacyViolationError",
    "ProfileCandidate",
    "ProfileMetadataViolation",
    "PROFILE_SENTINELS",
    "SURFACES",
    "ProviderChainBackend",
    "ProviderConfigError",
    "ProviderContractError",
    "ProviderError",
    "ProviderRegistry",
    "ProviderRow",
    "ProviderTransportError",
    "ReceiptStore",
    "RequestHandle",
    "RouteResult",
    "RouterConfig",
    "RoutingDecision",
    "SchemaError",
    "SyntheticBackend",
    "Thresholds",
    "Transport",
    "TransportRequest",
    "TransportResponse",
    "assert_no_free_text",
    "assert_profile_payload_whitelisted",
    "canonicalize_redacted",
    "route",
    "sha256_hex",
    "stdlib_transport",
]

__version__ = "1.1.0"
