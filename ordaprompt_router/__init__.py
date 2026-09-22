"""OrdaPilot classifier router -- bounded slice of the LOCKED leo-arch.md spec.

Package layout
    schemas.py   closed JSON schemas (section 1)
    adapter.py   ClassificationBackend + SyntheticBackend + gated OpenRouter (section 2)
    router.py    batch comparison, three-band policy, session utility, labels (sections 3, 5, 6)
    receipts.py  append-only, hashes-only receipt store (sections 1.4, 4)
    cli.py       `python3 -m ordaprompt_router.cli classify|route-session`

Stdlib only.  No network.  No Hermes core or profile-config changes.
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
    CandidateSet,
    ClassificationRequest,
    PrivacyViolationError,
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
    "OpenRouterBackend",
    "PrivacyViolationError",
    "ReceiptStore",
    "RequestHandle",
    "RouteResult",
    "RouterConfig",
    "RoutingDecision",
    "SchemaError",
    "SyntheticBackend",
    "Thresholds",
    "assert_no_free_text",
    "canonicalize_redacted",
    "route",
    "sha256_hex",
]

__version__ = "1.0.0"
