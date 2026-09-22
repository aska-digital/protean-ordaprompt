"""Operator-supplied provider registry and the OpenAI-compatible classifier adapter.

Bounded slice of the provider lane (ordaprompt-providers-20260922).  The locked
boundary implemented here, decision by decision:

D1  Credentials are read ONLY from the environment variable named by a row's
    ``api_key_env``.  No key value, prefix, length or header echo is ever written
    to a file, log, receipt, error message or stdout.  ``plugin.yaml`` keeps
    ``config_schema: {}``.
D2  One operator-supplied JSON file, passed explicitly as ``--providers <path>``.
    With no flag this module is not imported into the decision path at all and the
    router behaves exactly as before (local deterministic adapter only).
D3  Fail closed.  Unknown provider id, duplicate id, unset ``api_key_env`` value,
    unparsable JSON, a reply that does not cover exactly the candidate id set, and
    every reply/score contract violation raise a structured error
    (:class:`ProviderError` with a machine ``code``), and the router abstains.
    There is NO silent fallback: the only fallback is the operator-declared ordered
    chain (``allow_fallback: true`` + ``default_provider``), and every fallback that
    fires is recorded (``ProviderChainBackend.last_fallback_from``), surfaced by the
    router as the receipt's optional ``fallback_from`` field.
D4  SSRF boundary: the transport is stdlib ``urllib.request`` with redirects
    DISABLED; the request URL is built from the row's ``base_url`` only (never from
    model output, never from an environment variable); the scheme must be ``https``
    -- or ``http`` only when the row sets ``allow_private_network: true`` AND the
    host is loopback/private.  Anything else is refused before a socket is opened.
D5  The transport is injectable (``transport=``), so tests never open a socket.
D6  Provider raw scores pass through the calibration/threshold/band path unchanged.
    No provider-specific thresholds exist here.  Missing / extra / duplicate /
    unknown candidate id, non-numeric, bool-as-number, NaN/inf and out-of-[0, 1]
    scores are each a hard error.
D7  "Jev-style" only: a ``jev`` entry is a plain OpenAI-compatible endpoint row.
    This module bundles no model and makes no claim about a live Jev integration.
D8  The ``profile`` surface.  It is a first-class comparative surface beside ``topic``
    and ``session`` and sits behind exactly the same contract: ONE ``batch_score`` call
    per surface, all candidates at once in the caller's order, reply rows of exactly
    ``{"candidate","raw"}``, and a reply that does not cover exactly the candidate id
    set is a hard :class:`ProviderContractError` (never a partial accept, never a
    silent fallback).  Eligibility is computed UPSTREAM and arrives as the candidate
    set -- this module never widens, reorders, adds or drops a candidate.  The sentinel
    ``no_suitable_profile`` is always a candidate, so the explicit abstain outcome is
    always on the ballot.  Privacy: only ids/labels/enums leave the process; the single
    choke point :func:`_profile_request_rows` plus
    :func:`assert_profile_payload_whitelisted` guarantee that nothing beyond
    ``profile_id``, ``scope``, ``privacy_class`` and ``labels`` is ever serialised.

Stdlib only.  No socket is opened unless an operator supplies a providers file and
selects a provider row whose kind requires network.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from .adapter import (
    BackendError,
    ClassificationBackend,
    RequestHandle,
    SyntheticBackend,
    assert_hash_only_payload,
)
from .schemas import (
    ID_TOKEN_RE,
    PROFILE_PRIVACY_CLASSES,
    PROFILE_SCOPES,
    PROFILE_SENTINEL_ABSTAIN,
    SLUG_RE,
    SURFACES,
    CandidateSet,
    PrivacyViolationError,
)

# --------------------------------------------------------------------------
# Constants / closed domains
# --------------------------------------------------------------------------

PROVIDERS_SCHEMA_VERSION = 1

#: The CLOSED provider-family taxonomy (leo-adapter-architecture.md section 2.1).  `kind`
#: is a closed enum: a family is added by contract revision, never by configuration, and an
#: unknown token is a hard `kind_unknown` error.
PROVIDER_KINDS = (
    "local",
    "openai_compatible",
    "jev_decision",
    "nanojev_batch",
    "laya_local",
)

#: Families with a real adapter at THIS head.  The remaining three are SHAPED (their wire
#: mapping is designed and documented) but UNIMPLEMENTED: a row declaring one fails closed at
#: LOAD time with `kind_unimplemented` -- never a silent downgrade to the local scorer, never
#: a silent omission.
IMPLEMENTED_KINDS = ("local", "openai_compatible")
DESIGNED_UNIMPLEMENTED_KINDS = ("jev_decision", "nanojev_batch", "laya_local")

#: Per-family hard cap on `batch_max_candidates` (section 4 / L25).  A row above its
#: family's cap is `field_invalid` before any call.  The `jev_decision` cap is the tightest
#: VERIFIED row limit in that family (Simple-Jev `choice`, 2-50); the implemented kinds keep
#: the architecture's 256-candidate ceiling.
FAMILY_BATCH_LIMITS = {
    "local": 256,
    "openai_compatible": 256,
    "jev_decision": 50,
    "nanojev_batch": 255,
    "laya_local": 20,
}

#: Closed transform labels (section 8.1).  A transform names the per-row request/response
#: shape used when it differs from `/chat/completions`; there is no runtime guessing between
#: spellings.
TRANSFORMS = ("sj-choice-batch-v1", "laya-choice-v1")
#: Families whose request shape is NOT `/chat/completions`: they MUST declare a transform.
TRANSFORM_REQUIRED_KINDS = ("jev_decision", "laya_local")
ENDPOINT_MODES = ("in_process", "loopback_sidecar")

#: Provider-row data classes (section 2.3), most-public first.  A request may use only a row
#: whose class is AT LEAST as tight as the request's own class (section 3.4 / L21).
ROW_DATA_CLASSES = ("public", "internal", "private")
#: Default data class of a request when the operator does not declare one (the CLI
#: ``--data-class`` flag).  Most-public is the pre-existing behaviour: no row is filtered out
#: unless the operator says the payload is tighter than that.
DEFAULT_REQUEST_DATA_CLASS = "public"
#: Ranking used for that comparison.  `restricted` is a profile-candidate class, not a row
#: class: no provider row can ever serve a `restricted` request (fail closed).
DATA_CLASS_RANK = {"public": 0, "internal": 1, "private": 2, "restricted": 3}

CONFIG_FIELDS = ("schema_version", "providers", "default_provider", "allow_fallback")
ROW_FIELDS = (
    "id",
    "kind",
    "base_url",
    "model",
    "api_key_env",
    "transform",
    "endpoint_mode",
    "data_class",
    "surfaces",
    "allow_private_network",
    "timeout_s",
    "max_retries",
    "max_tokens",
    "batch_max_candidates",
    "supports_taxonomy_proposal",
)

PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
#: environment variable NAME (never a value): uppercase, digits, underscore
API_KEY_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,79}$")

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_MAX_CANDIDATES = 64
MAX_TIMEOUT_S = 120.0
MAX_RETRIES_CAP = 5
MAX_TOKENS_CAP = 8192
BATCH_CANDIDATES_CAP = 256
MIN_CONFIDENCE_TAXONOMY = 0.90

#: the ONLY failure codes that may trigger the operator-declared fallback chain.
#: Availability failures fall back; contract, credential and configuration failures
#: ALWAYS abstain (never papered over by another provider).
FALLBACK_TRIGGER_CODES = ("transport_failed", "http_status")

DISABLED_REASON_KEY_UNSET = "api_key_env_unset"

#: The built-in local adapter's model id.  Without `--providers` the running backend is the
#: deterministic local scorer, and an ACTIVE calibration must still name this exact key to
#: apply to it (L16/L23) -- a calibration fitted for a hosted model never silently governs a
#: local-only run.
LOCAL_MODEL_ID = "local-deterministic"


def local_backend_key() -> Dict[str, Any]:
    """The ``(kind, model, transform)`` key of the built-in local backend."""
    return {"kind": "local", "model": LOCAL_MODEL_ID, "transform": None}

#: payload content tokens (closed domain, checked by assert_hash_only_payload)
BATCH_CONTENT_TOKEN = "batch-score"
BATCH_SYSTEM_TOKEN = "ordaprompt-batch-score-v1"
PROPOSAL_CONTENT_TOKEN = "propose-taxonomy"
PROPOSAL_SYSTEM_TOKEN = "ordaprompt-propose-taxonomy-v1"
NO_CONTEXT_TOKEN = "none"


# --------------------------------------------------------------------------
# Structured errors (machine codes only; never credential material)
# --------------------------------------------------------------------------


class ProviderError(BackendError):
    """Structured provider failure with a machine ``code`` (D3).

    Messages carry the code and the non-secret provider id only.  Key values,
    prefixes, lengths and request headers are never formatted into a message.
    """

    def __init__(self, code: str, message: str, provider_id: Optional[str] = None) -> None:
        self.code = str(code)
        self.provider_id = provider_id
        text = "provider_error[%s]" % self.code
        if provider_id:
            text += " provider=%s" % provider_id
        super().__init__("%s: %s" % (text, message))


class ProviderConfigError(ProviderError):
    """Configuration, credential or selection failure.  Never falls back."""


class ProviderTransportError(ProviderError):
    """Availability failure (connection, timeout, HTTP status).  Fallback-eligible."""


class ProviderContractError(ProviderError):
    """Reply / score contract violation.  Never falls back: a provider that returns
    untrustworthy numbers must not be papered over by the next chain member."""


def is_fallback_trigger(exc: BaseException) -> bool:
    """True only for availability failures (see ``FALLBACK_TRIGGER_CODES``)."""
    return isinstance(exc, ProviderError) and exc.code in FALLBACK_TRIGGER_CODES


# --------------------------------------------------------------------------
# Transport seam (D4 / D5)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportRequest:
    method: str
    url: str
    headers: Dict[str, str]
    body: bytes
    timeout_s: float


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


Transport = Callable[[TransportRequest], TransportResponse]


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect (D4).  Returning None makes urllib raise HTTPError."""

    def redirect_request(  # type: ignore[override]
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def stdlib_transport(request: TransportRequest) -> TransportResponse:
    """The only network implementation in this package: stdlib, redirects off.

    A redirect surfaces as its 3xx status (never followed); any other HTTP status is
    returned verbatim; connection-level failures raise ``OSError``/``URLError`` which
    the caller maps to a structured transport error.  Response bodies are returned
    to the caller but never echoed into error messages or logs.
    """
    opener = urllib.request.build_opener(NoRedirectHandler())
    http_request = urllib.request.Request(
        request.url,
        data=request.body,
        headers=dict(request.headers),
        method=request.method,
    )
    try:
        with opener.open(http_request, timeout=request.timeout_s) as response:
            return TransportResponse(
                status=int(getattr(response, "status", 0) or 0),
                headers={str(k): str(v) for k, v in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return TransportResponse(
            status=int(exc.code),
            headers={str(k): str(v) for k, v in (exc.headers.items() if exc.headers else [])},
            body=exc.read() or b"",
        )


# --------------------------------------------------------------------------
# URL / host policy (D4)
# --------------------------------------------------------------------------


def is_loopback_or_private_host(host: str) -> bool:
    """True for loopback / private / link-local literals and ``localhost`` (D4)."""
    if not host:
        return False
    name = host.strip().strip("[]").lower()
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    )


def _validate_base_url(url: Any, allow_private_network: bool, path: str) -> str:
    """Fail-closed URL policy for a provider row (D4).  No socket is opened here."""
    if not isinstance(url, str) or not url.strip():
        raise ProviderConfigError("base_url_invalid", "%s must be a non-empty string" % path)
    text = url.strip()
    if any(ch in text for ch in "?#"):
        raise ProviderConfigError(
            "base_url_invalid", "%s must not carry a query string or fragment" % path
        )
    parts = urllib.parse.urlsplit(text)
    scheme = (parts.scheme or "").lower()
    host = parts.hostname or ""
    if scheme not in ("http", "https"):
        raise ProviderConfigError(
            "url_scheme_refused", "%s scheme %r is refused (https required)" % (path, scheme)
        )
    if parts.username or parts.password:
        raise ProviderConfigError(
            "url_userinfo_refused", "%s must not embed credentials in the URL" % path
        )
    if not host:
        raise ProviderConfigError("base_url_invalid", "%s must name a host" % path)
    if scheme == "http":
        if not allow_private_network:
            raise ProviderConfigError(
                "url_scheme_refused",
                "%s: plain http requires allow_private_network=true" % path,
            )
        if not is_loopback_or_private_host(host):
            raise ProviderConfigError(
                "private_network_not_allowed",
                "%s: http is permitted only for loopback/private hosts" % path,
            )
    return text


# --------------------------------------------------------------------------
# Row model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRow:
    """One ``providers.json`` row.  Holds the API key env var NAME, never a value.

    The row is the ONLY place a family binding lives: ``kind`` selects the family,
    ``transform``/``endpoint_mode`` select the request shape, ``data_class`` and ``surfaces``
    declare which requests the row may ever be offered (section 2.3 / L21).
    """

    id: str
    kind: str
    base_url: Optional[str] = None
    model: str = ""
    api_key_env: Optional[str] = None
    transform: Optional[str] = None
    endpoint_mode: Optional[str] = None
    data_class: str = "public"
    surfaces: Tuple[str, ...] = SURFACES
    allow_private_network: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_max_candidates: int = DEFAULT_BATCH_MAX_CANDIDATES
    supports_taxonomy_proposal: bool = False

    @property
    def implemented(self) -> bool:
        """True when this head ships a real adapter for the row's family."""
        return self.kind in IMPLEMENTED_KINDS

    @property
    def is_in_process(self) -> bool:
        """True when the row runs inside this process (no socket at all)."""
        return self.kind == "local" or (
            self.kind == "laya_local" and self.endpoint_mode == "in_process"
        )

    @property
    def requires_network(self) -> bool:
        return self.kind == "openai_compatible" or (
            self.kind == "laya_local" and self.endpoint_mode == "loopback_sidecar"
        )

    @property
    def batch_limit(self) -> int:
        return FAMILY_BATCH_LIMITS.get(self.kind, BATCH_CANDIDATES_CAP)

    def supports_surface(self, surface: str) -> bool:
        """A row that does not list a surface is NEVER a candidate for it (section 2.3)."""
        return str(surface) in self.surfaces

    def class_allows(self, request_class: str) -> bool:
        """True when a request of ``request_class`` may be offered to this row (L21).

        The comparison is fail closed both ways: an unknown request class ranks above every
        row class (no row may see it) and an unknown row class cannot be constructed.
        """
        request_rank = DATA_CLASS_RANK.get(str(request_class), 99)
        return request_rank <= DATA_CLASS_RANK.get(self.data_class, -1)

    def chat_completions_url(self) -> str:
        return (self.base_url or "").rstrip("/") + "/chat/completions"

    def to_capability(self, api_key_present: bool, enabled: bool, disabled_reason: Optional[str]) -> Dict[str, Any]:
        """Booleans + non-secret labels only (D1): never a key value or length.

        Everything here is a declared, non-secret label: the env var NAME is not secret
        material, its VALUE and any derived statistic (prefix, length, digest, entropy) is.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "implemented": self.implemented,
            "requires_network": self.requires_network,
            "supports_batch_scores": self.implemented,
            "supports_taxonomy_proposal": bool(self.supports_taxonomy_proposal),
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(api_key_present),
            "enabled": bool(enabled),
            "disabled_reason": disabled_reason,
            "surfaces": list(self.surfaces),
            "data_class": self.data_class,
            "transform": self.transform,
            "endpoint_mode": self.endpoint_mode,
            "profile_duty_eligible": bool(
                enabled and self.implemented and self.supports_surface("profile")
            ),
        }


# --------------------------------------------------------------------------
# Field validation helpers (closed schema, fail closed)
# --------------------------------------------------------------------------


def _need(condition: bool, code: str, message: str, provider_id: Optional[str] = None) -> None:
    if not condition:
        raise ProviderConfigError(code, message, provider_id=provider_id)


def _closed(obj: Any, allowed: Sequence[str], path: str, provider_id: Optional[str] = None) -> Mapping[str, Any]:
    _need(isinstance(obj, Mapping), "schema_invalid", "%s must be an object" % path, provider_id)
    unknown = sorted(set(obj.keys()) - set(allowed))
    _need(
        not unknown,
        "unknown_field",
        "%s: unknown field(s) rejected by the closed provider schema: %s" % (path, unknown),
        provider_id,
    )
    return obj


def _str_field(
    obj: Mapping[str, Any],
    key: str,
    path: str,
    pattern: Optional[re.Pattern] = None,
    required: bool = True,
    provider_id: Optional[str] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    value = obj.get(key)
    if value is None:
        if required:
            raise ProviderConfigError("missing_field", "%s.%s is required" % (path, key), provider_id)
        return default
    _need(
        isinstance(value, str) and value != "",
        "field_invalid",
        "%s.%s must be a non-empty string" % (path, key),
        provider_id,
    )
    if pattern is not None:
        _need(
            bool(pattern.match(value)),
            "field_invalid",
            "%s.%s value %r is outside the closed domain %s" % (path, key, value, pattern.pattern),
            provider_id,
        )
    return value


def _bool_field(
    obj: Mapping[str, Any], key: str, path: str, provider_id: Optional[str], default: bool = False
) -> bool:
    value = obj.get(key, default)
    _need(
        isinstance(value, bool),
        "field_invalid",
        "%s.%s must be a boolean" % (path, key),
        provider_id,
    )
    return bool(value)


def _int_field(
    obj: Mapping[str, Any],
    key: str,
    path: str,
    provider_id: Optional[str],
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = obj.get(key, default)
    _need(
        isinstance(value, int) and not isinstance(value, bool),
        "field_invalid",
        "%s.%s must be an integer" % (path, key),
        provider_id,
    )
    _need(
        minimum <= int(value) <= maximum,
        "field_invalid",
        "%s.%s must satisfy %d <= value <= %d" % (path, key, minimum, maximum),
        provider_id,
    )
    return int(value)


def _float_field(
    obj: Mapping[str, Any],
    key: str,
    path: str,
    provider_id: Optional[str],
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = obj.get(key, default)
    _need(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "field_invalid",
        "%s.%s must be a number" % (path, key),
        provider_id,
    )
    number = float(value)
    _need(
        math.isfinite(number) and minimum <= number <= maximum,
        "field_invalid",
        "%s.%s must satisfy %s <= value <= %s" % (path, key, minimum, maximum),
        provider_id,
    )
    return number


def _enum_list_field(
    obj: Mapping[str, Any],
    key: str,
    path: str,
    provider_id: Optional[str],
    allowed: Sequence[str],
    default: Sequence[str],
) -> Tuple[str, ...]:
    """A closed, non-empty, duplicate-free list of declared labels (order preserved)."""
    value = obj.get(key)
    if value is None:
        return tuple(default)
    _need(
        isinstance(value, list) and len(value) > 0 and all(isinstance(v, str) for v in value),
        "field_invalid",
        "%s.%s must be a non-empty list of strings" % (path, key),
        provider_id,
    )
    items = [str(v) for v in value]
    unknown = sorted({v for v in items if v not in allowed})
    _need(
        not unknown,
        "field_invalid",
        "%s.%s value(s) %s are outside the closed domain %s" % (path, key, unknown, list(allowed)),
        provider_id,
    )
    _need(
        len(set(items)) == len(items),
        "field_invalid",
        "%s.%s must not repeat a value" % (path, key),
        provider_id,
    )
    return tuple(items)


def parse_provider_row(obj: Any, index: int) -> ProviderRow:
    """Validate one row of the closed ``providers`` schema (D3).

    Order of the checks is deliberate: token domains first (``kind_unknown``), then the
    family field rules, then the fail-closed refusal of a designed-but-unimplemented family
    (``kind_unimplemented``), and only then the per-kind endpoint/key rules.  Nothing here
    opens a socket, reads an environment value, or coerces an out-of-domain token.
    """
    path = "providers[%d]" % index
    raw = _closed(obj, ROW_FIELDS, path)
    provider_id = cast(str, _str_field(raw, "id", path, PROVIDER_ID_RE))
    path = "providers[%d](%s)" % (index, provider_id)
    kind = cast(str, _str_field(raw, "kind", path, provider_id=provider_id))
    _need(
        kind in PROVIDER_KINDS,
        "kind_unknown",
        "%s.kind %r is unknown (expected one of %s)" % (path, kind, list(PROVIDER_KINDS)),
        provider_id,
    )

    allow_private_network = _bool_field(raw, "allow_private_network", path, provider_id)
    timeout_s = _float_field(raw, "timeout_s", path, provider_id, DEFAULT_TIMEOUT_S, 0.1, MAX_TIMEOUT_S)
    max_retries = _int_field(raw, "max_retries", path, provider_id, DEFAULT_MAX_RETRIES, 0, MAX_RETRIES_CAP)
    max_tokens = _int_field(raw, "max_tokens", path, provider_id, DEFAULT_MAX_TOKENS, 1, MAX_TOKENS_CAP)
    #: the family cap is known before the field default, so an omitted `batch_max_candidates`
    #: never defaults ABOVE the family's verified limit (it would be an instant field_invalid)
    family_limit = FAMILY_BATCH_LIMITS[kind]
    batch_max_candidates = _int_field(
        raw,
        "batch_max_candidates",
        path,
        provider_id,
        min(DEFAULT_BATCH_MAX_CANDIDATES, family_limit),
        1,
        BATCH_CANDIDATES_CAP,
    )
    supports_taxonomy_proposal = _bool_field(raw, "supports_taxonomy_proposal", path, provider_id)

    # -- family field rules (section 8.1): a gap is a hard error, never a coercion --------
    transform = _str_field(raw, "transform", path, required=False, provider_id=provider_id)
    if transform is not None:
        _need(
            transform in TRANSFORMS,
            "field_invalid",
            "%s.transform %r is not one of the declared transforms %s"
            % (path, transform, list(TRANSFORMS)),
            provider_id,
        )
    endpoint_mode = _str_field(raw, "endpoint_mode", path, required=False, provider_id=provider_id)
    if endpoint_mode is not None:
        _need(
            endpoint_mode in ENDPOINT_MODES,
            "field_invalid",
            "%s.endpoint_mode %r is not one of %s" % (path, endpoint_mode, list(ENDPOINT_MODES)),
            provider_id,
        )
    declared_class = _str_field(raw, "data_class", path, required=False, provider_id=provider_id)
    if declared_class is not None:
        _need(
            declared_class in ROW_DATA_CLASSES,
            "field_invalid",
            "%s.data_class %r is not one of %s" % (path, declared_class, list(ROW_DATA_CLASSES)),
            provider_id,
        )
    surfaces = _enum_list_field(raw, "surfaces", path, provider_id, SURFACES, SURFACES)

    if kind in TRANSFORM_REQUIRED_KINDS:
        _need(
            transform is not None,
            "transform_required",
            "%s: kind %r does not speak /chat/completions and must declare a transform "
            "(one of %s)" % (path, kind, list(TRANSFORMS)),
            provider_id,
        )
    else:
        _need(
            transform is None,
            "field_invalid",
            "%s: kind %r speaks the /chat/completions shape; transform is not allowed on it"
            % (path, kind),
            provider_id,
        )
    if kind == "laya_local":
        _need(
            endpoint_mode is not None,
            "field_invalid",
            "%s: kind 'laya_local' must declare endpoint_mode (in_process | loopback_sidecar)"
            % path,
            provider_id,
        )
    else:
        _need(
            endpoint_mode is None,
            "field_invalid",
            "%s: endpoint_mode is only meaningful for the laya_local family" % path,
            provider_id,
        )

    _need(
        batch_max_candidates <= family_limit,
        "field_invalid",
        "%s.batch_max_candidates %d exceeds the %r family limit of %d (L25: an oversized batch "
        "is refused at load, never split or truncated)"
        % (path, batch_max_candidates, kind, family_limit),
        provider_id,
    )
    # -- designed-but-unimplemented families fail closed at LOAD (section 2.1) ------------
    _need(
        kind in IMPLEMENTED_KINDS,
        "kind_unimplemented",
        "%s: family %r is designed but NOT implemented at this head (implemented: %s); "
        "the row is refused, never downgraded to the local scorer"
        % (path, kind, list(IMPLEMENTED_KINDS)),
        provider_id,
    )

    # -- data class default + the in-process-only rule for `private` (section 2.3) --------
    in_process = kind == "local" or (kind == "laya_local" and endpoint_mode == "in_process")
    if declared_class is None:
        data_class = "private" if in_process else "public"
    else:
        if declared_class == "private":
            _need(
                in_process,
                "field_invalid",
                "%s: data_class 'private' is reserved for in-process rows; a network row may be "
                "at most 'internal'" % path,
                provider_id,
            )
        data_class = declared_class

    if kind == "local":
        base_url = _str_field(raw, "base_url", path, required=False, provider_id=provider_id)
        _need(
            base_url is None,
            "field_invalid",
            "%s: a local row must not declare base_url (no network)" % path,
            provider_id,
        )
        _need(
            not supports_taxonomy_proposal,
            "taxonomy_unsupported_for_kind",
            "%s: the local deterministic adapter never proposes taxonomy" % path,
            provider_id,
        )
        model = _str_field(raw, "model", path, MODEL_ID_RE, required=False, provider_id=provider_id,
                           default=LOCAL_MODEL_ID) or LOCAL_MODEL_ID
        api_key_env = _str_field(raw, "api_key_env", path, API_KEY_ENV_RE, required=False,
                                 provider_id=provider_id)
        return ProviderRow(
            id=provider_id,
            kind=kind,
            base_url=None,
            model=model,
            api_key_env=api_key_env,
            transform=None,
            endpoint_mode=None,
            data_class=data_class,
            surfaces=surfaces,
            allow_private_network=allow_private_network,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_tokens=max_tokens,
            batch_max_candidates=batch_max_candidates,
            supports_taxonomy_proposal=False,
        )

    base_url = _validate_base_url(
        cast(str, _str_field(raw, "base_url", path, provider_id=provider_id)),
        allow_private_network,
        path + ".base_url",
    )
    model = cast(str, _str_field(raw, "model", path, MODEL_ID_RE, provider_id=provider_id))
    api_key_env = cast(str, _str_field(raw, "api_key_env", path, API_KEY_ENV_RE, provider_id=provider_id))
    return ProviderRow(
        id=provider_id,
        kind=kind,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        transform=transform,
        endpoint_mode=endpoint_mode,
        data_class=data_class,
        surfaces=surfaces,
        allow_private_network=allow_private_network,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        batch_max_candidates=batch_max_candidates,
        supports_taxonomy_proposal=supports_taxonomy_proposal,
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class ProviderRegistry:
    """Validated provider table with a fail-closed selector (D2/D3/D4).

    The declared order of ``providers`` IS the ordered fallback chain; the declared
    ``default_provider`` is tried first.  Validation happens once, at load time.
    """

    def __init__(
        self,
        rows: Sequence[ProviderRow],
        default_provider: Optional[str],
        allow_fallback: bool,
    ) -> None:
        self.rows: Tuple[ProviderRow, ...] = tuple(rows)
        self.order: List[str] = [row.id for row in self.rows]
        self.default_provider = default_provider
        self.allow_fallback = bool(allow_fallback)
        self._by_id: Dict[str, ProviderRow] = {row.id: row for row in self.rows}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, obj: Any) -> "ProviderRegistry":
        """Validate a providers document.  Raises ``ProviderConfigError`` (D3)."""
        data = _closed(obj, CONFIG_FIELDS, "providers_config")
        schema_version = data.get("schema_version")
        _need(
            isinstance(schema_version, int) and not isinstance(schema_version, bool),
            "schema_version_invalid",
            "providers_config.schema_version must be the integer %d" % PROVIDERS_SCHEMA_VERSION,
        )
        _need(
            schema_version == PROVIDERS_SCHEMA_VERSION,
            "schema_version_unsupported",
            "providers_config.schema_version %r is unsupported (expected %d)"
            % (schema_version, PROVIDERS_SCHEMA_VERSION),
        )

        raw_rows = cast(List[Any], data.get("providers"))
        _need(
            isinstance(raw_rows, list) and len(raw_rows) > 0,
            "providers_invalid",
            "providers_config.providers must be a non-empty list",
        )
        rows: List[ProviderRow] = []
        seen: Dict[str, int] = {}
        for index, raw_row in enumerate(raw_rows):
            row = parse_provider_row(raw_row, index)
            if row.id in seen:
                raise ProviderConfigError(
                    "duplicate_provider_id",
                    "providers[%d].id %r is already declared by providers[%d]"
                    % (index, row.id, seen[row.id]),
                    provider_id=row.id,
                )
            seen[row.id] = index
            rows.append(row)

        allow_fallback = _bool_field(data, "allow_fallback", "providers_config", None)
        default_provider = _str_field(
            data, "default_provider", "providers_config", PROVIDER_ID_RE, required=False
        )
        if default_provider is not None:
            _need(
                default_provider in seen,
                "unknown_provider_id",
                "providers_config.default_provider %r is not a declared provider id"
                % (default_provider,),
                provider_id=default_provider,
            )
        if allow_fallback:
            _need(
                default_provider is not None,
                "default_provider_required",
                "allow_fallback=true requires an explicit default_provider (the chain head)",
            )
        if default_provider is None and len(rows) > 1:
            raise ProviderConfigError(
                "provider_selection_ambiguous",
                "providers_config declares %d providers without a default_provider" % len(rows),
            )
        return cls(rows, default_provider, allow_fallback)

    @classmethod
    def from_file(cls, path: str) -> "ProviderRegistry":
        """Load one operator-supplied JSON file.  No implicit path, no env lookup."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise ProviderConfigError(
                "config_unreadable", "providers file could not be read (%s)" % type(exc).__name__
            )
        try:
            data = json.loads(raw)
        except ValueError:
            raise ProviderConfigError("config_unparsable_json", "providers file is not valid JSON")
        return cls.from_dict(data)

    # -- credential resolution (D1) ---------------------------------------

    def row(self, provider_id: str) -> ProviderRow:
        try:
            return self._by_id[str(provider_id)]
        except KeyError:
            raise ProviderConfigError(
                "unknown_provider_id",
                "provider id %r is not declared in the registry" % (provider_id,),
            )

    def api_key_env_for(self, row: ProviderRow) -> Optional[str]:
        return row.api_key_env

    def api_key(self, row: ProviderRow) -> Optional[str]:
        """The key VALUE, read from the environment by name -- never returned to callers
        other than the backend constructor, and never serialized."""
        if not row.api_key_env:
            return None
        value = os.environ.get(row.api_key_env)
        return value if value else None

    def api_key_present(self, row: ProviderRow) -> bool:
        return self.api_key(row) is not None

    def disabled_reason(self, row: ProviderRow) -> Optional[str]:
        """A machine code, or None when the provider is selectable (D3)."""
        if row.kind == "openai_compatible":
            if not self.api_key_present(row):
                return DISABLED_REASON_KEY_UNSET
        return None

    def capabilities(self) -> List[Dict[str, Any]]:
        """Per-provider capability rows: booleans and non-secret labels only (D1)."""
        out: List[Dict[str, Any]] = []
        for row in self.rows:
            reason = self.disabled_reason(row)
            out.append(row.to_capability(self.api_key_present(row), reason is None, reason))
        return out

    # -- eligibility (section 2.3 / L21) ----------------------------------

    @staticmethod
    def _require_row_eligible(
        row: ProviderRow,
        provider_id: str,
        surface: Optional[str] = None,
        request_class: Optional[str] = None,
    ) -> None:
        """Fail closed when a row may not be offered a surface or a data class.

        Both refusals are ``ProviderConfigError``: they are configuration failures, so they
        are never retried and never fall back to another row (D-D).
        """
        if surface is not None:
            _need(
                row.supports_surface(surface),
                "surface_unavailable",
                "provider %r does not declare the %r surface (surfaces=%s)"
                % (provider_id, surface, list(row.surfaces)),
                provider_id,
            )
        if request_class is not None:
            _need(
                row.class_allows(request_class),
                "data_class_refused",
                "provider %r (data_class=%s) may not be offered a %r request"
                % (provider_id, row.data_class, request_class),
                provider_id,
            )

    def eligible_rows(
        self, surface: Optional[str] = None, request_class: Optional[str] = None
    ) -> List[ProviderRow]:
        """Rows that may be considered for a call, in declared order.

        Eligibility is a FILTER computed before any model call, never a score and never a
        downgrade: a row that is not returned is simply one this request may not be offered
        to.  An omitted ``surface``/``request_class`` means "no filter on that axis".
        """
        out: List[ProviderRow] = []
        for row in self.rows:
            if surface is not None and not row.supports_surface(surface):
                continue
            if request_class is not None and not row.class_allows(request_class):
                continue
            out.append(row)
        return out

    def rows_for(self, surface: str, request_class: Optional[str] = None) -> List[str]:
        """The eligible provider ids for one surface (declared order preserved)."""
        return [row.id for row in self.eligible_rows(surface=surface, request_class=request_class)]

    # -- selection ---------------------------------------------------------

    def primary_id(self) -> str:
        """The chain head: the declared ``default_provider``, or the only declared row."""
        if self.default_provider is not None:
            return self.default_provider
        if len(self.order) == 1:
            return self.order[0]
        raise ProviderConfigError(
            "provider_selection_ambiguous",
            "no default_provider declared and the registry is not a single provider",
        )

    def backend_for(
        self,
        provider_id: str,
        transport: Optional[Transport] = None,
        surface: Optional[str] = None,
        request_class: Optional[str] = None,
    ) -> ClassificationBackend:
        """Build ONE backend.  A disabled provider is a structured hard error (D3)."""
        row = self.row(provider_id)
        self._require_row_eligible(row, provider_id, surface=surface, request_class=request_class)
        reason = self.disabled_reason(row)
        if reason is not None:
            raise ProviderConfigError(
                reason, "provider %r is disabled and is never silently replaced" % row.id, provider_id=row.id
            )
        if row.kind == "local":
            return SyntheticBackend()
        return OpenAICompatibleBackend(row, api_key=self.api_key(row), transport=transport)

    def select_backend(
        self,
        transport: Optional[Transport] = None,
        surface: Optional[str] = None,
        request_class: Optional[str] = None,
    ) -> ClassificationBackend:
        """The entry point used by the CLI: one backend, or the declared chain.

        ``surface`` and ``request_class`` filter the table BEFORE anything is built: an
        explicitly named ``default_provider`` that fails either filter is a hard error (the
        operator named it, so it is never silently substituted), while automatic selection
        and the fallback chain are drawn only from the filtered set.
        """
        primary = self.primary_id()
        self._require_row_eligible(self.row(primary), primary, surface=surface, request_class=request_class)
        if self.allow_fallback and len(self.order) > 1:
            eligible = [
                row.id for row in self.eligible_rows(surface=surface, request_class=request_class)
            ]
            ordered = [primary] + [pid for pid in eligible if pid != primary]
            entries: List[Tuple[str, ClassificationBackend]] = []
            for pid in ordered:
                entries.append(
                    (
                        pid,
                        self.backend_for(
                            pid,
                            transport=transport,
                            surface=surface,
                            request_class=request_class,
                        ),
                    )
                )
            return ProviderChainBackend(entries, chain_order=ordered)
        return self.backend_for(
            primary, transport=transport, surface=surface, request_class=request_class
        )


# --------------------------------------------------------------------------
# OpenAI-compatible backend (D3/D4/D6/D7)
# --------------------------------------------------------------------------


class OpenAICompatibleBackend(ClassificationBackend):
    """One chat-completions POST per surface, strict JSON reply contract.

    ``name`` is the reserved backend enum value ``openjev``: the receipt's
    ``backend_used`` domain is closed, so an external OpenAI-compatible classifier
    (a "Jev-style" row is exactly that) reports the enum value, never a dynamic id.
    """

    name = "openjev"

    def __init__(
        self,
        row: ProviderRow,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        min_confidence: float = MIN_CONFIDENCE_TAXONOMY,
    ) -> None:
        super().__init__()
        if getattr(row, "kind", None) != "openai_compatible":
            raise ProviderConfigError(
                "kind_not_compatible",
                "OpenAICompatibleBackend requires an openai_compatible row",
                provider_id=getattr(row, "id", None),
            )
        self.row = row
        self._api_key = api_key or ""
        self._transport = transport
        self.min_confidence = float(min_confidence)
        self.transport_calls = 0
        self.last_request_url: Optional[str] = None
        self.last_payload: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:  # never render credential material (D1)
        return "<OpenAICompatibleBackend id=%r model=%r key=redacted>" % (self.row.id, self.row.model)

    # -- payload / url -----------------------------------------------------

    def request_url(self) -> str:
        url = _validate_base_url(
            self.row.chat_completions_url(), self.row.allow_private_network, "request_url"
        )
        self.last_request_url = url
        return url

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        return headers

    def build_batch_payload(
        self,
        request_handle: RequestHandle,
        candidate_ids: Sequence[str],
        surface: str,
        profile_candidates: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Hash/id-only batch payload: no prompt text ever leaves the process.

        THE payload choke point (D8 item 5).  ``candidate_ids`` is exactly the list
        the caller's candidate set yielded for ``surface`` -- this method never
        widens, reorders, adds or drops one.  On the ``profile`` surface the
        whitelisted profile metadata block is appended; it must describe exactly the
        non-sentinel candidates, in the same order, or the batch is refused before any
        transport call.
        """
        user_message: Dict[str, Any] = {
            "content": BATCH_CONTENT_TOKEN,
            "surface": surface,
            "prompt_hash": request_handle.prompt_hash,
            "context_hash": request_handle.context_hash or NO_CONTEXT_TOKEN,
            "project_id_hash": request_handle.project_id_hash,
            "candidates": [str(c) for c in candidate_ids],
        }
        if surface == "profile":
            candidates_given = [str(c) for c in candidate_ids]
            eligible_ids = [c for c in candidates_given if c != PROFILE_SENTINEL_ABSTAIN]
            rows = _profile_request_rows(list(profile_candidates or ()))
            row_ids = [row["profile_id"] for row in rows]
            if row_ids != eligible_ids:
                # belt and braces: metadata must describe exactly the eligible
                # candidate set, in the caller's order.  A mismatch means something
                # tried to widen or reorder eligibility on the way to the wire.
                raise ProviderContractError(
                    "candidate_set_mismatch",
                    "profile metadata must cover exactly the eligible candidate set "
                    "(ids=%s metadata=%s)" % (candidates_given[:8], row_ids[:8]),
                    provider_id=self.row.id,
                )
            user_message[PROFILE_BLOCK_KEY] = rows
        payload = {
            "model": self.row.model,
            "temperature": 0,
            "max_tokens": int(self.row.max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_TOKEN},
                user_message,
            ],
        }
        assert_profile_payload_whitelisted(payload)
        assert_hash_only_payload(payload)
        return payload

    def build_proposal_payload(
        self, request_handle: RequestHandle, exemplar_hashes: Sequence[str]
    ) -> Dict[str, Any]:
        payload = {
            "model": self.row.model,
            "temperature": 0,
            "max_tokens": int(self.row.max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": PROPOSAL_SYSTEM_TOKEN},
                {
                    "role": "user",
                    "content": PROPOSAL_CONTENT_TOKEN,
                    "prompt_hash": request_handle.prompt_hash,
                    "context_hash": request_handle.context_hash or NO_CONTEXT_TOKEN,
                    "project_id_hash": request_handle.project_id_hash,
                    "exemplar_hashes": [str(h) for h in exemplar_hashes],
                },
            ],
        }
        assert_hash_only_payload(payload)
        return payload

    # -- transport ---------------------------------------------------------

    def _post_json(self, payload: Mapping[str, Any]) -> Any:
        """ONE POST.  Retries only connection-level availability failures (D3)."""
        url = self.request_url()
        body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        request = TransportRequest(
            method="POST",
            url=url,
            headers=self._headers(),
            body=body,
            timeout_s=float(self.row.timeout_s),
        )
        transport = self._transport or stdlib_transport
        attempts = int(self.row.max_retries) + 1
        last_error: Optional[ProviderTransportError] = None
        for attempt in range(attempts):
            self.transport_calls += 1
            try:
                response = transport(request)
            except ProviderTransportError as exc:
                last_error = exc
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = ProviderTransportError(
                    "transport_failed",
                    "transport raised %s" % type(exc).__name__,
                    provider_id=self.row.id,
                )
                continue
            status = int(getattr(response, "status", 0) or 0)
            if status != 200:
                if 300 <= status < 400:
                    raise ProviderTransportError(
                        "redirect_refused",
                        "redirect (HTTP %d) is refused and never followed" % status,
                        provider_id=self.row.id,
                    )
                error = ProviderTransportError(
                    "http_status",
                    "provider returned HTTP status %d" % status,
                    provider_id=self.row.id,
                )
                if 500 <= status < 600 and attempt + 1 < attempts:
                    last_error = error
                    continue
                raise error
            return self._decode_body(getattr(response, "body", b""))
        raise last_error or ProviderTransportError(
            "transport_failed", "transport exhausted %d attempt(s)" % attempts, provider_id=self.row.id
        )

    def _decode_body(self, raw: Any) -> Any:
        if isinstance(raw, (bytes, bytearray)):
            try:
                text = bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                raise ProviderContractError(
                    "reply_not_json", "provider reply body is not UTF-8 JSON", provider_id=self.row.id
                )
        elif isinstance(raw, str):
            text = raw
        else:
            raise ProviderContractError(
                "reply_not_json", "provider reply body is not text", provider_id=self.row.id
            )
        try:
            return json.loads(text)
        except ValueError:
            raise ProviderContractError(
                "reply_not_json", "provider reply body is not JSON", provider_id=self.row.id
            )

    # -- interface ---------------------------------------------------------

    def batch_score(
        self,
        request_handle: RequestHandle,
        candidates: CandidateSet,
        surface: str,
    ) -> List[Dict[str, Any]]:
        try:
            candidate_ids = candidates.candidate_ids_for(surface)
        except ValueError:
            # unknown surface tokens are refused, never coerced onto a known surface
            raise ProviderContractError(
                "unknown_surface", "unknown surface %r" % (surface,), provider_id=self.row.id
            )
        if not self.row.supports_surface(surface):
            # A surface the row does not declare is refused before anything is counted or
            # sent: the call is never rerouted to a different provider for a different
            # surface, and it is never quietly scored by another family.
            raise ProviderConfigError(
                "surface_unavailable",
                "provider %r does not declare the %r surface (surfaces=%s)"
                % (self.row.id, surface, list(self.row.surfaces)),
                provider_id=self.row.id,
            )
        if len(candidate_ids) > self.row.batch_max_candidates:
            raise ProviderConfigError(
                "batch_too_large",
                "surface %r carries %d candidates (row limit %d)"
                % (surface, len(candidate_ids), self.row.batch_max_candidates),
                provider_id=self.row.id,
            )
        self.batch_calls[surface] = self.batch_calls.get(surface, 0) + 1
        # D8: eligibility is upstream; the eligible profiles are passed through
        # unchanged (order included) so the request can never score a profile the
        # caller excluded, or drop one it included.
        profile_candidates = tuple(candidates.profiles) if surface == "profile" else None
        payload = self.build_batch_payload(
            request_handle, candidate_ids, surface, profile_candidates
        )
        self.last_payload = payload
        document = self._post_json(payload)
        return validate_batch_reply(
            document,
            candidate_ids,
            provider_id=self.row.id,
            strict_order=(surface == "profile"),
        )

    def propose_taxonomy(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not self.row.supports_taxonomy_proposal:
            raise ProviderConfigError(
                "taxonomy_proposal_not_enabled",
                "row %r does not enable taxonomy proposals "
                "(supports_taxonomy_proposal=false)" % self.row.id,
                provider_id=self.row.id,
            )
        payload = self.build_proposal_payload(request_handle, exemplar_hashes)
        self.last_payload = payload
        document = self._post_json(payload)
        return validate_taxonomy_reply(document, self.min_confidence, provider_id=self.row.id)


# --------------------------------------------------------------------------
# Declared fallback chain (D3) -- never silent
# --------------------------------------------------------------------------


class ProviderChainBackend(ClassificationBackend):
    """The operator-declared ordered chain.  Used ONLY when ``allow_fallback`` is
    true and a ``default_provider`` names the chain head.

    Every fallback that fires is recorded in ``fallback_events`` and in
    ``last_fallback_from`` (the id of the provider the scores were NOT taken from),
    which the router copies into the receipt's ``fallback_from`` field.
    """

    def __init__(
        self, entries: Sequence[Tuple[str, ClassificationBackend]], chain_order: Sequence[str]
    ) -> None:
        super().__init__()
        self.entries: Tuple[Tuple[str, ClassificationBackend], ...] = tuple(entries)
        self.chain_order: List[str] = [str(pid) for pid in chain_order]
        self.fallback_events: List[Dict[str, Any]] = []
        self.last_fallback_from: Optional[str] = None
        primary = self.entries[0][1] if self.entries else None
        self.name = getattr(primary, "name", "synthetic")

    def __repr__(self) -> str:
        return "<ProviderChainBackend chain=%r>" % (self.chain_order,)

    @property
    def providers(self) -> List[str]:
        return [pid for pid, _ in self.entries]

    def batch_score(
        self,
        request_handle: RequestHandle,
        candidates: CandidateSet,
        surface: str,
    ) -> List[Dict[str, Any]]:
        for index, (provider_id, backend) in enumerate(self.entries):
            try:
                rows = backend.batch_score(request_handle, candidates, surface)
            except ProviderError as exc:
                is_last = index == len(self.entries) - 1
                if is_last or not is_fallback_trigger(exc):
                    raise
                self.fallback_events.append(
                    {
                        "surface": surface,
                        "from": provider_id,
                        "to": self.entries[index + 1][0],
                        "code": exc.code,
                    }
                )
                self.last_fallback_from = provider_id
                continue
            return rows
        raise ProviderConfigError(
            "provider_selection_ambiguous", "the declared chain is empty"
        )

    def propose_taxonomy(
        self,
        request_handle: RequestHandle,
        exemplar_hashes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        for _provider_id, backend in self.entries:
            try:
                return backend.propose_taxonomy(request_handle, exemplar_hashes)
            except ProviderConfigError as exc:
                if exc.code == "taxonomy_proposal_not_enabled":
                    continue
                raise
        raise ProviderConfigError(
            "taxonomy_proposal_not_enabled", "no provider in the declared chain enables taxonomy proposals"
        )


# --------------------------------------------------------------------------
# Reply contract validators (D3 / D6)
# --------------------------------------------------------------------------

REPLY_FIELDS = ("choices",)
SCORE_ROW_FIELDS = ("candidate", "raw")
SCORE_DOC_FIELDS = ("scores",)
PROPOSAL_FIELDS = ("slug", "parent_slug", "confidence")

# --------------------------------------------------------------------------
# Profile payload whitelist -- the single choke point (D8 item 5)
# --------------------------------------------------------------------------

#: the ONLY profile metadata fields that may ever be serialised into a provider
#: request.  Eligibility, project hashes, idle days, ordering keys, display names,
#: free-text descriptions and every other attribute stay in-process.
PROFILE_REQUEST_FIELDS = ("profile_id", "scope", "privacy_class", "labels")
PROFILE_BLOCK_KEY = "profile_candidates"
#: payload keys allowed to mention the word "profile" at all (adversarial backstop:
#: a smuggled `profile_description` / `profile_note` fails the assertion)
PROFILE_KEY_ALLOWLIST = (PROFILE_BLOCK_KEY, "profile_id", "no_suitable_profile")


class ProfileMetadataViolation(PrivacyViolationError):
    """Raised when profile metadata outside the whitelist is about to leave (D8)."""


def _profile_request_row(profile: Any) -> Dict[str, Any]:
    """Build the whitelisted request row for ONE eligible profile.

    This is the only place a profile is converted for transport.  It constructs a
    FRESH dict from the whitelisted fields -- it never copies a caller mapping
    through -- and re-validates every value against its closed domain, so an unknown
    field, an out-of-domain enum value or a free-text string raises instead of being
    sent.
    """
    if isinstance(profile, Mapping):
        unknown = sorted(set(profile.keys()) - set(PROFILE_REQUEST_FIELDS))
        if unknown:
            raise ProfileMetadataViolation(
                "profile row carries non-whitelisted field(s): %s (whitelist=%s)"
                % (unknown, list(PROFILE_REQUEST_FIELDS))
            )
        row = {key: profile.get(key) for key in PROFILE_REQUEST_FIELDS}
    else:
        raw = getattr(profile, "to_dict", None)
        if raw is None or not callable(raw):
            raise ProfileMetadataViolation(
                "profile candidate must be a ProfileCandidate or a whitelisted mapping"
            )
        built_raw = raw()
        if not isinstance(built_raw, Mapping):
            raise ProfileMetadataViolation(
                "profile candidate must expose a whitelisted mapping, got %s"
                % type(built_raw).__name__
            )
        built = cast(Mapping[str, Any], built_raw)
        unknown = sorted(set(built.keys()) - set(PROFILE_REQUEST_FIELDS))
        if unknown:
            raise ProfileMetadataViolation(
                "profile row carries non-whitelisted field(s): %s" % (unknown,)
            )
        row = {key: built.get(key) for key in PROFILE_REQUEST_FIELDS}

    profile_id = row.get("profile_id")
    if not isinstance(profile_id, str) or not SLUG_RE.match(profile_id):
        raise ProfileMetadataViolation("profile_id is outside the closed id domain")
    scope = row.get("scope")
    if scope not in PROFILE_SCOPES:
        raise ProfileMetadataViolation(
            "scope %r is outside the closed domain %s" % (scope, list(PROFILE_SCOPES))
        )
    privacy_class = row.get("privacy_class")
    if privacy_class not in PROFILE_PRIVACY_CLASSES:
        raise ProfileMetadataViolation(
            "privacy_class %r is outside the closed domain %s"
            % (privacy_class, list(PROFILE_PRIVACY_CLASSES))
        )
    labels = row.get("labels") or ()
    if isinstance(labels, str) or not isinstance(labels, (list, tuple)):
        raise ProfileMetadataViolation("labels must be a sequence of label slugs")
    out_labels: List[str] = []
    for label in labels:
        if not isinstance(label, str) or not SLUG_RE.match(label):
            raise ProfileMetadataViolation(
                "label %r is outside the closed id domain (labels are ids, never text)" % (label,)
            )
        out_labels.append(label)
    return {
        "profile_id": profile_id,
        "scope": str(scope),
        "privacy_class": str(privacy_class),
        "labels": out_labels,
    }


def _profile_request_rows(profiles: Sequence[Any]) -> List[Dict[str, Any]]:
    """THE choke point: the only function that turns candidates into profile rows."""
    return [_profile_request_row(profile) for profile in profiles]


def assert_profile_payload_whitelisted(payload: Any, path: str = "payload") -> None:
    """Adversarial backstop for the D8 privacy invariant.

    Called on every assembled batch payload: outside ``profile_candidates``, no key
    may even *mention* a profile field name, and inside it every row must carry
    exactly the whitelist.  Anything else -- a description, a display name, a
    permission list, a stray eligibility hint -- fails loudly instead of shipping.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            if "profile" in key_text.lower() and key_text not in PROFILE_KEY_ALLOWLIST:
                raise ProfileMetadataViolation(
                    "%s: key %r is not on the profile payload allowlist %s"
                    % (path, key_text, list(PROFILE_KEY_ALLOWLIST))
                )
            if key_text == PROFILE_BLOCK_KEY:
                if not isinstance(value, (list, tuple)):
                    raise ProfileMetadataViolation("%s.%s must be a list" % (path, key_text))
                for index, row in enumerate(value):
                    if not isinstance(row, Mapping):
                        raise ProfileMetadataViolation(
                            "%s.%s[%d] must be an object" % (path, key_text, index)
                        )
                    unknown = sorted(set(row.keys()) - set(PROFILE_REQUEST_FIELDS))
                    if unknown:
                        raise ProfileMetadataViolation(
                            "%s.%s[%d] carries non-whitelisted field(s): %s"
                            % (path, key_text, index, unknown)
                        )
                    missing = [f for f in PROFILE_REQUEST_FIELDS if f not in row]
                    if missing:
                        raise ProfileMetadataViolation(
                            "%s.%s[%d] is missing whitelisted field(s): %s"
                            % (path, key_text, index, missing)
                        )
                continue
            assert_profile_payload_whitelisted(value, "%s.%s" % (path, key_text))
        return
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_profile_payload_whitelisted(item, "%s[%d]" % (path, index))


def _reply_content(document: Any, provider_id: Optional[str]) -> Any:
    """Extract and parse the strict JSON object carried in the assistant message."""
    if not isinstance(document, Mapping):
        raise ProviderContractError(
            "reply_not_object", "provider reply is not a JSON object", provider_id=provider_id
        )
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderContractError(
            "reply_choices_invalid", "provider reply carries no 'choices' list", provider_id=provider_id
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ProviderContractError(
            "reply_choice_invalid", "choices[0] is not an object", provider_id=provider_id
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ProviderContractError(
            "reply_message_invalid", "choices[0].message is not an object", provider_id=provider_id
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderContractError(
            "reply_content_missing", "choices[0].message.content is not a string", provider_id=provider_id
        )
    try:
        parsed = json.loads(content)
    except ValueError:
        raise ProviderContractError(
            "reply_content_not_json", "message content is not JSON", provider_id=provider_id
        )
    if not isinstance(parsed, Mapping):
        raise ProviderContractError(
            "reply_content_not_object", "message content JSON is not an object", provider_id=provider_id
        )
    return parsed


def validate_batch_reply(
    document: Any,
    expected_ids: Sequence[str],
    provider_id: Optional[str] = None,
    strict_order: bool = False,
) -> List[Dict[str, Any]]:
    """The exact-response validator of D3/D6.

    The reply must cover EXACTLY the candidate id set -- missing, extra, duplicate
    and unknown ids are each rejected -- and every score must be a finite number in
    [0, 1].  Order is normalised to the requested candidate order so the caller sees
    a deterministic batch.
    """
    expected = [str(c) for c in expected_ids]
    doc = _reply_content(document, provider_id)
    if "scores" not in doc:
        raise ProviderContractError(
            "reply_scores_missing", "message content has no 'scores' field", provider_id=provider_id
        )
    unknown = sorted(set(doc.keys()) - set(SCORE_DOC_FIELDS))
    if unknown:
        raise ProviderContractError(
            "reply_unknown_field",
            "message content carries unknown field(s): %s" % (unknown,),
            provider_id=provider_id,
        )
    rows = doc["scores"]
    if not isinstance(rows, list):
        raise ProviderContractError(
            "reply_scores_not_list", "'scores' must be a list", provider_id=provider_id
        )

    seen: Dict[str, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProviderContractError(
                "score_row_not_object", "scores[%d] is not an object" % index, provider_id=provider_id
            )
        unknown = sorted(set(row.keys()) - set(SCORE_ROW_FIELDS))
        if unknown:
            raise ProviderContractError(
                "score_row_unknown_field",
                "scores[%d] carries unknown field(s): %s" % (index, unknown),
                provider_id=provider_id,
            )
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or not ID_TOKEN_RE.match(candidate):
            raise ProviderContractError(
                "score_candidate_invalid",
                "scores[%d].candidate is outside the closed id domain" % index,
                provider_id=provider_id,
            )
        raw = row.get("raw")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProviderContractError(
                "score_not_numeric", "scores[%d].raw is not a number" % index, provider_id=provider_id
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ProviderContractError(
                "score_not_finite", "scores[%d].raw is not finite" % index, provider_id=provider_id
            )
        if not (0.0 <= value <= 1.0):
            raise ProviderContractError(
                "score_out_of_range",
                "scores[%d].raw is outside [0, 1]" % index,
                provider_id=provider_id,
            )
        if candidate in seen:
            raise ProviderContractError(
                "duplicate_candidate_id",
                "candidate %r appears more than once" % candidate,
                provider_id=provider_id,
            )
        seen[candidate] = value

    wanted = set(expected)
    missing = [candidate for candidate in expected if candidate not in seen]
    extra = sorted(candidate for candidate in seen if candidate not in wanted)
    if missing or extra:
        raise ProviderContractError(
            "candidate_set_mismatch",
            "reply must cover exactly the candidate id set (missing=%s extra=%s)"
            % (missing[:8], extra[:8]),
            provider_id=provider_id,
        )
    actual_order = [str(row.get("candidate")) for row in rows]
    if strict_order and actual_order != expected:
        raise ProviderContractError(
            "candidate_order_mismatch",
            "reply candidate order must match the requested candidate order",
            provider_id=provider_id,
        )
    return [{"candidate": candidate, "raw": seen[candidate]} for candidate in expected]


def validate_taxonomy_reply(
    document: Any, min_confidence: float = MIN_CONFIDENCE_TAXONOMY, provider_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Closed-domain proposal validation; a non-conforming proposal is refused."""
    doc = _reply_content(document, provider_id)
    unknown = sorted(set(doc.keys()) - set(PROPOSAL_FIELDS))
    if unknown:
        raise ProviderContractError(
            "proposal_unknown_field",
            "proposal carries unknown field(s): %s" % (unknown,),
            provider_id=provider_id,
        )
    slug = doc.get("slug")
    confidence = doc.get("confidence")
    parent_slug = doc.get("parent_slug")
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    value = float(confidence)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        return None
    if value < float(min_confidence):
        return None
    if parent_slug is not None and (not isinstance(parent_slug, str) or not SLUG_RE.match(parent_slug)):
        return None
    return {"slug": slug, "parent_slug": parent_slug, "confidence": value}
