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
from .schemas import ID_TOKEN_RE, SLUG_RE, CandidateSet

# --------------------------------------------------------------------------
# Constants / closed domains
# --------------------------------------------------------------------------

PROVIDERS_SCHEMA_VERSION = 1
PROVIDER_KINDS = ("local", "openai_compatible")

CONFIG_FIELDS = ("schema_version", "providers", "default_provider", "allow_fallback")
ROW_FIELDS = (
    "id",
    "kind",
    "base_url",
    "model",
    "api_key_env",
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
    """One ``providers.json`` row.  Holds the API key env var NAME, never a value."""

    id: str
    kind: str
    base_url: Optional[str] = None
    model: str = ""
    api_key_env: Optional[str] = None
    allow_private_network: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_max_candidates: int = DEFAULT_BATCH_MAX_CANDIDATES
    supports_taxonomy_proposal: bool = False

    @property
    def requires_network(self) -> bool:
        return self.kind == "openai_compatible"

    def chat_completions_url(self) -> str:
        return (self.base_url or "").rstrip("/") + "/chat/completions"

    def to_capability(self, api_key_present: bool, enabled: bool, disabled_reason: Optional[str]) -> Dict[str, Any]:
        """Booleans + non-secret labels only (D1): never a key value or length."""
        return {
            "id": self.id,
            "kind": self.kind,
            "requires_network": self.requires_network,
            "supports_batch_scores": self.requires_network,
            "supports_taxonomy_proposal": bool(self.supports_taxonomy_proposal),
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(api_key_present),
            "enabled": bool(enabled),
            "disabled_reason": disabled_reason,
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


def parse_provider_row(obj: Any, index: int) -> ProviderRow:
    """Validate one row of the closed ``providers`` schema (D3)."""
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
    batch_max_candidates = _int_field(
        raw, "batch_max_candidates", path, provider_id, DEFAULT_BATCH_MAX_CANDIDATES, 1, BATCH_CANDIDATES_CAP
    )
    supports_taxonomy_proposal = _bool_field(raw, "supports_taxonomy_proposal", path, provider_id)

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
                           default="local-deterministic") or "local-deterministic"
        api_key_env = _str_field(raw, "api_key_env", path, API_KEY_ENV_RE, required=False,
                                 provider_id=provider_id)
        return ProviderRow(
            id=provider_id,
            kind=kind,
            base_url=None,
            model=model,
            api_key_env=api_key_env,
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

    # -- selection ---------------------------------------------------------

    def backend_for(
        self, provider_id: str, transport: Optional[Transport] = None
    ) -> ClassificationBackend:
        """Build ONE backend.  A disabled provider is a structured hard error (D3)."""
        row = self.row(provider_id)
        reason = self.disabled_reason(row)
        if reason is not None:
            raise ProviderConfigError(
                reason, "provider %r is disabled and is never silently replaced" % row.id, provider_id=row.id
            )
        if row.kind == "local":
            return SyntheticBackend()
        return OpenAICompatibleBackend(row, api_key=self.api_key(row), transport=transport)

    def select_backend(self, transport: Optional[Transport] = None) -> ClassificationBackend:
        """The entry point used by the CLI: one backend, or the declared chain."""
        if self.default_provider is not None:
            primary = self.default_provider
        elif len(self.order) == 1:
            primary = self.order[0]
        else:
            raise ProviderConfigError(
                "provider_selection_ambiguous",
                "no default_provider declared and the registry is not a single provider",
            )
        if self.allow_fallback and len(self.order) > 1:
            ordered = [primary] + [pid for pid in self.order if pid != primary]
            entries: List[Tuple[str, ClassificationBackend]] = []
            for pid in ordered:
                entries.append((pid, self.backend_for(pid, transport=transport)))
            return ProviderChainBackend(entries, chain_order=ordered)
        return self.backend_for(primary, transport=transport)


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
        self, request_handle: RequestHandle, candidate_ids: Sequence[str], surface: str
    ) -> Dict[str, Any]:
        """Hash/id-only batch payload: no prompt text ever leaves the process."""
        payload = {
            "model": self.row.model,
            "temperature": 0,
            "max_tokens": int(self.row.max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_TOKEN},
                {
                    "role": "user",
                    "content": BATCH_CONTENT_TOKEN,
                    "surface": surface,
                    "prompt_hash": request_handle.prompt_hash,
                    "context_hash": request_handle.context_hash or NO_CONTEXT_TOKEN,
                    "project_id_hash": request_handle.project_id_hash,
                    "candidates": [str(c) for c in candidate_ids],
                },
            ],
        }
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
        if surface == "topic":
            candidate_ids = candidates.topic_candidate_ids()
        elif surface == "session":
            candidate_ids = candidates.session_candidate_ids()
        else:
            raise ProviderContractError(
                "unknown_surface", "unknown surface %r" % (surface,), provider_id=self.row.id
            )
        if len(candidate_ids) > self.row.batch_max_candidates:
            raise ProviderConfigError(
                "batch_too_large",
                "surface %r carries %d candidates (row limit %d)"
                % (surface, len(candidate_ids), self.row.batch_max_candidates),
                provider_id=self.row.id,
            )
        self.batch_calls[surface] = self.batch_calls.get(surface, 0) + 1
        payload = self.build_batch_payload(request_handle, candidate_ids, surface)
        self.last_payload = payload
        document = self._post_json(payload)
        return validate_batch_reply(document, candidate_ids, provider_id=self.row.id)

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
    document: Any, expected_ids: Sequence[str], provider_id: Optional[str] = None
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
