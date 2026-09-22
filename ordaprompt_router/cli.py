"""CLI: classify (topic side) and route-session (topic + session side).

    python3 -m ordaprompt_router.cli classify      --request req.json [--candidates c.json]
    python3 -m ordaprompt_router.cli route-session --request req.json [--candidates c.json] \
                                                   --receipts-dir receipts/

`--providers <path>` opts in to an operator-supplied provider configuration; without
the flag no provider code is constructed and the local deterministic adapter stands
alone.  A providers file that cannot be trusted is a structured hard error: exit 2,
nothing routed, nothing written, no socket opened.

Prints the RoutingDecision and the RoutingReceipt as JSON.  Exit codes:
    0  routing produced (receipt written when --receipts-dir is given)
    2  usage / schema validation failure (fail-closed, nothing written)
    3  receipt rejected by the privacy/schema validator (fail-closed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .adapter import sha256_hex
from .providers import ProviderConfigError
from .receipts import ReceiptStore
from .router import CalibrationModel, RouterConfig, route
from .schemas import (
    CandidateSet,
    ClassificationRequest,
    PrivacyViolationError,
    ProfileCandidate,
    SchemaError,
    SessionCandidate,
    TopicCandidate,
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_candidates(args: argparse.Namespace, request: ClassificationRequest) -> CandidateSet:
    if args.candidates:
        return CandidateSet.from_dict(_load_json(args.candidates))
    topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()]
    sessions = [s.strip() for s in (args.sessions or "").split(",") if s.strip()]
    profile_ids = [p.strip() for p in (args.profiles or "").split(",") if p.strip()]
    # D8: the `profile` surface is driven by the candidate set.  Eligibility is the
    # caller's decision, so the CLI only ever passes through the ids it was given --
    # no widening, no default profile, no implicit allowlist.  An id outside the
    # closed slug domain is a schema reject (exit 2), never a silent coercion.
    profiles = [
        ProfileCandidate.from_dict(
            {
                "profile_id": profile_id,
                "scope": args.profile_scope,
                "privacy_class": args.profile_privacy_class,
                "labels": [],
            },
            "cli.profiles[%d]" % index,
        )
        for index, profile_id in enumerate(profile_ids)
    ]
    return CandidateSet(
        request_id=request.request_id,
        topics=tuple(TopicCandidate(topic_id=t) for t in topics),
        sessions=tuple(
            SessionCandidate(session_id=s, project_id_hash=request.project_id_hash)
            for s in sessions
        ),
        profiles=tuple(profiles),
    )


def _build_config(args: argparse.Namespace) -> RouterConfig:
    from .providers import DEFAULT_REQUEST_DATA_CLASS, ProviderRegistry, local_backend_key

    config: RouterConfig
    if args.config:
        config = RouterConfig.from_dict(_load_json(args.config))
    else:
        config = RouterConfig()
    if args.calibration:
        # L23: an inconsistent calibration state is refused HERE, at load, before any call.
        config.calibration = CalibrationModel.from_dict(_load_json(args.calibration))
    if args.providers:
        # D2: the ONLY way a provider enters the decision path.  No provider file ->
        # no provider code runs at all; the local deterministic adapter stands alone.
        registry = ProviderRegistry.from_file(args.providers)
        request_class = getattr(args, "data_class", None) or DEFAULT_REQUEST_DATA_CLASS
        # L21: a row below the request's data class is filtered before anything is built, and
        # the named head is never silently substituted.
        config.backend = registry.select_backend(request_class=request_class)
        # L16/L23: an ACTIVE calibration binds to the (kind, model, transform) it was fitted
        # for.  A mismatch is a hard error here -- never a silent downgrade and never a band
        # upgrade on parameters fitted elsewhere.
        config.calibration.check_provider_key(registry.row(registry.primary_id()))
    else:
        # Local-only run: the running batch scorer is the built-in deterministic adapter, and
        # an active calibration must still name that exact key.
        config.calibration.check_provider_key(local_backend_key())
    if args.enable_openrouter:
        from .adapter import OpenRouterBackend

        if config.openrouter_backend is None:
            config.openrouter_backend = OpenRouterBackend(enabled=True)
        else:
            config.openrouter_backend = OpenRouterBackend(
                enabled=True, model=config.openrouter_backend.model
            )
    if args.legacy_fallback:
        config.legacy_fallback = True
    return config


def _emit(decision: Dict[str, Any], receipt: Dict[str, Any], pretty: bool) -> None:
    payload = {"routing_decision": decision, "routing_receipt": receipt}
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _run(args: argparse.Namespace) -> int:
    # I-5: --profiles and --candidates are mutually exclusive.  Refuse the
    # combination HERE, before any file is read, any transport is constructed,
    # or anything is routed or written -- never a silent downgrade where one
    # source quietly wins over the other.
    if args.candidates and args.profiles:
        print(
            "profiles_with_candidates_refused: --profiles and --candidates "
            "are mutually exclusive; pass exactly one candidate source "
            "(a --candidates file, or inline --topics/--sessions/--profiles)",
            file=sys.stderr,
        )
        return 2
    try:
        request = ClassificationRequest.from_dict(_load_json(args.request))
        candidates = _build_candidates(args, request)
    except SchemaError as exc:
        print("schema_reject: %s" % exc, file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print("input_error: %s" % exc, file=sys.stderr)
        return 2

    try:
        config = _build_config(args)
    except ProviderConfigError as exc:
        # D3: a provider file that cannot be trusted is a structured hard error.
        # Nothing is routed, nothing is written, and no socket is opened.
        print("provider_reject: %s" % exc, file=sys.stderr)
        return 2
    result = route(
        request,
        candidates,
        config,
        include_session=(args.command == "route-session"),
    )

    if args.receipts_dir:
        store = ReceiptStore(args.receipts_dir)
        try:
            store.append(result.receipt)
        except (SchemaError, PrivacyViolationError) as exc:
            print("receipt_reject: %s" % exc, file=sys.stderr)
            return 3

    _emit(result.decision.to_dict(), result.receipt, args.pretty)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ordaprompt_router.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("classify", "topic-side classification only (one batch comparison)"),
        ("route-session", "topic side plus session reuse comparison (two batch comparisons)"),
    ):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--request", required=True, help="ClassificationRequest JSON file")
        child.add_argument("--candidates", help="CandidateSet JSON file (mutually exclusive with --profiles: passing both is exit 2, profiles_with_candidates_refused)")
        child.add_argument("--topics", help="comma-separated topic slugs (when --candidates is absent)")
        child.add_argument("--sessions", help="comma-separated session ids (when --candidates is absent)")
        child.add_argument(
            "--profiles",
            help="comma-separated eligible profile ids for the `profile` surface "
            "(when --candidates is absent; mutually exclusive with --candidates: "
            "passing both is exit 2, profiles_with_candidates_refused); the no_suitable_profile abstain candidate is "
            "always added, and the order given here is the order scored",
        )
        child.add_argument(
            "--profile-scope",
            choices=("global", "project", "session"),
            default="project",
            help="scope label carried for every profile given by --profiles (closed enum)",
        )
        child.add_argument(
            "--profile-privacy-class",
            choices=("public", "internal", "private", "restricted"),
            default="private",
            help="privacy class carried for every profile given by --profiles (closed enum)",
        )
        child.add_argument("--receipts-dir", help="append-only receipt directory")
        child.add_argument(
            "--data-class",
            dest="data_class",
            choices=("public", "internal", "private"),
            default="public",
            help="data class of THIS request (closed enum, default public); a provider row "
            "whose declared data_class is below it is filtered out before selection and is "
            "never offered the call",
        )
        child.add_argument("--calibration", help="calibration model JSON (default: none -> nothing automatic)")
        child.add_argument("--config", help="router config JSON: router.backends.openrouter.enabled, legacy_fallback")
        child.add_argument("--providers", help="operator-supplied providers JSON (default: none -> local deterministic adapter only)")
        child.add_argument("--enable-openrouter", action="store_true", help="explicitly enable the gated adapter (default off)")
        child.add_argument("--legacy-fallback", action="store_true", help="section 8 rollback: escalate everything")
        child.add_argument("--compact", dest="pretty", action="store_false", help="single-line JSON")
        child.set_defaults(pretty=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
