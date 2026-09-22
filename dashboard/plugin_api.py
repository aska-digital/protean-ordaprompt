"""Desktop backend for the manual OrdaPrompt routing test.

This endpoint is deliberately narrow: it accepts only the router's structured
request and candidate documents, always uses the locked default calibration and
never accepts provider paths, credentials, receipts paths, or arbitrary code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ordaprompt_router.router import route, RouterConfig
from ordaprompt_router.schemas import CandidateSet, ClassificationRequest, SchemaError

router = APIRouter()
_ROOT = Path(__file__).resolve().parents[1]
_MAX_JSON_CHARS = 200_000


def _document(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail={"code": "schema_reject", "field": name})
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) > _MAX_JSON_CHARS:
        raise HTTPException(status_code=413, detail={"code": "input_too_large", "field": name})
    return value


def _run(body: dict[str, Any]) -> dict[str, Any]:
    request = ClassificationRequest.from_dict(_document(body.get("request"), "request"))
    candidates = CandidateSet.from_dict(_document(body.get("candidates"), "candidates"))
    if request.request_id != candidates.request_id:
        raise SchemaError("request_id mismatch")
    surface = body.get("surface", "topic")
    if surface not in {"topic", "session", "profile"}:
        raise SchemaError("surface must be topic, session, or profile")
    # The installed release is intentionally local-only and calibration-locked.
    # No provider or calibration document can be supplied through this endpoint.
    result = route(request, candidates, RouterConfig(), include_session=surface == "session")
    return {
        "surface": surface,
        "routing_decision": result.decision.to_dict(),
        "routing_receipt": result.receipt,
        "calibration_locked": True,
        "provider": "local",
    }


@router.get("/demo")
async def demo() -> dict[str, Any]:
    """Return the repository's synthetic, hash-only manual test fixture."""
    try:
        request = json.loads((_ROOT / "eval" / "demo" / "request.json").read_text())
        candidates = json.loads((_ROOT / "eval" / "demo" / "candidates.json").read_text())
        return {"request": request, "candidates": candidates}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail={"code": "demo_unavailable"}) from exc


@router.post("/route")
async def manual_route(body: dict[str, Any]) -> dict[str, Any]:
    """Execute one explicit local routing comparison; never routes a chat turn."""
    try:
        return _run(body)
    except HTTPException:
        raise
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail={"code": "schema_reject", "message": str(exc)}) from exc
    except Exception as exc:  # fail closed: no guessed decision on unexpected errors
        raise HTTPException(status_code=422, detail={"code": "router_error", "message": type(exc).__name__}) from exc
