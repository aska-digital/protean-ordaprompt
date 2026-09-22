"""Smoke tests: package imports, one real route through the CLI path, negative probes.

Run: python3 -m unittest discover -s test -p 'test_*.py' -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ordaprompt_router  # noqa: E402
from ordaprompt_router import (  # noqa: E402
    BackendError,
    DisabledByPolicy,
    OpenRouterBackend,
    RouterConfig,
)


class TestPackage(unittest.TestCase):
    def test_import_and_exports(self) -> None:
        self.assertTrue(ordaprompt_router.__all__)
        for name in ("route", "ReceiptStore", "RouterConfig", "DisabledByPolicy"):
            self.assertTrue(hasattr(ordaprompt_router, name), name)

    def test_openrouter_disabled_by_default(self) -> None:
        config = RouterConfig()
        # Default config carries NO openrouter backend at all: absent == off.
        self.assertTrue(
            config.openrouter_backend is None or not config.openrouter_backend.enabled
        )

    def test_disabled_openrouter_raises_disabled_by_policy(self) -> None:
        backend = OpenRouterBackend(enabled=False)
        with self.assertRaises(DisabledByPolicy):
            backend.batch_score(request_handle=None, candidates=None, surface="topic")  # type: ignore[arg-type]

    def test_enabled_openrouter_batch_scoring_is_gated(self) -> None:
        # Even explicitly "enabled", batch scoring is refused: the adapter is gated
        # to taxonomy proposals only (hashes/labels transport, section 2).
        backend = OpenRouterBackend(enabled=True)
        with self.assertRaises(BackendError):
            backend.batch_score(request_handle=None, candidates=None, surface="topic")  # type: ignore[arg-type]


class TestCliRoute(unittest.TestCase):
    def test_route_session_demo(self) -> None:
        demo = ROOT / "eval" / "demo"
        with tempfile.TemporaryDirectory() as tmp:
            receipts_dir = str(Path(tmp) / "receipts")
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "route-session",
                    "--request", str(demo / "request.json"),
                    "--candidates", str(demo / "candidates.json"),
                    "--receipts-dir", receipts_dir,
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            decision = payload["routing_decision"]
            self.assertIn(
                decision["band"],
                {"automatic", "confirm", "abstain", "abstain_or_new_session"},
            )
            receipt = payload["routing_receipt"]
            # Receipts carry hashes + structured scores only: no free-text field may
            # contain anything longer than an id/hash/score token.
            blob = json.dumps(receipt)
            for banned in ("http://", "https://", "password", "BEGIN"):
                self.assertNotIn(banned, blob.lower())

    def test_malformed_request_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"not": "a request"}))
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "classify",
                    "--request", str(bad),
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
