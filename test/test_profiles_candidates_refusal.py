"""Regression tests for I-5: --profiles + --candidates is a hard refusal.

Passing both candidate sources to one CLI invocation must exit 2 with the
machine code `profiles_with_candidates_refused` on stderr -- before any file
is read, any transport is constructed, anything is routed, or anything is
written.  Never a silent downgrade where one source quietly wins.

Every test here is offline (stdlib only, local deterministic adapter or no
adapter at all): no socket is opened, no third-party service is called, and no
real credential is used.

Run: python3 -m unittest discover -s test -p 'test_*.py' -v
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ordaprompt_router import cli as cli_module  # noqa: E402

REQUEST = str(ROOT / "eval" / "demo" / "request.json")
CANDIDATES = str(ROOT / "eval" / "demo" / "candidates.json")

MACHINE_CODE = "profiles_with_candidates_refused"


class TestProfilesWithCandidatesRefused(unittest.TestCase):
    """The combined invocation is refused; each source alone still routes."""

    def test_both_flags_exit_2_with_machine_code_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts = Path(tmp) / "receipts"
            proc = subprocess.run(
                [
                    sys.executable, "-m", "ordaprompt_router.cli", "classify",
                    "--request", REQUEST,
                    "--candidates", CANDIDATES,
                    "--profiles", "profile-alpha,profile-beta",
                    "--receipts-dir", str(receipts),
                ],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn(MACHINE_CODE, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "", "nothing is routed to stdout")
            written = list(receipts.rglob("*")) if receipts.exists() else []
            self.assertEqual(written, [], "nothing is written to the receipts dir")

    def test_refusal_fires_before_any_file_is_read(self):
        # Nonexistent inputs would be input_error/schema_reject if they were
        # reached -- the refusal must win, proving the guard runs first.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli_module.main([
                "classify",
                "--request", str(ROOT / "does-not-exist.json"),
                "--candidates", str(ROOT / "also-missing.json"),
                "--profiles", "profile-alpha",
            ])
        self.assertEqual(code, 2)
        self.assertIn(MACHINE_CODE, stderr.getvalue())

    def test_refusal_routes_nothing_and_opens_no_socket(self):
        calls = []

        def recording_route(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("route must never run on a refused invocation")

        real_route = cli_module.route
        real_socket = socket.socket
        cli_module.route = recording_route  # type: ignore[assignment]

        def guarded_socket(*args, **kwargs):
            raise AssertionError("no socket may open on a refused invocation")

        socket.socket = guarded_socket  # type: ignore[assignment]
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_module.main([
                    "classify",
                    "--request", REQUEST,
                    "--candidates", CANDIDATES,
                    "--profiles", "profile-alpha",
                ])
        finally:
            cli_module.route = real_route  # type: ignore[assignment]
            socket.socket = real_socket  # type: ignore[assignment]
        self.assertEqual(code, 2)
        self.assertIn(MACHINE_CODE, stderr.getvalue())
        self.assertEqual(calls, [], "route() must not be called")

    def test_candidates_alone_still_routes(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "ordaprompt_router.cli", "classify",
                "--request", REQUEST,
                "--candidates", CANDIDATES,
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("routing_decision", json.loads(proc.stdout))

    def test_profiles_alone_still_routes(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "ordaprompt_router.cli", "classify",
                "--request", REQUEST,
                "--topics", "topic-auth,topic-billing",
                "--profiles", "profile-alpha,profile-beta",
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("routing_decision", json.loads(proc.stdout))

    def test_help_documents_the_precedence(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ordaprompt_router.cli", "classify", "--help"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(MACHINE_CODE, proc.stdout)

    def test_readme_documents_the_precedence(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(MACHINE_CODE, readme)


if __name__ == "__main__":
    unittest.main()
