# Security policy

`protean-ordaprompt` is a stdlib-only Hermes plugin. It makes no network calls,
runs no background work, and serializes no free text into receipts or logs.
There is no authentication, credential, or session-data surface to attack.

If you find a bug that weakens the fail-closed guarantees (for example, an
automatic-band route without calibration, or free text leaking into a receipt),
please open a GitHub issue on this repository with a minimal reproduction.
