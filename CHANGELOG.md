# Changelog

All notable changes to `protean-ordaprompt`.

## 1.0.0 — 2026-09-22

Initial release. Jev-style comparative classifier for OrdaPilot request routing.

- Comparative classification: one backend comparison per surface, all topic labels
  and session choices scored together (including `novel`, `ambiguous`, and
  `new_session` sentinels).
- Closed-enum `RoutingDecision` (`automatic`, `confirm`, `abstain`,
  `abstain_or_new_session`) with hashes-only `RoutingReceipt`s.
- Fail-closed calibration lock: the `automatic` band is structurally unreachable
  until ECE ≤ 0.05 on real feedback data. Ships with
  `calibration_model_id="none"`.
- Stdlib only, no dependencies. No network, no telemetry, no background work.
- OpenRouter adapter ships disabled by policy (hashes-and-labels only when
  explicitly enabled for one invocation).
- Privacy validator (`assert_no_free_text`) re-checks every outbound payload;
  violations raise and nothing is written.
- Offline eval: 60/60 self-tests, 400 synthetic fixtures, 5/5 negative probes.
  Full results in `eval/results.md`.
- Merged into the Hermes plugin catalog via
  [NousResearch/hermes-agent#119124](https://github.com/NousResearch/hermes-agent/pull/119124)
  at pinned commit `60e89af2853072c7801c465f30999f3c6ceb35b2`.
