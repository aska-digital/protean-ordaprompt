# OrdaPilot classifier router -- offline eval (synthetic fixtures)

> SYNTHETIC FIXTURES ONLY: outcomes are designed by the fixture generator and scored by the deterministic hash backend. These numbers validate the routing/calibration machinery, not real-world classifier quality. Real calibration requires real receipts plus human feedback.

Generated: 2026-09-22T12:36:09Z   Cases: 400 (tuning 267 / holdout 133)

## Holdout metrics (calibrated simulation)
- topic accuracy (labelled): 1.0000 over 103 cases
- automatic assignment precision: 1.0000 over 61 automatic decisions
- novelty precision: 0.9375 (calls 16) / novelty recall: 1.0000
- session reuse accuracy: 1.0000 over 93 reuse decisions
- session contamination rate: 0.0000 (must be 0)
- session fragmentation: 0.0000 over 93 clean-reuse cases
- session decision accuracy: 1.0000 over 128 labelled cases
- abstention rate: 0.2707 ; automatic rate: 0.4586
- latency per decision (ms): mean 0.0533 / p50 0.0475 / p95 0.0851
- simulated tokens loaded: 20800 total, 156.39 per decision, 202000 refused for contamination

## Band distribution
- abstain_or_new_session: 36
- automatic: 61
- fallback_escalate: 36

## Calibration (Platt on tuning split)
- topic: a=1.7773 b=-1.8304 (n=210, positives=177)
- session: a=1.8240 b=-0.6516 (n=197, positives=187)
- holdout topic ECE 0.1330 / Brier 0.0408 ; session ECE 0.0635 / Brier 0.0236
- holdout would-be-automatic: 61 in band, precision 1.0000
- unlock checks:
    - holdout_in_band_ge_50: PASS
    - holdout_in_band_precision_ge_0.98: PASS
    - holdout_session_brier_le_0.10: PASS
    - holdout_session_ece_le_0.05: FAIL
    - holdout_topic_brier_le_0.10: PASS
    - holdout_topic_ece_le_0.05: FAIL
- unlocked: False -> production calibration model id: none

## Locked-constraint invariants
- uncalibrated automatic rate (tuning/holdout): 0.0000 / 0.0000 (locked #5: must be 0)
- one batch call per surface: {'topic_calls': 400, 'session_calls': 340, 'expected_topic_calls': 400, 'expected_session_calls': 340, 'session_side_skipped_cases': 60, 'note': 'one batch call per surface; the session surface is skipped when the candidate set has no live session', 'holds': True}
- uncalibrated abstention rate (holdout): 0.8496

## Fixture census
- ambiguous/clean_reuse: 30
- clear_topic/clean_reuse: 175
- clear_topic/contaminated: 15
- clear_topic/expensive_context: 15
- clear_topic/new_session_argmax: 15
- clear_topic/no_sessions: 30
- clear_topic/session_near_tie: 15
- low_confidence/clean_reuse: 15
- near_tie/clean_reuse: 30
- near_tie/no_sessions: 15
- novel/clean_reuse: 30
- novel/no_sessions: 15

Wall clock: 28.77s
