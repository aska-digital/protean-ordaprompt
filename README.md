# protean-ordaprompt

[![Website](https://img.shields.io/badge/website-aska--digital.github.io-blue)](https://aska-digital.github.io/protean-ordaprompt/)
[![Hermes catalog](https://img.shields.io/badge/hermes--agent-plugin%20catalog-PR%20%23119124-green)](https://github.com/NousResearch/hermes-agent/pull/119124)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Routes each OrdaPilot request to the right topic and the right session.** It is a plugin
for Hermes Agent. Instead of judging one label at a time, it compares every option side
by side and commits only when the evidence is strong. When the evidence is weak, it says
so instead of guessing.

## Install

```sh
hermes plugins install aska-digital/protean-ordaprompt --ref 60e89af2853072c7801c465f30999f3c6ceb35b2
hermes plugins enable protean-ordaprompt
```

Or install from the Hermes plugin catalog (`plugin-catalog/protean-ordaprompt.yaml` in the
hermes-agent repo). The rendered catalog page is
[here](https://hermes-agent.nousresearch.com/docs/plugins/protean-ordaprompt).

## How it decides

You send one request: a fingerprint of what was asked (a hash, never the raw text) plus
your candidate topics and sessions. The router scores every candidate in a single
comparison, then returns one of three answers:

- **Route it:** one option clearly wins. High score and a clear gap to second place.
  This answer is switched off in this release until real-world calibration proves it safe
  (see below).
- **Ask first:** the router hands back the ranked options and lets the caller decide.
- **Abstain:** the router declines and explains why. Low confidence, a genuinely new
  topic, real ambiguity, or a session that doesn't fit. Abstaining is a designed outcome,
  not an error.

Session reuse gets extra scrutiny. A session that matches the topic but belongs to a
different project, or costs too much context to reload, never wins automatically. If the
best topic and the best session disagree, the router starts fresh rather than forcing
a fit.

## What it never does

- **No guessing.** A weak score never routes.
- **No silent automation.** The automatic answer stays locked until calibration on real
  feedback data passes a strict bar. The synthetic tests honestly miss that bar, so this
  release ships locked.
- **No spying.** Pure Python, no extra dependencies. No network calls, no background
  work, nothing sent anywhere.
- **No raw text in its records.** Every receipt stores hashes and scores only. A built-in
  check rejects any receipt containing free text, and nothing is written when it trips.

The optional OpenRouter connection proposes new topic names after a strong novelty
signal. It ships switched off and stays off unless you enable it for one call. Even then
it carries hashes and labels only.

## Try it

```sh
git clone https://github.com/aska-digital/protean-ordaprompt
cd protean-ordaprompt
python3 -m ordaprompt_router.cli classify \
  --request eval/demo/request.json \
  --candidates eval/demo/candidates.json \
  --receipts-dir receipts/ --compact
```

Exit codes: `0` means a decision was produced. `2` means your input was malformed (nothing
was written). `3` means a receipt failed its privacy check (nothing was written).

## How it was tested

400 synthetic test cases, 267 for tuning and 133 held out. On the held-out set: every
topic label right, no contaminated session ever reused, and the router abstained on 27%
of cases. These tests prove the machinery works. They don't prove real-world accuracy,
which is exactly why the automatic answer stays locked. Run them yourself:

```sh
python3 eval/harness.py --self-test   # 60 internal checks
python3 eval/harness.py               # the 400 cases
python3 -m unittest discover -s test -p 'test_*.py' -v
python3 eval/negative_probes.py       # 5 fail-closed scenarios
```

## Details for integrators

- Declares no tools, hooks, or settings. It doesn't touch the Hermes session loop or read
  your Hermes config. Its only surface is the command line.
- Needs Python 3.11 or newer. Nothing to install beyond the plugin itself.
- Optional per-call settings via `--config` and `--calibration` JSON files. No
  configuration is required.
- Receipts append to a JSONL file in the directory you pass with `--receipts-dir`.
- MIT licensed.

## Links

- Project site: <https://aska-digital.github.io/protean-ordaprompt/>
- Rendered catalog page: <https://hermes-agent.nousresearch.com/docs/plugins/protean-ordaprompt>
- Catalog PR: [NousResearch/hermes-agent#119124](https://github.com/NousResearch/hermes-agent/pull/119124)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
