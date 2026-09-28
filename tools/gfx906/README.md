# gfx906 tooling

## `run-vllm.sh` — a copy of the reference-box launcher, with the API key removed

`run-vllm.sh` is the script that boots the 4×MI50 (gfx906) reference box for
`Qwen3.8-Flash-Next-AWQ-INT4`: TP=4, MTP spec decode, piecewise graphs, and every
knob that came out of the decode/prefill investigation (`OPTIMIZED`, `PLE_RANDOM`,
`PLE_PREFETCH`, `GC_FREEZE`, `DRAFTER_GRAPHS`, `NCCL_ENVELOPE`, `PREFLIGHT`, …)
documented in its own comments, plus the boot manifest it writes into every log.

It is checked in because the flags and the measurement notes behind them took weeks
to accumulate and they are useless if they only exist on one machine.

**It is not a supported entry point.** Paths (`/ai/models/…`, `logs/`,
`scripts/preflight-nccl.py`) belong to the reference box, and the canary it calls is
not in this repo. Read it; do not run it elsewhere.

### What was removed

Exactly one line changes; a header and one lint line are added:

```diff
-export VLLM_API_KEY=<64-char secret>
+export VLLM_API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY in the environment - this copy ships without the key}"
```

The copy therefore refuses to boot until you supply the key yourself:

```bash
VLLM_API_KEY=... ./run-vllm.sh
```

Nothing else was edited — no reformatting, no path rewriting — so `diff` against the
live script is meaningful. The only additions are the provenance header and a
`# shellcheck disable=SC2012,SC2086,SC2206` line (this repo lints every `.sh`; the
live script's five findings are benign by design, and silencing them in the copy
beats diverging the copy from what a box actually runs). Verify the claim instead of
trusting it:

```bash
tools/gfx906/export-run-vllm.sh --check /path/to/live/run-vllm
tools/gfx906/test-optimized-bundle.sh            # 20 arms against the in-tree copy
```

`--check` re-derives the copy from the live script and compares; it also fails if any
long literal token survives in the tracked file, so a careless re-sync cannot leak a
key. To publish a newer launcher: run the exporter without `--check`, then
`test-optimized-bundle.sh` (argv golden test + the bundle precedence arms) before
committing.

### `test-optimized-bundle.sh` — the launcher's own harness

Neither the `OPTIMIZED` bundle nor `boot_once`'s argv can be tested by booting the
launcher (on the reference box that restarts the engine serving the agent doing the
testing), so the harness **extracts** the two marked blocks and runs them against
stubs. 20 arms: bundle precedence (default / on / explicit override / set-but-empty /
garbage / export-to-child), the engine command line asserted token by token against a
golden list, and a red control — a deliberately mutated launcher has to fail, so a
green run means something. Pass a path to test a different launcher.

The golden argv names the reference box's model and chat-template paths, so on another
box those two arms fail by construction rather than by regression.

### Related

- `docs/gfx906/running.md` §1b — how this relates to the docker recipes.
- `docs/gfx906/degradation.md` — the event log every knob in the script came from.
- `docs/gfx906/tp_decode_investigation.md`, `docs/gfx906/oom-256k-prefill.md` — the
  decode-bound and long-context measurements the knobs respond to.
- Note: comments inside `run-vllm.sh` cite `docs/prefill-io-wall.md` and
  `docs/decode-perf-suggestions.md` by the paths they have **on the reference box**
  (`/ai/engines/gfx906/unverbraucht/docs/`). Those two write-ups are not (yet) in this
  repo, so those citations do not resolve here.
