#!/usr/bin/env bash
# Test harness for the two run-vllm changes that decide what a boot is configured
# with. Neither is testable by running ./run-vllm -- that boots the engine this
# agent runs on -- so both blocks are EXTRACTED and run against stubs:
#
#   1. the OPTIMIZED bundle     (between the BEGIN/END OPTIMIZED BUNDLE markers)
#      arms: default, bundle on, explicit-override, set-but-empty, garbage value,
#      and one arm that proves the bundle EXPORTS (a child process must see it,
#      because the engine is a child and not a subshell of this script).
#   2. boot_once building its command in an array instead of a backslash
#      continuation. The engine command line is asserted token by token against a
#      golden list, so a dropped flag fails with a diff instead of at 04:00.
#      Includes a RED CONTROL: a deliberately mutated launcher must FAIL.
#
#   tools/gfx906/test-optimized-bundle.sh [path-to-launcher]
#      default: the in-tree copy, tools/gfx906/run-vllm.sh
# The golden argv names the reference box's model and chat-template paths; on a
# different box those two golden lines fail by construction, not by regression.
# Exit 0 = every arm behaved as documented.
set -uo pipefail

# In-tree the launcher copy sits beside this script and is named run-vllm.sh; on the
# reference box it is ../run-vllm, so pass a path if you are testing the live script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${1:-$HERE/run-vllm.sh}"
[ -r "$LAUNCHER" ] || { echo "cannot read launcher: $LAUNCHER" >&2; exit 2; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fails=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

# ---------------------------------------------------------------- bundle block
sed -n '/^# BEGIN OPTIMIZED BUNDLE/,/^# END OPTIMIZED BUNDLE/p' "$LAUNCHER" \
    > "$TMP/bundle.sh"
[ -s "$TMP/bundle.sh" ] || { echo "no OPTIMIZED BUNDLE markers in $LAUNCHER" >&2; exit 2; }

cat > "$TMP/bundle_runner.sh" <<'EOS'
# One shell does eval + source + print, because OPT_SRC is an associative array
# and those cannot be exported to a child. The child is still probed separately:
# the engine is a child process, so `export` inside the bundle is load-bearing.
eval "$BUNDLE_ENV"
source "$BUNDLE"
for k in PLE_RANDOM PLE_PREFETCH GC_FREEZE DRAFTER_GRAPHS; do
    if [ -z "${!k+set}" ]; then v="<unset>"
    elif [ -z "${!k}" ];       then v="<empty>"
    else                            v="${!k}"
    fi
    printf '%s=%s|%s\n' "$k" "$v" "${OPT_SRC[$k]}"
done
bash -c 'printf "child_PLE_RANDOM=%s\n" "${PLE_RANDOM-<unset>}"'
EOS

# run_bundle <env assignments> ...   ->  canonical output on stdout
run_bundle() {
    env -u OPTIMIZED -u PLE_RANDOM -u PLE_PREFETCH -u GC_FREEZE -u DRAFTER_GRAPHS \
        BUNDLE="$TMP/bundle.sh" RUNNER="$TMP/bundle_runner.sh" BUNDLE_ENV="$*" \
        bash "$TMP/bundle_runner.sh" 2>"$TMP/err"
}

expect_bundle() {   # expect_bundle <label> <expected> <env...>
    local label="$1" exp="$2"; shift 2
    local got; got=$(run_bundle "$@")
    if [ "$got" = "$exp" ]; then ok "$label"
    else bad "$label"
        printf '        got:\n%s\n        want:\n%s\n' "$got" "$exp"
    fi
}

echo "== OPTIMIZED bundle ($LAUNCHER)"

expect_bundle "default: no OPTIMIZED, no knobs -> all default-off, nothing exported" \
'PLE_RANDOM=<unset>|default-off
PLE_PREFETCH=<unset>|default-off
GC_FREEZE=<unset>|default-off
DRAFTER_GRAPHS=<unset>|default-off
child_PLE_RANDOM=<unset>'

expect_bundle "OPTIMIZED=1 -> all four on, provenance=bundle, EXPORTED to a child" \
'PLE_RANDOM=1|bundle
PLE_PREFETCH=1|bundle
GC_FREEZE=1|bundle
DRAFTER_GRAPHS=1|bundle
child_PLE_RANDOM=1' \
'export OPTIMIZED=1'

expect_bundle "explicit knob beats the bundle (single-variable A/B still possible)" \
'PLE_RANDOM=1|bundle
PLE_PREFETCH=0|explicit(0)
GC_FREEZE=1|bundle
DRAFTER_GRAPHS=1|bundle
child_PLE_RANDOM=1' \
'export OPTIMIZED=1 PLE_PREFETCH=0'

expect_bundle "set-but-empty is NOT overwritten with 1 (bare VAR= means off)" \
'PLE_RANDOM=1|bundle
PLE_PREFETCH=<empty>|explicit(<empty>)
GC_FREEZE=1|bundle
DRAFTER_GRAPHS=1|bundle
child_PLE_RANDOM=1' \
'export OPTIMIZED=1 PLE_PREFETCH='

expect_bundle "OPTIMIZED=0 with one explicit knob works without the bundle" \
'PLE_RANDOM=<unset>|default-off
PLE_PREFETCH=<unset>|default-off
GC_FREEZE=1|explicit(1)
DRAFTER_GRAPHS=<unset>|default-off
child_PLE_RANDOM=<unset>' \
'export OPTIMIZED=0 GC_FREEZE=1'

garbage_out=$(run_bundle 'export OPTIMIZED=yes')
if [ "$garbage_out" = "$(printf '%s\n' \
    'PLE_RANDOM=<unset>|default-off' 'PLE_PREFETCH=<unset>|default-off' \
    'GC_FREEZE=<unset>|default-off' 'DRAFTER_GRAPHS=<unset>|default-off' \
    'child_PLE_RANDOM=<unset>')" ]; then
    ok "OPTIMIZED=yes falls back to the known-good defaults"
else
    bad "OPTIMIZED=yes must behave exactly like OPTIMIZED=0"
    printf '        got:\n%s\n' "$garbage_out"
fi
if grep -q "not 0 or 1" "$TMP/err"; then
    ok "OPTIMIZED=yes says so on stderr instead of silently ignoring the operator"
else
    bad "OPTIMIZED=yes must warn on stderr (operator asked for something they did not get)"
fi

# ------------------------------------------------------------------- boot_once
sed -n '/^boot_once() {$/,/^}/p' "$LAUNCHER" > "$TMP/boot_once.part"
[ -s "$TMP/boot_once.part" ] || { echo "no boot_once() found in $LAUNCHER" >&2; exit 2; }

cat > "$TMP/argv_runner.sh" <<'EOS'
# A stub that records the command name and the argv it was called with, one token
# per line, so the assertions below can pin the engine command line exactly.
# ${FUNCNAME[0]} supplies the command name, which "$@" alone cannot.
vllm() { printf '%s\n' "${FUNCNAME[0]}" "$@" > "$OUT"; }
TP=4; MAXLEN=147456; MAXSEQS=3; MBT=4096
comp_cfg='{"mode": 3, "cudagraph_mode": 1, "cudagraph_capture_sizes": [4,8,12]}'
graph_args=()
jit_args=(--env A=1)
spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
mm_args=(--mm-processor-kwargs '{"min_pixels":16}')
kv_args=()
LOG_FILE="$LOG"
source "$PART"
boot_once --tail-flag 'two words'
EOS

EXPECTED_ARGV=(
    vllm serve /ai/models/hf.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
    --served-model-name vllm-chat
    --host 0.0.0.0 --port 9000
    --tensor-parallel-size 4
    --dtype float16
    --max-model-len 147456
    --max-num-seqs 3
    --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.95
    --compilation-config '{"mode": 3, "cudagraph_mode": 1, "cudagraph_capture_sizes": [4,8,12]}'
    --no-async-scheduling
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --generation-config auto
    --chat-template /ai/models/hf.co/froggeric/Qwen-Fixed-Chat-Templates/chat_template.jinja
    --env A=1
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --mm-processor-kwargs '{"min_pixels":16}'
    --tail-flag two words
)

check_argv() {   # check_argv <launcher> <label> ; sets global ARGV_DIFF
    local L="$1"
    sed -n '/^boot_once() {$/,/^}/p' "$L" > "$TMP/part"
    : > "$TMP/argv.log"
    OUT="$TMP/argv.out" LOG="$TMP/argv.log" PART="$TMP/part" \
        bash "$TMP/argv_runner.sh" 2>"$TMP/argv_err"
    ARGV_DIFF=$(diff <(printf '%s\n' "${EXPECTED_ARGV[@]}") "$TMP/argv.out" 2>&1)
}

echo "== boot_once argv"
if check_argv "$LAUNCHER"; then :; fi
if [ -z "$ARGV_DIFF" ]; then
    ok "engine argv matches the golden command line, incl. \"\$@\" pass-through"
else
    bad "engine argv changed:"
    printf '%s\n' "$ARGV_DIFF" | sed 's/^/        /'
fi
if grep -q '^run-vllm: engine argv: vllm serve ' "$TMP/argv.log"; then
    ok "the argv is written into the log file before the engine starts"
else
    bad "boot_once did not write 'engine argv:' to \$LOG_FILE -- the log must state what it ran"
fi

# RED CONTROL: drop one flag from the copy. If the harness still says ok, the
# golden assertion above is not evidence of anything.
sed 's/^    --no-async-scheduling$/    --async-scheduling-disabled/' "$LAUNCHER" > "$TMP/mutant"
if grep -q -- '--no-async-scheduling' "$TMP/mutant"; then
    bad "red control: the mutation did not apply (pattern did not match $LAUNCHER) -- nothing can be concluded"
else
    check_argv "$TMP/mutant"
    if [ -n "$ARGV_DIFF" ]; then
        ok "red control: a dropped engine flag IS caught by the golden list"
    else
        bad "red control: the argv assertion passed a launcher missing --no-async-scheduling"
    fi
fi

# Cross-check against the pre-refactor launcher when it is still around: the
# argv refactor must be a pure no-op for the command line.
if [ -n "${ARGV_BASELINE:-}" ] && [ -r "$ARGV_BASELINE" ]; then
    echo "== argv equivalence vs $ARGV_BASELINE"
    check_argv "$ARGV_BASELINE"
    base_diff="$ARGV_DIFF"
    check_argv "$LAUNCHER"
    if [ "$base_diff" = "$ARGV_DIFF" ]; then
        ok "new boot_once produces byte-identical argv to the pre-refactor one"
    else
        bad "new boot_once differs from the pre-refactor one"
    fi
fi

echo
echo "== boot manifest"
sed -n '/^# BEGIN BOOT MANIFEST/,/^# END BOOT MANIFEST/p' "$LAUNCHER" > "$TMP/manifest.part"
[ -s "$TMP/manifest.part" ] || { echo "no BOOT MANIFEST markers in $LAUNCHER" >&2; exit 2; }

cat > "$TMP/manifest_runner.sh" <<'EOS'
# The real launcher resolves these before it calls write_manifest; stub them here
eval "$MANIFEST_ENV"
source "$BUNDLE"
TP=4 MAXLEN=147456 MAXSEQS=3 MBT=4096 GRAPHMODE=piecewise
SPEC='{"method":"mtp","num_speculative_tokens":3}'
NCCL_BLOCKING_WAIT=1 BOOT_TRIES=3 PREFLIGHT=0 TVMFFI=disable TEXT_ONLY=1
LOG_FILE="$LOG"
source "$PART"
write_manifest
EOS

man_arm() {   # man_arm <label> <env> <regex that must match>
    local label="$1" env_="$2" re="$3" got
    : > "$TMP/man.log"
    got=$(MANIFEST_ENV="$env_" BUNDLE="$TMP/bundle.sh" PART="$TMP/manifest.part" \
        LOG="$TMP/man.log" bash "$TMP/manifest_runner.sh" 2>&1)
    if printf '%s\n' "$got" | grep -Eq "$re"; then ok "$label"
    else bad "$label"; printf '        wanted /%s/ in:\n%s\n' "$re" "$got" | sed 's/^/        /'
    fi
    # Silent on success: tee means the terminal and the log get the same block, and
    # a manifest that only reached the tty is the bug this whole check exists for.
    if ! grep -q 'BOOT MANIFEST' "$TMP/man.log"; then
        bad "  manifest did not reach \$LOG_FILE -- the log must record its own launch"
    fi
}

# The four knobs print on four separate lines, so match them one at a time.
for k in PLE_RANDOM PLE_PREFETCH GC_FREEZE DRAFTER_GRAPHS; do
    man_arm "manifest lists $k = 1 [bundle] under OPTIMIZED=1" 'export OPTIMIZED=1' \
        "${k}[[:space:]]+=[[:space:]]+1[[:space:]]+\\[bundle\\]"
done
man_arm "an explicit override is visible as such in the log" \
    'export OPTIMIZED=1 PLE_PREFETCH=0' \
    'PLE_PREFETCH[[:space:]]+=[[:space:]]+0[[:space:]]+\[explicit\(0\)\]'
man_arm "defaults are listed as default-off (an absent knob is a finding)" '' \
    'PLE_RANDOM[[:space:]]+=[[:space:]]+0[[:space:]]+\[default-off\]'
man_arm "NCCL variables are reported even when unset" 'export OPTIMIZED=1' \
    'proto=<unset>.*not set here'
man_arm "the engine shape is in the log (TP/maxlen/maxseqs/batched/graphmode)" \
    'export OPTIMIZED=1' \
    'engine: TP=4 maxlen=147456 maxseqs=3 max-batched=4096 graphmode=piecewise'
man_arm "OPTIMIZED itself is stated, so a reader knows which config this log is" \
    'export OPTIMIZED=1' 'OPTIMIZED=1 .{0,3}.0 = known-good defaults'

echo
if [ "$fails" = 0 ]; then echo "ALL ARMS PASSED"; else echo "$fails ARM(S) FAILED"; fi
exit "$((fails > 0))"
