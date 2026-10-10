#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Publish a copy of the gfx906 deployment launcher into this repo WITHOUT its API key.
#
#   tools/gfx906/export-run-vllm.sh /path/to/run-vllm              # write the copy
#   tools/gfx906/export-run-vllm.sh --check /path/to/run-vllm      # fail if it drifted
#
# The tracked copy (run-vllm.sh) is the live script with exactly three differences:
#
#   1. the literal `export VLLM_API_KEY=<secret>` line becomes
#      `export VLLM_API_KEY="${VLLM_API_KEY:?...}"`, so the key is required from the
#      environment instead of living in a git object;
#   2. the SPDX + provenance header below is prepended (after the shebang);
#   3. one `# shellcheck disable=` line, because this repo lints every .sh and the
#      live script has five benign findings (SC2012 `ls` for path discovery, SC2086
#      on `tee -a $LOG_FILE`, SC2206 on the intentional unquoted $@ pass-through).
#      The copy must stay byte-identical to what a box runs, so the suppressions are
#      applied here rather than by editing the launcher. Never add a fourth: if the
#      live script grows new findings, fix the live script instead.
#
# Everything else is byte-identical, which is the point: the copy can be diffed
# against the script a box actually runs and the diff will be that header, that
# lint-suppression line and that one key line. --check re-derives the copy and compares,
# and additionally fails if any long literal token survives in the tracked file, so
# a re-sync cannot leak a key by accident.
#
# The exporter does NOT copy anything else: local absolute paths, the log directory
# and the canary scripts the launcher calls are box-specific and stay out of the repo.
set -uo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$DEST_DIR/run-vllm.sh"

usage() {
    sed -n '4,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

CHECK=0
if [ "${1:-}" = "--check" ]; then
    CHECK=1
    shift
fi
SRC="${1:-}"
[ -n "$SRC" ] && [ -r "$SRC" ] || usage

# The header text lives here, so the copy and this script can never disagree about
# what was added.
header() {
    cat <<'HDR'
# --- COPY OF THE GFX906 DEPLOYMENT LAUNCHER (API key removed) -------------
# This file is a snapshot of the launcher that boots this box, exported by
# tools/gfx906/export-run-vllm.sh so that the flags, knobs and measurement notes
# that took weeks to accumulate travel with the source tree.
#
#   Two differences from the script a box actually runs, plus one repo-only lint
#   suppression, and only these:
#   * `export VLLM_API_KEY=...` no longer carries a literal secret - the value
#     must come from the environment (VLLM_API_KEY=... ./run-vllm.sh).
#   * this header.
#   * the shellcheck disable line below, because this repo lints every .sh and the
#     live script's five findings are benign-by-design. Nothing else was edited:
#     the copy stays byte-identical to what the box runs.
#
# It is NOT a supported entry point. Paths (/ai/models/..., logs/, scripts/*.py)
# are those of the reference box, and the preflight canary it calls is not in this
# repo. Re-export with the exporter above; `--check` fails if the copy drifted.
# Docs: docs/gfx906/running.md   Measurement log: docs/gfx906/degradation.md
# --- END COPY NOTE --------------------------------------------------------
HDR
}

redact() {
    # Replace the literal-key export with an environment-required form. Only the
    # line that assigns VLLM_API_KEY is touched; nothing else in the stream is.
    awk '
        /^([[:space:]]*)export VLLM_API_KEY=[^$]/ {
            print "export VLLM_API_KEY=\"${VLLM_API_KEY:?set VLLM_API_KEY in the environment - this copy ships without the key}\""
            next
        }
        { print }
    '
}

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
# The shebang has to stay on line 1 or the copy is not executable, so the header
# goes after it and the body is redacted from line 2 onward.
head -1 "$SRC" > "$TMP"
header >> "$TMP"
# Repo-only: this repo's shellcheck hook lints every .sh. See the header note.
printf '# shellcheck disable=SC2012,SC2086,SC2206\n' >> "$TMP"
tail -n +2 "$SRC" | redact >> "$TMP"

leak=$(grep -nE 'VLLM_API_KEY=[A-Za-z0-9_-]{20,}' "$TMP" || true)
if [ -n "$leak" ]; then
    echo "export-run-vllm: a literal key survived the redaction:" >&2
    echo "$leak" >&2
    exit 1
fi

if [ "$CHECK" = 1 ]; then
    if [ ! -r "$DEST" ]; then
        echo "export-run-vllm: no tracked copy at $DEST" >&2
        exit 1
    fi
    if cmp -s "$TMP" "$DEST"; then
        echo "export-run-vllm: copy matches $SRC (header + key redaction only)"
        exit 0
    fi
    echo "export-run-vllm: $DEST has drifted from $SRC:" >&2
    diff -u "$DEST" "$TMP" | head -40 >&2
    exit 1
fi

cp "$TMP" "$DEST"
chmod 0755 "$DEST"
echo "export-run-vllm: wrote $DEST from $SRC"
echo "export-run-vllm: $(grep -c '' "$DEST") lines; key lines removed: $(grep -cE 'VLLM_API_KEY=[A-Za-z0-9_-]{20,}' "$SRC" || true)"
