#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# gfx906-perf -- pin or release the MI50 DPM performance level.
#
# Why this exists: on this box the memory clock sits at 350 MHz (idle state) or
# 800 MHz for bursty work and only reaches 1000 MHz under sustained load. A long
# prefill measured 39.84 s at 350 MHz and 4.36 s at 800 MHz for identical work,
# so leaving the governor in "auto" makes long-context prefill measurements (and
# first-request latency) depend on luck. See docs/gfx906/DEVLOG-kv-formats.md and
# /local/git/AGENTS.md.
#
# Usage:
#   gfx906-perf on       pin every GPU to the "high" performance level
#   gfx906-perf off      return every GPU to "auto"
#   gfx906-perf status   print level + active memory/engine clock per GPU
#
# Designed to be granted via sudoers for on/off, so it accepts exactly one word
# from a fixed set and never evaluates user input. It writes only the two sysfs
# attributes named below.
set -u

LEVEL_ATTR="power_dpm_force_performance_level"
MCLK_ATTR="pp_dpm_mclk"
SCLK_ATTR="pp_dpm_sclk"

cards() {
    # Every DRM card that exposes the DPM level attribute (this box has card1 and
    # card2 for the two MI50s; card numbering is not stable across reboots, so
    # glob rather than hardcode).
    for attr in /sys/class/drm/card*/device/$LEVEL_ATTR; do
        [ -e "$attr" ] || continue
        printf '%s\n' "${attr%/$LEVEL_ATTR}"
    done
}

status() {
    for dev in $(cards); do
        card=$(basename "$(dirname "$dev")")
        level=$(cat "$dev/$LEVEL_ATTR" 2>/dev/null || echo "?")
        mclk=$(grep '\*' "$dev/$MCLK_ATTR" 2>/dev/null | sed 's/^ *//' || true)
        sclk=$(grep '\*' "$dev/$SCLK_ATTR" 2>/dev/null | sed 's/^ *//' || true)
        printf '%-6s level=%-8s mclk=%-16s sclk=%s\n' \
            "$card" "$level" "${mclk:-?}" "${sclk:-?}"
    done
}

set_level() {
    target=$1
    ok=0
    for dev in $(cards); do
        if printf '%s\n' "$target" > "$dev/$LEVEL_ATTR" 2>/dev/null; then
            ok=$((ok + 1))
        else
            printf 'failed to write %s to %s/%s (need root)\n' \
                "$target" "$dev" "$LEVEL_ATTR" >&2
        fi
    done
    if [ "$ok" -eq 0 ]; then
        echo "no GPU accepted the level change" >&2
        return 1
    fi
    sleep 1
    status
}

case "${1:-}" in
    on)     set_level high ;;
    off)    set_level auto ;;
    status) status ;;
    *)
        echo "usage: $0 on|off|status" >&2
        exit 2
        ;;
esac
