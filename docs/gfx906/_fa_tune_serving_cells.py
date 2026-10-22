"""Measure the shapes the engine actually presents (from the debug log)."""
import sys
sys.path.insert(0, "docs/gfx906")
import torch
from _fa_tune import KVSPLIT_CANDIDATES, measure_gated, write_record
from vllm.gfx906_fa.tuning import FAConfig, cache_path, identity, static_config

HQ, HKV, D = 8, 1, 256          # 35B at TP=2 (per-rank)
GQA = HQ // HKV
# (Sq_pad, Sk_pad, batch) as observed in the serving debug log
# Decode shapes for a [1,2,3,4] capture ladder: Sq pads to 1 or 2 and the batch
# to the capture size, so every (Sq, batch) in the ladder is reachable.
CELLS = [
    (sq, 32768, b)
    for sq in (1, 2)
    for b in (1, 2, 3, 4)
] + [
    (4096, 8192, 1), (4096, 4096, 1), (3072, 8192, 1),
]
device = torch.cuda.get_device_name(torch.cuda.current_device())
for sq, sk, b in CELLS:
    base = static_config(sq, GQA, batch=b)
    times, report, attempt = measure_gated(
        b, HQ, HKV, D, sk, sq, base.nc2, KVSPLIT_CANDIDATES, 25, 3.0
    )
    if times is None:
        print(f"sq{sq}-sk{sk}-b{b}: DISCARDED ({report})")
        continue
    best = min(times, key=lambda k: times[k])
    rule_ms = times[base.kvsplit]
    delta = 100.0 * (rule_ms - times[best]) / rule_ms
    tag = ""
    if best != base.kvsplit and delta >= 1.0:
        write_record(device, "float16", HQ, HKV, D, sq, sk, b,
                     FAConfig(nc2=base.nc2, kvsplit=best, clip=base.clip))
        tag = "  RECORDED"
    print(f"sq{sq}-sk{sk}-b{b}: y8={times[8]:.3f} y16={times[16]:.3f} "
          f"y32={times[32]:.3f} rule=y{base.kvsplit} best=y{best} "
          f"({delta:+.1f}%){tag} [{report}]")
print("=== CELLS DONE")
