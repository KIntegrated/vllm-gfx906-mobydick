#!/bin/bash
# gfx906 full benchmark: upstream0.23 (dense only) vs our0.26 vs main.
set -u
BENCH=/tmp/bench
HFM=/data/cache/huggingface
mkdir -p "$BENCH"
cp "$(dirname "$0")/_bench_gfx906.py" "$BENCH/_b.py"

PP="${BENCH_PP:-2048}"; TG="${BENCH_TG:-256}"; UTIL="${BENCH_GPU_UTIL:-0.85}"; ML="${BENCH_MAXLEN:-3328}"

clean() {
  for pid in $(ps -eo pid,args | grep -iE "VLLM::EngineCore|/bench/_b.py" | grep -v grep | awk '{print $1}'); do
    docker run --rm --pid=host --privileged docker.io/library/busybox:latest kill -9 "$pid" 2>/dev/null
  done
  sleep 2
}

job() {
  local img="$1" over="$2" model="$3" tag="$4"
  echo "==================== [$tag] $model (pp=$PP tg=$TG util=$UTIL maxlen=$ML) ===================="
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 993 --group-add 44 \
    -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    $( [ -n "$over" ] && echo "-e $over" ) \
    -v "$HFM:/root/.cache/huggingface:ro" \
    -v "$BENCH:/bench" \
    --entrypoint bash \
    "$img" -c "BENCH_PP=$PP BENCH_TG=$TG BENCH_GPU_UTIL=$UTIL BENCH_MAXLEN=$ML python3 -u /bench/_b.py '$model'" 2>&1 | \
      grep -E "BENCH:|BENCH warmup|RuntimeError|ValueError|Error|AttributeError|HFValidation|VLLMValidation" | grep -viE "WARNING 0|disable_log" | tail -5
  echo "------------------- /done [$tag] -------------------"
  clean
}

# --- Dense (all 3 images) ---
job "aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0" \
    "HSA_OVERRIDE_GFX_VERSION=9.0.6" "QuantTrio/Qwen3.5-9B-AWQ" "0.23-dense"
job "mixa3607/vllm-gfx906:0.26.0-rocm-7.2.1-kintegrated" \
    "HSA_OVERRIDE_GFX_VERSION=9.0.6" "QuantTrio/Qwen3.5-9B-AWQ" "0.26-dense"
job "mixa3607/vllm-gfx906:0.27.99rc0-rocm-7.14-kintegrated" \
    "" "QuantTrio/Qwen3.5-9B-AWQ" "main-dense"

# --- MoE (0.26 + main; 0.23 unsupported) ---
job "mixa3607/vllm-gfx906:0.26.0-rocm-7.2.1-kintegrated" \
    "HSA_OVERRIDE_GFX_VERSION=9.0.6" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "0.26-MoE"
job "mixa3607/vllm-gfx906:0.27.99rc0-rocm-7.14-kintegrated" \
    "" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "main-MoE"

echo "ALL DONE"