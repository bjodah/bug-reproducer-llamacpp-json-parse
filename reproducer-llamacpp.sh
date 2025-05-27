#!/bin/bash
if ! which llama-server >/dev/null; then >&2 echo "Make sure (a debug build of) llama-server is on \$PATH"; exit 1; fi
set -x
PORT=7476
URL_BASE=http://localhost:$PORT

( sleep 1;
  while ! curl -s "$URL_BASE/health" 2>/dev/null | grep '"ok"'; do
      sleep 1.0s;
  done;
  curl -s -X POST "$URL_BASE/v1/chat/completions" \
       -H "Content-Type: application/json" \
       -d @data3.json;
) &

gdb -ex r -args \
    llama-server \
    --port $PORT \
    --verbose \
    --log-file /tmp/reproducer-llamacpp.log \
    --hf-repo bartowski/Qwen_Qwen3-14B-GGUF:Q8_0 \
    --n-gpu-layers 999 \
    --jinja \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn \
    --ctx-size 32768 \
    --samplers 'top_k;dry;min_p;temperature;top_p' \
    --min-p 0.005 \
    --top-p 0.97 \
    --top-k 40 \
    --temp 0.7 \
    --dry-multiplier 0.7 \
    --dry-allowed-length 4 \
    --dry-penalty-last-n 2048 \
    --presence-penalty 0.05 \
    --frequency-penalty 0.005 \
    --repeat-penalty 1.01 \
    --repeat-last-n 16
