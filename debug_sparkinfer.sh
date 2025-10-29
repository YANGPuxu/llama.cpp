#!/bin/bash

set -x

env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
    gdb --args ./build/bin/llama-cli \
    -m ../prosparse-llama-2-7b.gguf \
    -spif-ms ../prosparse-llama-2-7b-sparkinfer-model-split.gguf \
    -ngl 999 -cffn --no-mmap -p "I believe the meaning of life is" \
    --samplers "temperature;top_k;top_p" --temp 0.8 --top_k 40 --top-p 0.9 \
    -n 64 -no-cnv -t 1 -s 42 --no-warmup -v
