#!/bin/bash

set -x

env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
    gdb --args ./build/bin/llama-cli \
    -m ../prosparse-llama-2-7b.gguf \
    -spif-ms ../prosparse-llama-2-7b-sparkinfer-model-split.gguf \
    -ngl 999 -cffn --no-mmap -p "I believe the meaning of life is" \
    -n 32 -no-cnv -t 4 -s 42
