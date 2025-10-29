#!/bin/bash

set -x

if [ "$1" == "nvtx" ]; then
    CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx \
        ./build/bin/llama-cli \
        -m ../prosparse-llama-2-7b.gguf \
        -spif-ms ../prosparse-llama-2-7b-sparkinfer-model-split.gguf \
        -ngl 999 -cffn --no-mmap \
        -p "I believe the meaning of life is" \
        -n 32 -no-cnv -t 4 -s 42
else
    CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
        -m ../prosparse-llama-2-7b.gguf \
        -spif-ms ../prosparse-llama-2-7b-sparkinfer-model-split.gguf \
        -ngl 999 -cffn --no-mmap \
        -p "I believe the meaning of life is" \
        -n 32 -no-cnv -t 4 -s 42
fi
