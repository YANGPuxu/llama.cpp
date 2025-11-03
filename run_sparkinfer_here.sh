#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0

if [ "$1" == "nvtx" ]; then
    nsys profile --trace=cuda,nvtx \
        ./build/bin/llama-cli \
        -m /share/models/sparkinfer-sharing/prosparse-llama-2-7b.gguf \
        -spif-ms /share/models/sparkinfer-sharing/prosparse-llama-2-7b-sparkinfer-model-split.gguf \
        -ngl 999 -cffn --no-mmap \
        -p "I believe the meaning of life is" \
        --samplers "temperature;top_k;top_p" --temp 0.8 --top_k 40 --top-p 0.9 \
        -n 64 -no-cnv -t 4 -s 42 --no-warmup
else
    ./build/bin/llama-cli \
        -m /share/models/sparkinfer-sharing/prosparse-llama-2-7b.gguf \
        -spif-ms /share/models/sparkinfer-sharing/prosparse-llama-2-7b-sparkinfer-model-split.gguf \
        -ngl 999 -cffn --no-mmap \
        -p "I believe the meaning of life is" \
        --samplers "temperature;top_k;top_p" --temp 0.8 --top_k 40 --top-p 0.9 \
        -n 64 -no-cnv -t 4 -s 42 --no-warmup
fi
