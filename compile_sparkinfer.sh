#!/bin/bash

set -x

if [ "$1" == "release" ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF
elif [ "$1" == "nvtx" ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_DEBUG=ON \
        -DCMAKE_C_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include" \
        -DCMAKE_CXX_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include" \
        -DCMAKE_CUDA_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
else
    cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_DEBUG=ON
fi

if [ "$1" == "release" ]; then
    cmake --build build --config Release -j --target llama-cli
else
    cmake --build build --config Debug -j --target llama-cli
fi
