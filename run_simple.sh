# !bin bash
# export PATH=/opt/nvidia/nsight-systems/2025.2.1/target-linux-x64:$PATH

# bash run_simple.sh 
# to start nvtx: bash run_simple.sh nvtx
if [ "$1" == "nvtx" ]; then
  nsys profile --trace=cuda,nvtx \
    ./build/bin/llama-simple \
      -m /root/autodl-tmp/models/spif_pspllama.gguf \
      -n 128
else
  ./build/bin/llama-simple \
    -m /root/autodl-tmp/models/spif_pspllama.gguf \
    -n 128
fi
        

#   ./build/bin/llama-simple \
#         -m /root/autodl-tmp/models/llama-160m-GGUF/llama-160m.fp16.gguf \
#         -n 128

  # ./build/bin/llama-simple \
  #       -m /root/autodl-tmp/models/llama-160m-GGUF/prosparse-llama-2-7b.gguf \
  #       -n 128
