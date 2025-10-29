#pragma once

#include "llama-model.h"

#include <string>
#include <vector>

struct ggml_backend_cuda_context;

struct sparkinfer_layer_cache {
    ggml_tensor * layer_ffn_pred_up     = nullptr;
    ggml_tensor * layer_ffn_pred_down   = nullptr;
    ggml_tensor * layer_ffn_pred_up_b   = nullptr;
    ggml_tensor * layer_ffn_pred_down_b = nullptr;

    ggml_tensor * layer_ffn_up     = nullptr;
    ggml_tensor * layer_ffn_gate   = nullptr;
    ggml_tensor * layer_ffn_down   = nullptr;
    ggml_tensor * layer_ffn_up_b   = nullptr;
    ggml_tensor * layer_ffn_gate_b = nullptr;
    ggml_tensor * layer_ffn_down_b = nullptr;

    ggml_tensor * ffn_up_cache   = nullptr;
    ggml_tensor * ffn_gate_cache = nullptr;
    ggml_tensor * ffn_down_cache = nullptr;

    ggml_tensor * sparse_idx  = nullptr;
    ggml_tensor * neuron_mask = nullptr;
    ggml_tensor * neuron_idx  = nullptr;
    ggml_tensor * neuron_map  = nullptr;

    ggml_tensor * dfr_score = nullptr;
    float         dfr_decay = 0.85f;

    sparkinfer_layer_cache()  = default;
    ~sparkinfer_layer_cache() = default;

    ggml_tensor * build_reload_impl(ggml_context * ctx, const char * name);
    void          reload_neurons(ggml_backend_cuda_context * ctx, ggml_tensor * dst);
};

struct sparkinfer_cache_manager {
    int64_t layer_neuron_count;
    int64_t layer_group_count;
    int64_t layer_group_size;

    bool          is_gated_mlp = false;
    ggml_tensor * threshold    = nullptr;
    ggml_tensor * one          = nullptr;

    ggml_context *        ctx_cpu     = nullptr;
    ggml_context *        ctx_gpu     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_t        backend_gpu = nullptr;
    ggml_backend_buffer_t buffers_cpu = nullptr;
    ggml_backend_buffer_t buffers_gpu = nullptr;

    std::vector<sparkinfer_layer_cache *> layer_caches;
    std::vector<int64_t>                  layer_cache_sizes;

    sparkinfer_cache_manager(const std::string & spif_ms_path, llama_model & model);
    ~sparkinfer_cache_manager();
};
