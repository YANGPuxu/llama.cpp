#include "llama-sparkinfer.h"

#include "ggml-cuda.h"
#include "ggml-sparkinfer.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <numeric>

ggml_tensor * sparkinfer_layer_cache::build_reload_impl(ggml_context * ctx, const char * name) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_tensor * src_tensor = nullptr;
    ggml_tensor * dst_tensor = nullptr;

    if (strstr("up", name)) {
        src_tensor = layer_ffn_up;
        dst_tensor = ffn_up_cache;
    } else if (strstr("gate", name)) {
        src_tensor = layer_ffn_gate;
        dst_tensor = ffn_gate_cache;
    } else if (strstr("down", name)) {
        src_tensor = layer_ffn_down;
        dst_tensor = ffn_down_cache;
    } else {
        GGML_ABORT("fatal error");
    }

    memcpy(&(result->op_params[0]), this, sizeof(void *));

    result->op     = GGML_OP_RELOAD_WEIGHTS;
    result->src[0] = sparse_idx;

    return result;
}

void sparkinfer_layer_cache::reload_neurons(ggml_backend_cuda_context * ctx, ggml_tensor * dst) {}

extern "C" {
void reload_neurons(ggml_sparkinfer_layer_cache *      obj,
                    struct ggml_backend_cuda_context * ctx,
                    struct ggml_tensor *               dst) {
    auto * spif_lc = reinterpret_cast<sparkinfer_layer_cache *>(obj);
    spif_lc->reload_neurons(ctx, dst);
}
}

sparkinfer_cache_manager::sparkinfer_cache_manager(const std::string & spif_ms_path, llama_model & model) {
    ggml_context *   ctx_meta    = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    gguf_context * ctx_gguf = gguf_init_from_file(spif_ms_path.c_str(), gguf_params);

    layer_neuron_count = gguf_get_val_i64(ctx_gguf, gguf_find_key(ctx_gguf, "layer_neuron_count"));
    layer_group_count  = gguf_get_val_i64(ctx_gguf, gguf_find_key(ctx_gguf, "layer_group_count"));
    layer_group_size   = layer_neuron_count / layer_group_count;
    const auto * cache_sizes_from_gguf =
        static_cast<const int32_t *>(gguf_get_arr_data(ctx_gguf, gguf_find_key(ctx_gguf, "layer_neuron_cache_size")));

    ggml_init_params ctx_params = {
        /*.mem_size   = */ ggml_tensor_overhead() * 512,  // magic number here
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ctx_cpu = ggml_init(ctx_params);
    ctx_gpu = ggml_init(ctx_params);

    auto &     layers  = model.layers;
    const auto n_layer = model.hparams.n_layer;
    const auto n_embd  = model.hparams.n_embd;
    is_gated_mlp       = !(layers.cbegin()->ffn_gate == nullptr);
    layer_caches       = std::vector<sparkinfer_layer_cache *>(n_layer);
    layer_cache_sizes  = std::vector<int64_t>(cache_sizes_from_gguf, cache_sizes_from_gguf + n_layer);

    // for simulation, we select some layers to store all their neurons
    std::vector<int> id = std::vector<int>(n_layer);
    std::iota(id.begin(), id.end(), 0);
    std::nth_element(id.begin(), id.begin() + 32, id.end(),
                     [&](int i, int j) { return cache_sizes_from_gguf[i] > cache_sizes_from_gguf[j]; });
    std::for_each(id.begin(), id.begin() + 32, [&](int i) { layer_cache_sizes[i] = layer_neuron_count; });

    auto create_tensor = [&](ggml_context * ctx, ggml_type type, std::vector<int64_t> ne, uint32_t il,
                             const char * name) {
        char tensor_name[GGML_MAX_NAME];
        std::snprintf(tensor_name, sizeof(tensor_name), "blk.%u.%s", il, name);
        ggml_tensor * tensor_meta = ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
        return ggml_set_name(tensor_meta, tensor_name);
    };

    for (uint32_t il = 0; il < n_layer; ++il) {
        auto *     layer_cache = new sparkinfer_layer_cache();
        const auto cache_size  = layer_cache_sizes[il];

        layer_cache->layer_ffn_pred_up     = layers[il].ffn_pred_up;
        layer_cache->layer_ffn_pred_down   = layers[il].ffn_pred_down;
        layer_cache->layer_ffn_pred_up_b   = layers[il].ffn_pred_up_b;
        layer_cache->layer_ffn_pred_down_b = layers[il].ffn_pred_down_b;

        layer_cache->layer_ffn_up     = layers[il].ffn_up;
        layer_cache->layer_ffn_gate   = layers[il].ffn_gate;
        layer_cache->layer_ffn_down   = layers[il].ffn_down_t;
        layer_cache->layer_ffn_up_b   = layers[il].ffn_up_b;
        layer_cache->layer_ffn_gate_b = layers[il].ffn_gate_b;
        layer_cache->layer_ffn_down_b = layers[il].ffn_down_b;

        layer_cache->ffn_up_cache =
            create_tensor(ctx_gpu, layers[il].ffn_up->type, { n_embd, cache_size }, il, "ffn_up_cache.weight");
        if (is_gated_mlp) {
            layer_cache->ffn_gate_cache =
                create_tensor(ctx_gpu, layers[il].ffn_gate->type, { n_embd, cache_size }, il, "ffn_gate_cache.weight");
        }
        layer_cache->ffn_down_cache =
            create_tensor(ctx_gpu, layers[il].ffn_down_t->type, { n_embd, cache_size }, il, "ffn_down_cache.weight");

        layer_cache->neuron_idx  = create_tensor(ctx_gpu, GGML_TYPE_I64, { cache_size }, il, "ffn_neuron_idx");
        layer_cache->neuron_mask = create_tensor(ctx_cpu, GGML_TYPE_I64, { layer_neuron_count }, il, "ffn_neuron_mask");
        layer_cache->neuron_map  = create_tensor(ctx_cpu, GGML_TYPE_I64, { layer_neuron_count }, il, "ffn_neuron_map");

        layer_caches[il] = layer_cache;
    }
    threshold = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, layer_neuron_count);
    one       = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, layer_neuron_count);

    backend_cpu = ggml_backend_cpu_init();
    if (backend_cpu && ggml_get_first_tensor(ctx_cpu)) {
        buffers_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    }
    backend_gpu = ggml_backend_cuda_init(0);
    if (backend_gpu && ggml_get_first_tensor(ctx_gpu)) {
        buffers_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    }

    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char *  name       = gguf_get_tensor_name(ctx_gguf, i);
        ggml_tensor * src_tensor = ggml_get_tensor(ctx_meta, name);
        ggml_tensor * dst_tensor = ggml_get_tensor(ctx_cpu, name);

        const auto nbytes = ggml_nbytes(src_tensor);
        ggml_backend_tensor_set(dst_tensor, src_tensor->data, 0, nbytes);
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    auto reorder_tensor_2d = [&](ggml_tensor * tensor, std::vector<int64_t> & perm) {
        const auto n_cols        = tensor->ne[0];
        const auto n_rows        = ggml_nrows(tensor);
        const auto row_size      = ggml_row_size(tensor->type, n_cols);
        const auto row_stride    = tensor->nb[1];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        std::vector<uint8_t> src_buf(tensor_nbytes);
        std::vector<uint8_t> dst_buf(tensor_nbytes);
        ggml_backend_tensor_get(tensor, src_buf.data(), 0, tensor_nbytes);
        std::memcpy(dst_buf.data(), src_buf.data(), tensor_nbytes);
        for (int64_t new_row = 0; new_row < n_rows; ++new_row) {
            const auto old_row = perm[new_row];
            std::memcpy(dst_buf.data() + new_row * row_stride, src_buf.data() + old_row * row_stride, row_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf.data(), 0, tensor_nbytes);
    };
    auto reorder_tensor_1d = [&](ggml_tensor * tensor, std::vector<int64_t> & perm) {
        const auto n_elem        = tensor->ne[0];
        const auto elem_size     = ggml_row_size(tensor->type, 1);
        const auto elem_stride   = tensor->nb[0];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        std::vector<uint8_t> src_buf(tensor_nbytes);
        std::vector<uint8_t> dst_buf(tensor_nbytes);
        ggml_backend_tensor_get(tensor, src_buf.data(), 0, tensor_nbytes);
        std::memcpy(dst_buf.data(), src_buf.data(), tensor_nbytes);
        for (int64_t new_i = 0; new_i < n_elem; ++new_i) {
            const auto old_i = perm[new_i];
            std::memcpy(dst_buf.data() + new_i * elem_stride, src_buf.data() + old_i * elem_stride, elem_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf.data(), 0, tensor_nbytes);
    };
    auto reorder_if_exists = [&](ggml_tensor * tensor, std::vector<int64_t> & perm) {
        if (tensor) {
            GGML_ASSERT(ggml_is_contiguous(tensor));
            if (tensor->ne[1] > 1) {
                reorder_tensor_2d(tensor, perm);
            } else {
                reorder_tensor_1d(tensor, perm);
            }
        }
    };

    float total_cache_n_mega_bytes = 0.0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto *       layer_cache = layer_caches[il];
        const auto   cache_size  = layer_cache_sizes[il];
        const auto * neuron_map  = layer_cache->neuron_map;

        std::vector<int64_t> perm(layer_neuron_count);
        ggml_backend_tensor_get(neuron_map, perm.data(), 0, ggml_nbytes(neuron_map));

        // reorder_if_exists(layer_cache->layer_ffn_pred_down, perm);
        // reorder_if_exists(layer_cache->layer_ffn_pred_down_b, perm);
        // reorder_if_exists(layer_cache->layer_ffn_up, perm);
        // reorder_if_exists(layer_cache->layer_ffn_up_b, perm);
        // if (is_gated_mlp) {
        //     reorder_if_exists(layer_cache->layer_ffn_gate, perm);
        //     reorder_if_exists(layer_cache->layer_ffn_gate_b, perm);
        // }
        // reorder_if_exists(layer_cache->layer_ffn_down, perm);
        // reorder_if_exists(layer_cache->layer_ffn_down_b, perm);

        const auto cache_nbytes = ggml_nbytes(layer_cache->ffn_up_cache);
        ggml_backend_tensor_set(layer_cache->ffn_up_cache, layer_cache->layer_ffn_up->data, 0, cache_nbytes);
        if (is_gated_mlp) {
            ggml_backend_tensor_set(layer_cache->ffn_gate_cache, layer_cache->layer_ffn_gate->data, 0, cache_nbytes);
        }
        ggml_backend_tensor_set(layer_cache->ffn_down_cache, layer_cache->layer_ffn_down->data, 0, cache_nbytes);

        auto neuron_idx = std::vector<int64_t>(cache_size);
        std::iota(neuron_idx.begin(), neuron_idx.end(), 0);
        auto neuron_mask = std::vector<int64_t>(layer_neuron_count);
        std::fill_n(neuron_mask.begin(), cache_size, 1);

        ggml_backend_tensor_set(layer_cache->neuron_idx, neuron_idx.data(), 0, ggml_nbytes(layer_cache->neuron_idx));
        ggml_backend_tensor_set(layer_cache->neuron_mask, neuron_mask.data(), 0, ggml_nbytes(layer_cache->neuron_mask));

        const auto cache_n_mega_bytes = (cache_nbytes * (is_gated_mlp ? 3 : 2)) / (1024.0 * 1024.0);
        LLAMA_LOG_INFO("%s: [layer %2u] offloaded %6.2f MiB and cached %5lu (%6.2f%%) neurons to GPU\n", __func__, il,
                       cache_n_mega_bytes, cache_size, cache_size * 100.0 / layer_neuron_count);
        total_cache_n_mega_bytes += cache_n_mega_bytes;
    }
    LLAMA_LOG_INFO("%s: totally offloaded %.2f MiB neurons to GPU\n", __func__, total_cache_n_mega_bytes);
    ggml_set_f32(threshold, 0.5f);
    ggml_set_f32(one, 1.0f);
}

sparkinfer_cache_manager::~sparkinfer_cache_manager() {
    ggml_backend_buffer_free(buffers_cpu);
    ggml_free(ctx_cpu);
    ggml_backend_free(backend_cpu);

    ggml_backend_buffer_free(buffers_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_free(backend_gpu);
}
