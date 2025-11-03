#include "llama-sparkinfer.h"

#include "ggml-cuda.h"
#include "ggml-sparkinfer.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <numeric>

// extern "C" {
// void reload_neurons(ggml_sparkinfer_layer_cache *      obj,
//                     struct ggml_backend_cuda_context * ctx,
//                     struct ggml_tensor *               dst) {
//     auto * spif_lc = reinterpret_cast<sparkinfer_layer_cache *>(obj);
//     spif_lc->reload_neurons(ctx, dst);
// }
// }

// ----------Spakinfer reload (graph building)----------- //
ggml_tensor * sparkinfer_layer_cache:: build_reload_plan(ggml_context * ctx, ggml_tensor * sparse_idx){
    GGML_ASSERT(sparse_idx && "sparse_idx is required for reloading");

    // a demo tensor for graph building
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    
    // added reload params
    // We pass &this instead of this because we want to pass the value of this (i.e., the pointer itself), not the content of the object that this pointer points to.
    auto * this_ptr = this;
    memcpy(&result->op_params[0], &this_ptr, sizeof(sparkinfer_layer_cache*)); // Pass the pointer to sparkinfer_layer_cache

    result->op = GGML_OP_RELOAD_PLAN;
    result->src[0] = sparse_idx;
    return result;
}

ggml_tensor * sparkinfer_layer_cache:: build_reload_exec(ggml_context * ctx, ggml_tensor * plan_done, const char * name){
    GGML_ASSERT(neuron_idx && "neuron_idx is required for reloading");
    GGML_ASSERT(plan_done && "plan_done is required for reloading");

    ggml_tensor * gpu_ffn = nullptr;
    ggml_tensor * cpu_ffn = nullptr;

    if (std::string(name) == "gate") {
        gpu_ffn = ffn_gate_cache;
        cpu_ffn = layer_ffn_gate;
    } else if (std::string(name) == "up") {
        gpu_ffn = ffn_up_cache;
        cpu_ffn = layer_ffn_up;
    } else if (std::string(name) == "down") {
        gpu_ffn = ffn_down_cache;
        cpu_ffn = layer_ffn_down;
    } else {
        GGML_ASSERT(false && "unsupported name for reload");
    }

    // ggml_tensor * result = ggml_view_2d(ctx, ffn_up_cache, ffn_up_cache->ne[0], ffn_up_cache->ne[1], 0, 0);
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    // added reload params
    memcpy(&result->op_params[0], this, sizeof(sparkinfer_layer_cache*)); // Pass the pointer to SparkInfer_layer_cache
    
    result->op = GGML_OP_RELOAD_EXEC;
    result->src[0] = gpu_ffn;
    result->src[1] = cpu_ffn;
    result->src[2] = plan_done; // just a dependency

    return result;
}
// --------------------------------------------------------- //

// this is a C wrapper to call llama-sparkinfer function, used in ggml codebase for reloading operation
extern "C" {  
    bool ggml_spif_reload_plan(ggml_spif_context* ctx, ggml_tensor * tensor) {  
        sparkinfer_layer_cache * spif_cache = reinterpret_cast<sparkinfer_layer_cache*>(ctx);  
        return spif_cache->spif_reload_plan(tensor);
    }

    bool ggml_spif_reload_exec(ggml_spif_context* ctx, ggml_tensor * tensor) {  
        sparkinfer_layer_cache * spif_cache = reinterpret_cast<sparkinfer_layer_cache*>(ctx);  
        return spif_cache->spif_reload_exec(tensor);
    }
}

bool sparkinfer_layer_cache::spif_reload_plan(ggml_tensor * tensor){
    // [YPX] [B] Do we really need to use it? We already have it as a class member.
    struct ggml_tensor * unused_sparse_idx      = tensor->src[0];
    // 确保 sparse_idx 的后端在 CPU 上
    GGML_ASSERT(ggml_backend_buffer_is_host(sparse_idx->buffer) && "[Error] sparse_idx tensor must be on CPU backend before planning reload.");
    // 1. 第一步：执行 `_spif_reload_plan_count_activated_neurons()`，得到 `group_activated_neurons_count` 数组
    // 2. 第二步：执行 `_spif_reload_plan_get_groups_to_ensure()`，得到 `groups_to_ensure` 数组
    // 3. 第三步：执行 `_spif_reload_plan_compute_diff_and_update_state()`，计算差集并更新状态
    _spif_reload_plan_compute_diff_and_update_state(_spif_reload_plan_get_groups_to_ensure(_spif_reload_plan_count_activated_neurons()));
    return true;
}

bool sparkinfer_layer_cache::spif_reload_exec(ggml_tensor * tensor) {

    const int io_stream_id = 1;

    // 1. Get the "Reload Plan" from 'this' (spif_cache)
    std::vector<int>& groups_to_reload = this->plan_result.groups_to_reload;
    std::vector<int>& slots_for_evict  = this->plan_result.slots_for_evict;
    GGML_ASSERT(groups_to_reload.size() == slots_for_evict.size() && "groups_to_reload and slots_for_evict must have the same size");

    // 2. From 'tensor' (the operator node), get the "Reload Targets"
    struct ggml_tensor * gpu_cache_tensor = tensor->src[0]; // e.g., ffn_up_cache
    struct ggml_tensor * cpu_weight_tensor = tensor->src[1]; // e.g., layer_ffn_up

    // 3. Calculate parameters needed for offsets
    const size_t element_size = ggml_type_size(cpu_weight_tensor->type); // e.g., 2 (for F16)
    const int64_t neurons_per_group = this->layer_group_size;
    const int64_t ffn_dim_in = cpu_weight_tensor->ne[0]; // e.g., 4096
    
    // The size of one neuron group in bytes
    const size_t group_size_bytes = neurons_per_group * ffn_dim_in * element_size; 

    char* cpu_base_ptr = (char *)cpu_weight_tensor->data;
    char* gpu_base_ptr = (char *)gpu_cache_tensor->data;

    // 5. H2D Copy for each group to reload
    for (size_t i = 0; i < groups_to_reload.size(); ++i) {
        int group_id = groups_to_reload[i];
        int slot_id  = slots_for_evict[i];

        size_t group_offset_bytes = (size_t)group_id * group_size_bytes;
        size_t slot_offset_bytes  = (size_t)slot_id  * group_size_bytes;

        void * src_addr = (void *)(cpu_base_ptr + group_offset_bytes);
        // void * dst_addr = (void *)(gpu_base_ptr + slot_offset_bytes);

        GGML_ASSERT(this->gpu_backend != nullptr && "GPU backend must be initialized for async copy");

        ggml_backend_tensor_set_async_stream(
            this->gpu_backend,    // ggml_backend_t
            gpu_cache_tensor,     // Target GPU tensor
            src_addr,             // Source CPU address
            slot_offset_bytes,    // Offset in target GPU tensor
            group_size_bytes,     // Size to copy
            io_stream_id          // I/O Stream
        );
    }

    return true;
}

// --- Reload Plan Implementation ---
/**
 * @brief [Plan Step 1] 统计激活的神经元数量。
 * @return std::vector<int> group_activated_neuron_count 每个组激活的神经元数量。
 */
std::vector<int> sparkinfer_layer_cache::_spif_reload_plan_count_activated_neurons(){
    std::vector<int> group_activated_neuron_count(this->layer_group_count, 0);
    const int* p_sparse_idx = (const int*)this->sparse_idx->data;

    for (int neuron_idx = 0; neuron_idx < this->layer_neuron_count; ++neuron_idx) {
        if (p_sparse_idx[neuron_idx]) { // 1 = activated
            int group_idx = neuron_idx / this->layer_group_size;
            if (group_idx < this->layer_group_count) { // 安全检查
                group_activated_neuron_count[group_idx]++;
            }
        }
    }
    return group_activated_neuron_count;
}

/**
 * @brief [Plan Step 2] 根据 DFR (或 LRU 等) 策略，计算出应在 GPU 上的 Top-K 组。
 * @param group_activated_neuron_count Step 1 的结果。
 * @return std::vector<int> groups_to_ensure 应该确保在 GPU 上的组列表。
 */
std::vector<int> sparkinfer_layer_cache::_spif_reload_plan_get_groups_to_ensure(const std::vector<int>& group_activated_neuron_count){
    // 使用 constexpr if 达到编译时分派
    if constexpr (SPARKINFER_RELOAD_STRATEGY == sparkinfer_reload_strategy::USE_DFR) {
        return _spif_reload_plan_use_dfr(group_activated_neuron_count);
    } else if constexpr (SPARKINFER_RELOAD_STRATEGY == sparkinfer_reload_strategy::USE_LRU) {
        // _spif_reload_plan_use_lru(group_activated_neuron_count); // 调用 LRU 实现
        LLAMA_LOG_ERROR("%s: [Error] LRU strategy not implemented yet.\n", __func__); 
    } else {
        LLAMA_LOG_WARN("%s: [Warning] Unknown Reload Strategy, fall back to DFR\n", __func__);
        return _spif_reload_plan_use_dfr(group_activated_neuron_count); // 默认使用 DFR
    }
}

/**
 * @brief [Plan Step 2.1] DFR 策略的具体实现。
 */
std::vector<int> sparkinfer_layer_cache::_spif_reload_plan_use_dfr(const std::vector<int>& group_activated_neuron_count) {
    typedef std::pair<float, int> ScoreGroupPair;
    std::priority_queue<ScoreGroupPair, std::vector<ScoreGroupPair>, std::greater<ScoreGroupPair>> min_heap;

    for (int group_id = 0; group_id < this->layer_group_count; ++group_id) {
        //  dfr_score = dfr_score * decay + is_activated * (1 - decay)
        float is_activated = (group_activated_neuron_count[group_id] > 0) ? 1.0f : 0.0f;
        float old_score = this->dfr_scores[group_id];
        float new_score = old_score * this->dfr_decay_rate + is_activated * (1.0f - this->dfr_decay_rate);
        
        this->dfr_scores[group_id] = new_score; // 持久化更新 DFR 分数

        // 维护 Top-K 最小堆
        if (min_heap.size() < (size_t)this->layer_group_capacity) {
            min_heap.push({new_score, group_id});
        } else if (new_score > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({new_score, group_id});
        }
    }

    // 从堆中提取 Top-K 组 (groups_to_ensure)
    std::vector<int> groups_to_ensure;
    groups_to_ensure.reserve(this->layer_group_capacity);
    while (!min_heap.empty()) {
        groups_to_ensure.push_back(min_heap.top().second); // .second 是 group_id
        min_heap.pop();
    }
    return groups_to_ensure;
}

/**
 * @brief [Plan Step 3] 计算差集并更新状态。
 * @param groups_to_ensure Step 2 的结果。
 * @return reload_plan_result 包含 plan_result.groups_to_reload 和 plan_result.slots_for_evict。
 */
const reload_plan_result & sparkinfer_layer_cache::_spif_reload_plan_compute_diff_and_update_state(const std::vector<int>& groups_to_ensure) {
    // Clear previous plan results
    this->plan_result.groups_to_reload.clear();
    this->plan_result.slots_for_evict.clear();

    // 1. O(k) 建立目标组的哈希集合
    std::unordered_set<int> target_groups(groups_to_ensure.begin(), groups_to_ensure.end());

    // 2. O(k) 遍历当前 GPU 上的组 (group_to_slot_hash)
    //    找出 Hit (保留), Miss (需驱逐)
    std::vector<int> groups_to_evict;
    
    for (auto it = this->group_to_slot_hash.begin(); it != this->group_to_slot_hash.end(); /* no increment */) {
        int group_in_gpu = it->first;
        int slot_id      = it->second;

        if (target_groups.find(group_in_gpu) == target_groups.end()) {
            // Miss: 在 GPU, 但不在 Target -> 驱逐
            groups_to_evict.push_back(group_in_gpu);
            this->plan_result.slots_for_evict.push_back(slot_id);
            
            // 更新状态: 从哈希表中移除
            it = this->group_to_slot_hash.erase(it);
        } else {
            // Hit: 在 GPU, 也在 Target -> 保留
            it++;
        }
    }

    // 3. O(k) 遍历目标组, 找出需要新加载的组
    for (const int group_id : target_groups) {
        if (this->group_to_slot_hash.find(group_id) == this->group_to_slot_hash.end()) {
            // 在 Target, 但不在 GPU (哈希表) -> 加载
            this->plan_result.groups_to_reload.push_back(group_id);
        }
    }

    // 4. 关键断言：空出的槽位必须等于需要加载的组
    GGML_ASSERT(this->plan_result.slots_for_evict.size() == this->plan_result.groups_to_reload.size() && 
                "Mismatch between slots to evict and groups to reload");

    // 5. 更新状态: 将新加载的组“放回”哈希表，复用空闲 slot
    for (size_t i = 0; i < this->plan_result.groups_to_reload.size(); ++i) {
        int group_to_load = this->plan_result.groups_to_reload[i];
        int slot_to_use   = this->plan_result.slots_for_evict[i]; // 复用被驱逐的 slot
        
        this->group_to_slot_hash[group_to_load] = slot_to_use;
    }

    return this->plan_result;
}

// --- sparkInfer_cache_manager Implementation ---

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
    threshold = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, 1);
    one       = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, 1);

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

        ggml_backend_tensor_set(layer_cache->neuron_idx, neuron_idx.data(), 0, ggml_nbytes(layer_cache->neuron_idx)); // [YPX] [B] 同样的命名冲突
        ggml_backend_tensor_set(layer_cache->neuron_mask, neuron_mask.data(), 0, ggml_nbytes(layer_cache->neuron_mask)); // [YPX] [B] 同样的命名冲突

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