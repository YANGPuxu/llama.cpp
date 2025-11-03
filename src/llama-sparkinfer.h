#pragma once

#include "llama-model.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
// #include "ggml-backend.h"
#include "ggml-spif.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <ggml-backend.h>
#include <numeric>
#include <ggml-cuda.h>
#include <list>
#include <omp.h>
#include <inttypes.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>  // LRU list
#include <queue> // DFR priority_queue
#include <set>   // set difference

struct ggml_backend_cuda_context;

enum class sparkinfer_reload_strategy : int {
    USE_DFR = 0,
    USE_LRU = 1,
    USE_FIFO = 2,
    USE_OPT = 3,
    USE_CLOCK = 4,
    USE_NRU = 5,
    USE_PLACEHOLDER = 6,
};

constexpr sparkinfer_reload_strategy SPARKINFER_RELOAD_STRATEGY = sparkinfer_reload_strategy::USE_DFR;

struct sparkinfer_layer_cache {

public:

    struct ggml_tensor * layer_ffn_pred_up     = nullptr;
    struct ggml_tensor * layer_ffn_pred_down   = nullptr;
    struct ggml_tensor * layer_ffn_pred_up_b   = nullptr;
    struct ggml_tensor * layer_ffn_pred_down_b = nullptr;

    // --- 后端与上下文 ---
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    struct ggml_context* tmp_ctx = nullptr; // 用于创建临时视图

    // --- CPU 权重 (数据源) ---
    struct ggml_tensor * layer_ffn_up     = nullptr;
    struct ggml_tensor * layer_ffn_gate   = nullptr;
    struct ggml_tensor * layer_ffn_down   = nullptr;
    struct ggml_tensor * layer_ffn_up_b   = nullptr;
    struct ggml_tensor * layer_ffn_gate_b = nullptr;
    struct ggml_tensor * layer_ffn_down_b = nullptr;

    // --- GPU 缓存池 (目标) ---
    ggml_backend_buffer_t gpu_weights_buffer = nullptr;
    struct ggml_tensor* ffn_gate_cache  = nullptr;
    struct ggml_tensor* ffn_up_cache    = nullptr;
    struct ggml_tensor* ffn_down_cache  = nullptr;

    // --- 稀疏预测相关张量 ---
    struct ggml_tensor * sparse_idx     = nullptr; // 在 GPU 上, 每次 plan 前需要从 GPU 同步
    struct ggml_tensor * neuron_idx     = nullptr; // 在 GPU 上
    struct ggml_tensor * neuron_mask    = nullptr; // 在 GPU 上
    struct ggml_tensor * neuron_map     = nullptr;

    // --- (待清理) 多 GPU 遗留? ---
    // [YPX] [Todo] 这很可能是多GPU split用的，待团队确认是否可移除
    std::vector<ggml_backend_buffer_t> split_idx_allocated_buffers;

    // --- 核心超参数 ---
    int layer_neuron_count     = 0; // 每层神经元总量
    int layer_group_count      = 0; // 每层分组总量
    int layer_group_size       = 0; // 每层分组大小
    int layer_group_capacity   = 0; // GPU缓存池能容纳的分组数量
    
    // --- 状态映射 (核心) ---
    std::unordered_map<int, int> group_to_slot_hash; // group_id -> slot_id

    // --- Reload Plan Result (核心) ---
    reload_plan_result plan_result;

    // --- DFR 替换策略状态 ---
    std::vector<float> dfr_scores; // size = layer_group_count
    float dfr_decay_rate = 0.9f;

    // --- 统计信息 ---
    size_t reload_group_count=0;
    size_t offloaded_bytes=0;

public:

    sparkinfer_layer_cache() = default;
    ~sparkinfer_layer_cache();

    // Sparkinfer reload (graph building)
    ggml_tensor * build_reload_plan(ggml_context * ctx, ggml_tensor * sparse_idx);
    ggml_tensor * build_reload_exec(ggml_context * ctx, ggml_tensor * sparse_idx, const char * name);

    // Sparkinfer reload (real implementation)
    bool spif_reload_plan(ggml_tensor * tensor);
    bool spif_reload_exec(ggml_tensor * tensor);

private:
    /**
     * @brief 更新缓存的元数据。
     */
    // [YPX] [B] 现在更新元数据（也就是这个类保管的张量）的工作应该由 Reload 算子的 compute_forward() 来负责，但是 init() 函数要调用这个，怎么办？干脆集成到 init() 函数里面算了。
    // [YPX] [Todo] 把这个函数集成到 init() 里面。
    void update_mappings(int64_t neuron_idx, int64_t slot_idx);

    /**
     * @brief [Plan Step 1] 统计激活的神经元数量。
     * @return std::vector<int> group_activated_neuron_count 每个组激活的神经元数量。
     */
    std::vector<int> _spif_reload_plan_count_activated_neurons();

    /**
     * @brief [Plan Step 2] 根据 DFR (或 LRU 等) 策略，计算出应在 GPU 上的 Top-K 组。
     * @param group_activated_neuron_count Step 1 的结果。
     * @return std::vector<int> groups_to_ensure 应该确保在 GPU 上的组列表。
     */
    std::vector<int> _spif_reload_plan_get_groups_to_ensure(const std::vector<int>& group_activated_neuron_count);

    /**
     * @brief [Plan Step 2.1] DFR 策略的具体实现。
     */
    std::vector<int> _spif_reload_plan_use_dfr(const std::vector<int>& group_activated_neuron_count);
    
    // [YPX] [Todo] 其他策略，比如 _spif_reload_plan_use_lru()

    /**
     * @brief [Plan Step 3] 计算差集并更新状态。
     * @param groups_to_ensure Step 2 的结果。
     * @return reload_plan_result 包含 groups_to_reload 和 slots_for_evict。
     */
    const reload_plan_result & _spif_reload_plan_compute_diff_and_update_state(const std::vector<int>& groups_to_ensure);
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
