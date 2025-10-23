#pragma once

 #include "llama-model.h"
 #include "llama-impl.h"
 #include "llama-model-loader.h"
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


 /*
 * @brief 管理单个层中FFN权重神经元的GPU缓存。
 * 
 * 负责在GPU上为部分卸载的神经元分配固定大小的缓存，并处理动态的换入换出操作。
 */
struct sparkInfer_layer_cache {

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;

    // 指向CPU上的完整FFN权重张量（数据源）
    struct ggml_tensor * cpu_ffn_gate = nullptr;
    struct ggml_tensor * cpu_ffn_up   = nullptr;
    struct ggml_tensor * cpu_ffn_down_t = nullptr;

    // offloaded ffn neuron bufferr
    ggml_backend_buffer_t gpu_weights_buffer = nullptr;
    // gpu split index buffers
    std::vector<ggml_backend_buffer_t> split_idx_allocated_buffers;

    // 代表GPU缓存池中所有槽位的张量
    struct ggml_tensor* gpu_ffn_gate_cache = nullptr;
    struct ggml_tensor* gpu_ffn_up_cache   = nullptr;
    struct ggml_tensor* gpu_ffn_down_t_cache = nullptr;

    // ffn sparse inference relevant
    struct ggml_tensor * ffn_gpu_neu_idx          = nullptr; // on gpu
    struct ggml_tensor * ffn_gpu_neu_mask         = nullptr;
    struct ggml_tensor * ffn_gpu_group_idx        = nullptr; // on gpu
    struct ggml_tensor * ffn_gpu_group_mask       = nullptr;
    struct ggml_tensor * ffn_neuron_to_group_map  = nullptr;

    int64_t neuron_cache_capacity = 0; // GPU缓存池能容纳的神经元数量
    uint64_t layer_neuron_count = 0; // 每层神经元总量
    uint64_t layer_group_count = 0; // 每层分组总量
    uint64_t layer_group_size = 0; // 每层分组大小
    
    // mappings
    std::unordered_map<int64_t, int64_t> neuron_to_slot_map; // 原始神经元索引 -> GPU缓存槽位索引
    std::vector<int64_t> slot_to_neuron_map; // GPU缓存槽位索引 -> 原始神经元索引

    // 实现LRU(最近最少使用)的替换策略，存储的是【原始神经元索引】, ofc we dont use lru, remove later
    std::list<int64_t> lru_tracker;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> lru_map;

    // DFR 
    struct ggml_tensor* dfr_score = nullptr;
    float decay_ratio = 0.9f;

    // 临时的ggml上下文，用于创建视图等临时张量
    struct ggml_context* tmp_ctx = nullptr;

    size_t offloaded_bytes=0;

    sparkInfer_layer_cache() = default;
    ~sparkInfer_layer_cache();

    /**
     * @brief 初始化层的缓存。
     * 
     * @param layer llama模型中的层。
     * @param backend 用于GPU操作的后端。
     * @param initial_gpu_neuron_indices 初始需要加载到GPU的神经元原始索引列表。
     * @return true 如果初始化成功。
     */
    bool init(int layer_idx, llama_model& model, llama_layer& layer, ggml_backend_t backend, const std::vector<int64_t>& initial_gpu_neu_idx);

    ggml_tensor * build_reload_impl(ggml_context * ctx, ggml_tensor * sparse_idx, const char * name, const int il);
    reload_plan_result * spif_reload_plan(ggml_tensor * tensor);

    /**
     * @brief 确保一个神经元在GPU上。如果不在，就执行换入操作。
     * 
     * @param neuron_idx 神经元的原始索引。
     * @return int32_t 神经元在GPU缓存中的槽位索引。
     */
    int64_t ensure_neuron_on_gpu(int64_t neuron_idx);

private:
    /**
     * @brief 将一个神经元从CPU复制到GPU的指定slot。
     */
    void copy_neuron_to_gpu_slot(int64_t neuron_idx, int64_t slot_idx);
    
    /**
     * @brief 更新缓存的元数据。
     */
    void update_mappings(int64_t neuron_idx, int64_t slot_idx);
};


/**
 * @brief 全局神经元缓存管理器，管理模型中所有层的缓存。
 *
 */
struct sparkInfer_cache_manager {
    std::vector<sparkInfer_layer_cache*> layer_caches;
    llama_model *model = nullptr;
    size_t total_offloaded_bytes=0;

    bool init(llama_model &p_model, ggml_backend_t gpu_backend);

    /**
     * @brief 在推理前，准备好指定层所需的一组神经元。
     *
     * @param layer_idx 层的索引。
     * @param required_neuron_indices 需要确保在GPU上的神经元原始索引列表。
     * @param out_gpu_slot_indices [输出参数] 填充更新后的GPU槽位索引，用于计算图。
     */
    void prepare_hot_neurons(int layer_idx, const std::vector<int64_t>& required_neuron_indices, std::vector<int64_t>& out_gpu_slot_indices);
};

/**
 * @brief 负责加载和应用分割张量的类。
 * 
 * 该类从指定的GGUF分割文件中加载张量，并根据模型层的GPU卸载比例将它们分配到CPU或GPU上下文中。
 * 它还管理分配的缓冲区，以确保正确的内存使用。
 */
struct sparkinfer_split_loader {
    std::string fname;
    struct gguf_context * ctx_gguf = nullptr;
    struct ggml_context * ctx_meta = nullptr;

    std::vector<ggml_backend_buffer_t> allocated_buffers;
    // 分别管理CPU和GPU张量的元数据
    struct ggml_context * ctx_cpu_tensors = nullptr;
    struct ggml_context * ctx_gpu_tensors = nullptr;

    int n_tensors = 0;
    uint64_t vram_required = 0;
    uint64_t layer_neuron_count = 0;
    uint64_t layer_group_count = 0;
    uint64_t layer_group_size = 0;

    sparkinfer_split_loader(const std::string & fname);

    ~sparkinfer_split_loader();

    /**
     * @brief 加载分割张量并将其应用到模型层。
     *        为动态张量预分配最大容量。
     * 
     * @param model 要应用张量的llama模型。
     * @param gpu_backend 用于卸载的GPU后端句柄，如果只想用CPU则为nullptr。
     * @return true 如果成功。
     */
    bool load_and_apply_split(llama_model & model, ggml_backend_t gpu_backend);

private:
    struct ggml_tensor* get_tensor_meta_from_gguf(int layer_idx, const std::string& suffix);
    struct ggml_tensor* create_static_tensor_in_ctx(ggml_context* ctx, int layer_idx, const std::string& suffix);
};

void debug_print_tensor_i64_to_file(FILE* log_file, const struct ggml_tensor* tensor);

sparkInfer_cache_manager* sparkinfer_init_and_manage_ffn_cache(struct llama_model* model, ggml_backend_t gpu_backend);

static bool sparkinfer_load_gpu_split_from_split_file(llama_model & model, std::string split_path, size_t vram_allocatable_bytes);

static bool sparkinfer_load_gpu_split_with_budget(llama_model_loader & ml, llama_model & model, size_t vram_allocatable_bytes, bool no_cache);

size_t sparkinfer_load_gpu_split_and_offload_weight(llama_model_loader & ml, llama_model & model, size_t vram_budget_bytes,bool no_cache);
