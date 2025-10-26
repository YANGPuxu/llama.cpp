# 2025.10.19 会议纪要

## 1. up, gate, down weight 三合一问题（可能的新创新点）

### 设计
- 对 CPU、GPU 上的 weight cache 进行整合（假设 gate 存在）：
    - 不再分为 up、gate、down 三个 cache，而是整合为一个统一的 cache。
    - CPU 和 GPU 的 cache 中，同一个神经元的 up、gate、down 权重排列在一起，物理排列形如：
        $$
        [\text{Neuron}^{1}_{up},\ \text{Neuron}^{1}_{down},\ \text{Neuron}^{1}_{gate},\ 
            \text{Neuron}^{2}_{up},\ \text{Neuron}^{2}_{down},\ \text{Neuron}^{2}_{gate},\ \cdots,\ 
            \text{Neuron}^{11008}_{up},\ \text{Neuron}^{11008}_{down},\ \text{Neuron}^{11008}_{gate}]
        $$
    - 创建三个 view tensor，使得在稀疏算子看来，权重仍然按照 up、gate、down 三个 cache 的方式排列。

### 疑问
- 这样的排列方式对 GPU 的稀疏算子函数的性能有多大影响？

## 2. manager 创建、初始化和 CPU 权重重排的时机

### 疑问
- 核心思路是否是：将 `sparkInfer_cache_manager*` 作为 `llama_context` 的成员，和 `llama_model` 同级？
    - 如果是的话，为什么 `llama-context.cpp` 里面有 `spif_cache = model.spif_cache`，把 cache manager 作为 model 的成员？
- 现在是不是还没有调用 `sparkInfer_cache_manager` 初始化逻辑的代码？（// [YPX] [C] 是的，应该没有正确初始化）

### 设计
- 在合适的时机执行 `sparkInfer_cache_manager()` 构造函数。
- 在加载完权重后，找一个时机执行 `sparkInfer_cache_manager::init()`，并在 `init()` 中添加 CPU 权重重排的逻辑。

## 3. 重要 Cache 张量是如何初始化并管理的？

### 疑问
- 为什么 `ffn_gpu_neu_idx` 等张量在 `sparkInfer_split_loader` 里面初始化，却作为 `llama_layer` 的成员，又被 `sparkInfer_cache_manager` 使用？这几个张量是如何被管理的？