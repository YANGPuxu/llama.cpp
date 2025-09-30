## 9.30 last updated [TODO]

### [done]任务 1: 整理 sparse loading 加载阶段的类(在模型加载过程中)
- 将这个类迁移到 `llama-sparkinfer.h/cpp` 文件中。
- 不添加额外功能，仅对命名进行规范化。
- 确保迁移后运行结果与当前一致。
- 对后续的功能完善初始化函数，要求将 `llama-model` 中的以下全局张量交由 `spif_cache` 统一管理：
    - `ffn_gpu_up`
    - `ffn_gpu_gate`
    - `ffn_gpu_down_t`
    - `gpu_neu_idx/mask`
    - ...
    - 直接赋值指针??

### [done]任务 2: 初始化与整合
- 任务 1 完成了在模型加载阶段初始化 `sparkInfer_cache_manager`。
- 在 `llama-context` 初始化时，将已初始化的 `sparkInfer_cache_manager` 实例保存到 `llama-context` 中。
- 我们不应该在 `sparkInfer_cache_manager` 类里面保存 `model`？？然后在初始化 `sparkInfer_cache_manager` 
- 后把其保存到 `model` 里面，后续用 `model` 里的 `spif_cache` 初始化 `llama-context` 里面的 s`parkinfer_cache_manager` 成员
- **补充说明**:  
    在 `main` 函数中，模型和上下文的初始化顺序如下：
    ```cpp
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    ...
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    ```
    这样，在当前 `reload` 分支中，从 `llama-context` 初始化到传递到 `build_graph` 的路径，仅需修改 `llama-context` 初始化部分。

### 任务 3: 构建计算图

将 `build_reload` 函数写入 `sparkInfer_cache_manager` 类中，作为成员函数调用。

#### 3.1 构建 `build_reload` 节点
将 `build_reload` 作为节点插入计算图中，并强制后端为 CPU。设置 `src` 的内容如下：
```cpp
result->src[0] = cpu_ffn_up;
result->src[1] = gpu_ffn_up_cache;
result->src[2] = sparse_idx;
result->src[3] = ffn_gpu_neu_idx;
result->src[4] = dfr_score;
```
这些内容供图计算时获取和使用。

#### 3.2 GPU 张量同步问题
在 llama.cpp 的机制中，GPU 上的 `gpu_ffn_up_cache` 会被同步到与 `build_reload` 节点相同的 CPU 后端。  
**[TODO]**: 修改划分 splits 的 `pass5`，避免将 `gpu_ffn_up_cache` 从 GPU 同步到 CPU。  
已完成的解决方案：
```cpp
// 在 sparkinfer 的 reload splits 中，我们不需要将 qgu 输入复制回 GPU。
// 当前确保只有 reload splits 会包含 gpu_ffn 张量的输入。
// 这种方式较为 hacky，但目前可行。[GTODO] 需要更好的方法来处理这个问题。
if (strstr(input->name, "ffn_qgu_") != NULL) {
    continue;
}

好像有问题： 在图计算阶段，sparse_idx gpu_neu_idx还在 CUDA 上，[TODO:]再仔细看看这个跳过 splits input cpy 的逻辑
```

#### 3.3 Reload 张量管理
目前 reload 的三个张量 `up`、`gate` 和 `down` 是分别对每个 reload 节点调用 `build_forward_expand`。  
需要考虑以下问题：
- 是否添加一个新的节点，将上述三个张量作为 `src`，以便后续仅需管理这个 end 节点？
- 统筹管理的含义是什么？是否与后续 splits 并行相关？
- 会多出错误：`update_cuda_graph_executable: CUDA graph update failed` 的错误原因是什么？

当前采用的方式是对每个 reload 节点分别调用 `build_forward_expand`。


### 任务 4: graph splits 的并行计算实现
 - reload 的 splits 在并行的时候怎么异步保证 reload 完成？不影响下一个 cpu 后端的 mulmat splits 的计算

### 任务 5: cold hit 依旧很高
 - 后几层的 cpu 计算 burden 很大，远超 gpu 计算时间

### 任务 6: batch 的 sparse 问题