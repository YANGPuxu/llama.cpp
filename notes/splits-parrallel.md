# 关于 splits 之间的并行设计

## 为什么目前 split 是串行执行？

### 1. 依赖机制
目前的 llamacpp 实现中，splits 计算前都 `sync(input_backend)`，对于我们的 CPU 和 GPU 交替的 splits 结构来说，也就是等上一个 splits 的后端计算完（CPU 等 GPU，GPU 等 CPU）。

我们目前的 heterogenous FFN computing:
- `Attd+Pred(GPU) -> Reload+FNN(CPU) -> FNN(GPU)`
- 到了 `FNN(GPU)` 的时候会强制同步 CPU 后端。

**[TODO:]** 重新设计 splits 顺序和依赖，让更容易并行。

### 2. 异步计算的支持
在 `ggml-backend.cpp` 中：
```cpp
for each splits:
    ggml_backend_graph_compute_async(split_backend, &split->graph);
```

然后这个 `ggml_backend_graph_compute_async` 调用 `split_backend` 的后端接口：
```cpp
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}
```

#### 1.1 对于 CPU 后端：
CPU 后端对应的 `graph_compute` 函数是没有异步返回的，只能计算完再统一返回。

#### 1.2 对于 CUDA 后端：
CUDA 后端对应的计算逻辑，按道理 CPU 发送完 CUDA 计算图后就可以返回了，但是目前看到还是等待 GPU 计算完成后才进行下一个 splits 的计算。

原因是上述所说的依赖关系，下一个 CPU splits 不管是否依赖当前 GPU splits，都 `sync(gpu_backend)`。

如果 CUDA API 发送不打算设计成和 CPU computing 并行的话：
**[TODO:]** 对于 FFN 中可以并行的 FFN splits 调整计算顺序，让 GPU 先算，发送完计算图就可以返回进行 CPU 的 split 计算。

---



---

## 如何设计并行计算？
两种思路：
1. 在调用后端函数前就开始并行？把调起后端函数这一层操作就开始并行操作。
```cpp
    pthread_create(compute_splits())
    compute_splits(){
        if (ith == 0 && split backend==GPU) compute
        if (ith != 0 && split_backend==CPU) compute 
    }
```

2. 只并行 cuda 计算和 cpu 计算
    - [TODO]先规划好 splits 的顺序
    - 让 GPU splits 发送计算图给 cuda 后立刻返回
    - 先把 reload 算子放在 CUDA 后端，


---

**PS:** 在 llamacpp 中会有 pipeline-parallel 等控制，比如在 `backend_sched` 中有：
```cpp
// pipeline parallelism support
int n_copies;
int cur_copy;
ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
int n_graph_inputs;
```

但是这些是给多请求的流水线并行的，我们是单请求，用不上。
