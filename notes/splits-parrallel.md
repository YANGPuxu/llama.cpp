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

1. **在调用后端函数前就开始并行？**  
    把调起后端函数这一层操作就开始并行操作。
    ```cpp
    pthread_create(compute_splits())
    compute_splits(){
         if (ith == 0 && split backend==GPU) compute
         if (ith != 0 && split_backend==CPU) compute 
    }
    ```

2. **只并行 CUDA 计算和 CPU 计算** *目前采用这种*  
    - ~~[TODO]先规划好 splits 的顺序~~  
    - ~~[TODO]先把 reload 算子放在 CUDA 后端~~  
    - ~~[TODO] 改写 splits 同步机制，加入依赖？~~

---

## split 并行的具体实现

不管哪种方法，无非都是两件事情：

1. **在 compute 到开始并行的节点 (ffn_norm & (ffn_gate_par || ffn_up_act)) 做 eventRecord**  
    因为是要在这个 node 的 kernel 发送后做 Record，因此只能跑到 split_compute 的后端代码 `ggml-cuda.cu` 中去写而不是 `ggml-backend.cpp` 中。

2. **在可以并行的 CPU split 计算前，调用 eventSync**  
    那么这个 event 应该存在哪里，好让在 splits 的for循环能够够到？

---

### 0. 相关的变量抽象包装

```cpp
typedef struct ggml_backend_event * ggml_backend_event_t;
struct ggml_backend_event {
     struct ggml_backend_device * device;
     void * context;
};
```

---

### 1. 把 event 放在 cgraph 里

- 因为计算 split 的时候只传入 backend 和 graph：
  ```cpp
  enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
  ```

- 我们可以在 graph 中做 mark，告诉 event 在哪里 record：
  ```cpp
  // ggml-impl.h
  struct ggml_cgraph {
        int size;    // maximum number of nodes/leafs/grads/grad_accs
        int n_nodes; // number of nodes currently in use
        int n_leafs; // number of leafs currently in use

        struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
        struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
        struct ggml_tensor ** grad_accs; // accumulators for node gradients
        struct ggml_tensor ** leafs;     // tensors with constant data

        struct ggml_hash_set visited_hash_set;

        enum ggml_cgraph_eval_order order;

        // sparkinfer splits parallel support
        int parallel_node_id;  // if this is -1, then the graph is not parallel
        void * parallel_event; // nullptr if not parallel (ggml_backend_event_t)
        bool need_waiting;    // whether the graph needs to wait for the parallel event
  };
  ```

- 然后在 split 初始化的时候同时初始化新增成员：
  ```cpp
  // SparkInfer: initialized parallel-support member in cgraph
  split->graph.parallel_node_id = -1;
  split->graph.need_waiting = false;
  split->graph.parallel_event = nullptr;
  ```

---

### 2. 在进入 `ggml_backend_sched_compute_splits` 函数之前，对 `sched->splits` 中的每一个 graph 做规划（在这里解耦了并行规划和并行实现）

```cpp
enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
     if (!sched->is_reset && !sched->is_alloc) {
          ggml_backend_sched_reset(sched);
     }

     if (!sched->is_alloc) {
          if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                return GGML_STATUS_ALLOC_FAILED;
          }
     }

     // Sparkinfer: design parallel strategy for splits
     enum ggml_status par_status = ggml_backend_sched_splits_parallel(sched);
     if (par_status != GGML_STATUS_SUCCESS){
          return par_status;
     }

     return ggml_backend_sched_compute_splits(sched);
}
```

- `ggml_backend_sched_splits_parallel` 具体实现看代码，主要干了两件事情：
  1. 给可并行的 GPU split 标记出可以开始给下一个 CPU 并行的 node，记录在 `graph->parallel_node_id`。
  2. 找到需要跟上一（或 n）个 GPU split 并行的 CPU split，设置 `need_waiting = true`。

---

### 3. event 的 record

- 我们只用在 CUDA 的后端中做 record event，大概的结构：
  ```cpp
  // ggml-cuda.cu
  for each node i in graph:
        bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);  // first we compute the node

        //...

        // Sparkinfer: create and record event after the kernel of parallel node
        if (i == cgraph->parallel_node_id) {  // 如果这个 node_id 是 parallel_node_id
             ggml_backend_event_t parallel_event = ggml_backend_cuda_device_event_new(backend->device);  // 新建 event

             ggml_backend_cuda_event_record(backend, parallel_event); // 记录 event

             // 然后把 event 保存进 graph 的 event 中
             if (cgraph->parallel_event != nullptr) {
                  GGML_ASSERT("cgraph->parallel_event is not null");
             } else {
                  cgraph->parallel_event = (void *) parallel_event;
             }
        }
  ```

---

### 4. event 的 sync

- 在 `ggml_backend_sched_compute_splits` 中，大概结构如下：
  ```cpp
  // ggml-backend.cpp
  ggml_backend_event_t parallel_event = nullptr;  // 在外围定义一个 event 变量供 for 循环使用
  for each split in splits:
        // Sparkinfer: if this is a parallelable CPU split, we need to wait for the parallel event
        if (split->graph.need_waiting) {
             ggml_backend_event_synchronize(parallel_event);
        }

        // copy the input
        for each input in split->n_inputs:
             // Sparkinfer: copy input
             if (split->graph.need_waiting) {
                  ggml_backend_tensor_copy(input, input_cpy);
                  continue;
             }

             // fallback to original synchronize
             // ...
        
        // compute
        enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);

        // Sparkinfer: if this is a parallelable GPU split, get parallel_event, and this variable would be used in next few split
        if (split->graph.parallel_node_id != -1){
             parallel_event = (ggml_backend_event_t)(split->graph.parallel_event);
        }
  ```


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
