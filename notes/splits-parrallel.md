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

## split 并行的具体实现（这个已经弃用）

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
## 对 reload 的并行改进 (上述的 split 依赖关系已经弃用)

### 1. 到 reload split 的时候，新建立一个线程去做计划，然后立刻返回
- 如何新建线程？
- 原线程建立新线程后立马异步返回，不阻塞 CPU 的 FFN 计算。

### 2. 新线程会在获得记录 sparse_idx 计算完成的事件 (eventSync) 后才开始reload plan
- 在 pred-out 后面进行事件记录 (eventRecord)。

### 3. 如何保证下一层 FFN 计算前 reload 完成？
- 因为原先 reload split 已经异步返回了，无法通过 llamacpp 原先的 split 之间同步去保证依赖关系。
- 通过 event 去管理：
     - 这个 event 在 reload 后，保存在哪里？放在 `ggml_tensor * done_reload` 里面？

## 只能通过 node 为粒度去做 pipeline 的 overlap 依赖

### 1. 在 `backend-sched` 里面为每一个 event 分配空间，所有 event init 和释放都在 `backend_sched` 里面实现。

```cpp
struct ggml_backend_sched {
    ....

    // Sparkinfer pipeline support, especially for hyterogeneous backends splits
    std::unordered_map<std::string, ggml_backend_event_t> spif_events;

    ....
};
```


### 2. 在 `make_pipeline_strategy` 的时候，对所有需要的依赖 nodes 对，做 `link(node1, node2)`。

```cpp
// Sparkinfer: in this function, we design the parallel strategy for splits
enum ggml_status ggml_backend_sched_make_parallel_strategy_new(ggml_backend_sched_t sched){
    // we skip the first CPU split (inpu_embd)
    for (int i = 1;i < sched->n_splits;i++){
        ggml_backend_sched_split * split = &sched->splits[i];
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, split->backend_id);

        if (ggml_backend_dev_type(backend->device) == GGML_BACKEND_DEVICE_TYPE_CPU){
            // for CPU split, we link the input[0] and the first node to an event
            spif_link_node_to_event(sched, split->inputs[0], split->graph.nodes[0]);
            // printf("split %d on CPU linked event between %s and %s\n", i, split->inputs[0]->name, split->graph.nodes[0]->name);
        }

    }
    return GGML_STATUS_SUCCESS;
}
```

- `link(node1, node2)` 做的是把这两个 nodes 的 `extra`（或其他新建字段）赋值为已经 init 的 event，同时标识哪个 node 做 record，哪个 node 做 sync。

```cpp
// Sparkinfer: this is a link function, to link two node to an event, the event was store in sched
void spif_link_node_to_event(ggml_backend_sched_t sched, ggml_tensor * record_node, ggml_tensor * wait_node){
    // create a new event
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, record_node);
    ggml_backend_event_t event = ggml_backend_event_new(backend->device);

    // find or create the event in sched
    ggml_backend_event_t event = nullptr;
    std::string key = std::string(record_node->name) + "_" + std::string(wait_node->name);
    if (sched->spif_events.count(key)){
        event = sched->spif_events[key];
    }else{
        event = ggml_backend_event_new(backend->device);
        sched->spif_events[key] = event;
    }

    // link the event to record_node and wait_node
    struct spif_node_event * record_ev = (spif_node_event *)new spif_node_event();
    struct spif_node_event *   wait_ev = (spif_node_event *)new spif_node_event();
    record_ev->event = event;  record_ev->record_or_wait = 0;
    wait_ev->event   = event;  wait_ev->record_or_wait = 1;
    
    // store the spif_node_event to (void *)extra in ggml_tensor
    record_node->extra = (void *)record_ev;
    wait_node->extra   = (void *)wait_ev;

    printf("link event %p between %s and %s\n", event, record_node->name, wait_node->name);
}
```

### 3. 在 `graph_compute` 函数的后端函数中，对 node 进行判断，做 record 和 sync。
- in cpu backend:
```cpp
    for node in nodes: 
#ifdef SPIF_PIPELINE
        // Sparkinfer: wait for the parallel event before computing the next node
        if (node->extra != NULL) {
            spif_node_event * node_event = (spif_node_event *) node->extra;
            if (node_event->record_or_wait == 1) {
                ggml_backend_event_t parallel_event = (ggml_backend_event_t) node_event->event;
                GGML_ASSERT(parallel_event != NULL);

                // only thread 0 waits for the event
                if (state->ith == 0) {
                    printf("thread %d waiting for event %p before computing node %s\n", state->ith, parallel_event, node->name); 
                    ggml_backend_event_synchronize(parallel_event);
                }
                // ensure all threads wait for thread 0 to finish waiting
                ggml_barrier(state->threadpool);

            } else {
                GGML_ABORT("Invalid node event state in CPU graph compute");
            }
        }
#endif
        compute(node)
```

```cpp
static void evaluate_and_capture_cuda_graph(ggml_backend_t backend, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
    
    for node in nodes: 

#ifdef SPIF_PIPELINE
                // Sparkinfer: wait for the parallel event before computing the next node
                if (node->extra != nullptr) {
                    spif_node_event * node_event = (spif_node_event *) node->extra;
                    if(node_event->record_or_wait == 1){
                        ggml_backend_event_t parallel_event = (ggml_backend_event_t) node_event->event;
                        GGML_ASSERT(parallel_event != nullptr);
                        ggml_backend_event_synchronize(parallel_event);
                    }
                }
#endif
          // ...

               // computation
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);

          // ...

#ifdef SPIF_PIPELINE
                // Sparkinfer: record the event after the kernel of the node
                if (node->extra != nullptr) {
                    spif_node_event * node_event = (spif_node_event *) node->extra;
                    if(node_event->record_or_wait == 0){
                        ggml_backend_event_t parallel_event = (ggml_backend_event_t) node_event->event;
                        GGML_ASSERT(parallel_event != nullptr);
                        ggml_backend_cuda_event_record(backend, parallel_event);
                    }
                }
#endif
            
    }
```

---

## 那 llamacpp 原先的 input backend sync 怎么保留？

### 实际上对于一个 split，如果上一个 split 是 CPU，那么根本不用做 input backend sync，因为 CPU 后端是同步的，直接 copy input 就好
     - 这里可以开洞，对于 `prev_split` 是 CPU 的，直接 copy input 即可
```cpp
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_split * splits = sched->splits;

    for (int i = 0; i < sched->n_splits; i++) {
        ...
        for (int j = 0; j < split->n_inputs; j++) {
            ...
            
            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                ...
            } else {
                ...
#ifdef SPIF_PIPELINE
                // for CPU backend, we ensure(?) that there will be a node in this split has the event dependency
                if (ggml_backend_dev_type(split_backend->device) == GGML_BACKEND_DEVICE_TYPE_CPU && i != 0) {
                    ggml_backend_tensor_copy(input, input_cpy);
                    continue;
                }
#endif
                if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                    ggml_backend_synchronize(input_backend);
                    ggml_backend_tensor_copy(input, input_cpy);
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            ...
        } else {
            //....
        }

    }
    return GGML_STATUS_SUCCESS;
}
```

- 对于上一个 split 是 GPU 的 split 计算，怎么开洞？
     - 上一个 split 是 GPU 的， 一般是 CPU split，都会有 pipeline event 依赖，因此也是直接 copy input
     - 我们只要找到没有 pipeline event 依赖的 CPU split，再做 input backend sync
     - 目前还没处理

## 目前先 link 了 ffn 的计算依赖，reload 相关的还没 link，出现可复现的错误：
### 每次都能过 prefill，但在第一次 decode，同步倒数第二个 CPU split 的时候出现报错：
```
<s> Once upon a time, link event 0x55ab10e00850 between ffn_norm-1 and ffn_up_sparse_cpu-1
link event 0x55ab0faa9140 between ffn_gate_par-1 and ffn_down_sparse_cpu-1
link event 0x55ab0faa93a0 between ffn_norm-16 and ffn_up_sparse_cpu-16
link event 0x55ab0faa9600 between ffn_gate_par-16 and ffn_down_sparse_cpu-16
link event 0x55ab0faa9860 between ffn_norm-17 and ffn_up_sparse_cpu-17
link event 0x55ab0faa9ac0 between ffn_gate_par-17 and ffn_down_sparse_cpu-17
link event 0x55ab0faa9d20 between ffn_norm-18 and ffn_up_sparse_cpu-18
link event 0x55ab0faa9f80 between ffn_gate_par-18 and ffn_down_sparse_cpu-18
link event 0x55ab0faaa1e0 between ffn_norm-19 and ffn_up_sparse_cpu-19
link event 0x55ab0faaa440 between ffn_gate_par-19 and ffn_down_sparse_cpu-19
link event 0x55ab0faaa6a0 between ffn_norm-20 and ffn_up_sparse_cpu-20
link event 0x55ab0faaa900 between ffn_gate_par-20 and ffn_down_sparse_cpu-20
link event 0x55ab0faaab60 between ffn_norm-21 and ffn_up_sparse_cpu-21
link event 0x55ab0faaadc0 between ffn_gate_par-21 and ffn_down_sparse_cpu-21
link event 0x55ab0faab020 between ffn_norm-22 and ffn_up_sparse_cpu-22
link event 0x55ab0faab280 between ffn_gate_par-22 and ffn_down_sparse_cpu-22
link event 0x55ab0faab4e0 between ffn_norm-23 and ffn_up_sparse_cpu-23
link event 0x55ab10dcda80 between ffn_gate_par-23 and ffn_down_sparse_cpu-23
link event 0x55ab10dcdce0 between ffn_norm-24 and ffn_up_sparse_cpu-24
link event 0x55ab10dcdf40 between ffn_gate_par-24 and ffn_down_sparse_cpu-24
link event 0x55ab10dce1a0 between ffn_norm-25 and ffn_up_sparse_cpu-25
link event 0x55ab10dce400 between ffn_gate_par-25 and ffn_down_sparse_cpu-25
link event 0x55ab10dce660 between ffn_norm-26 and ffn_up_sparse_cpu-26
link event 0x55ab10dce8c0 between ffn_gate_par-26 and ffn_down_sparse_cpu-26
link event 0x55ab10dceb20 between ffn_norm-27 and ffn_up_sparse_cpu-27
link event 0x55ab10dced80 between ffn_gate_par-27 and ffn_down_sparse_cpu-27
link event 0x55ab10dcefe0 between ffn_norm-28 and ffn_up_sparse_cpu-28
link event 0x55ab10dcf240 between ffn_gate_par-28 and ffn_down_sparse_cpu-28
link event 0x55ab10dcf4a0 between ffn_norm-29 and ffn_up_sparse_cpu-29
link event 0x55ab10dcf700 between ffn_gate_par-29 and ffn_down_sparse_cpu-29
link event 0x55ab10dcf960 between ffn_norm-30 and ffn_up_sparse_cpu-30
link event 0x55ab10dcfbc0 between ffn_gate_par-30 and ffn_down_sparse_cpu-30
link event 0x55ab10dcfe20 between ffn_norm-31 and ffn_up_sparse_cpu-31
link event 0x55ab10dd0080 between ffn_gate_par-31 and ffn_down_sparse_cpu-31
thread 0 waiting for event 0x55ab10e00850 before computing node ffn_up_sparse_cpu-1
thread 0 waiting for event 0x55ab0faa9140 before computing node ffn_down_sparse_cpu-1
thread 0 waiting for event 0x55ab0faa93a0 before computing node ffn_up_sparse_cpu-16
thread 0 waiting for event 0x55ab0faa9600 before computing node ffn_down_sparse_cpu-16
thread 0 waiting for event 0x55ab0faa9860 before computing node ffn_up_sparse_cpu-17
thread 0 waiting for event 0x55ab0faa9ac0 before computing node ffn_down_sparse_cpu-17
thread 0 waiting for event 0x55ab0faa9d20 before computing node ffn_up_sparse_cpu-18
thread 0 waiting for event 0x55ab0faa9f80 before computing node ffn_down_sparse_cpu-18
thread 0 waiting for event 0x55ab0faaa1e0 before computing node ffn_up_sparse_cpu-19
thread 0 waiting for event 0x55ab0faaa440 before computing node ffn_down_sparse_cpu-19
thread 0 waiting for event 0x55ab0faaa6a0 before computing node ffn_up_sparse_cpu-20
thread 0 waiting for event 0x55ab0faaa900 before computing node ffn_down_sparse_cpu-20
thread 0 waiting for event 0x55ab0faaab60 before computing node ffn_up_sparse_cpu-21
thread 0 waiting for event 0x55ab0faaadc0 before computing node ffn_down_sparse_cpu-21
thread 0 waiting for event 0x55ab0faab020 before computing node ffn_up_sparse_cpu-22
thread 0 waiting for event 0x55ab0faab280 before computing node ffn_down_sparse_cpu-22
thread 0 waiting for event 0x55ab0faab4e0 before computing node ffn_up_sparse_cpu-23
thread 0 waiting for event 0x55ab10dcda80 before computing node ffn_down_sparse_cpu-23
thread 0 waiting for event 0x55ab10dcdce0 before computing node ffn_up_sparse_cpu-24
thread 0 waiting for event 0x55ab10dcdf40 before computing node ffn_down_sparse_cpu-24
thread 0 waiting for event 0x55ab10dce1a0 before computing node ffn_up_sparse_cpu-25
thread 0 waiting for event 0x55ab10dce400 before computing node ffn_down_sparse_cpu-25
thread 0 waiting for event 0x55ab10dce660 before computing node ffn_up_sparse_cpu-26
thread 0 waiting for event 0x55ab10dce8c0 before computing node ffn_down_sparse_cpu-26
thread 0 waiting for event 0x55ab10dceb20 before computing node ffn_up_sparse_cpu-27
thread 0 waiting for event 0x55ab10dced80 before computing node ffn_down_sparse_cpu-27
thread 0 waiting for event 0x55ab10dcefe0 before computing node ffn_up_sparse_cpu-28
thread 0 waiting for event 0x55ab10dcf240 before computing node ffn_down_sparse_cpu-28
thread 0 waiting for event 0x55ab10dcf4a0 before computing node ffn_up_sparse_cpu-29
thread 0 waiting for event 0x55ab10dcf700 before computing node ffn_down_sparse_cpu-29
thread 0 waiting for event 0x55ab10dcf960 before computing node ffn_up_sparse_cpu-30
thread 0 waiting for event 0x55ab10dcfbc0 before computing node ffn_down_sparse_cpu-30
thread 0 waiting for event 0x55ab10dcfe20 before computing node ffn_up_sparse_cpu-31
thread 0 waiting for event 0x55ab10dd0080 before computing node ffn_down_sparse_cpu-31
 spllink event 0x55ab0f91d730 between ffn_norm-1 and ffn_up_sparse_cpu-1
link event 0x55ab20f66fe0 between ffn_gate_par-1 and ffn_down_sparse_cpu-1
link event 0x55ab0f7fb1a0 between ffn_norm-16 and ffn_up_sparse_cpu-16
link event 0x55ab0ff2fc10 between ffn_gate_par-16 and ffn_down_sparse_cpu-16
link event 0x55ab10da1fd0 between ffn_norm-17 and ffn_up_sparse_cpu-17
link event 0x55ab10dcfaf0 between ffn_gate_par-17 and ffn_down_sparse_cpu-17
link event 0x55ab10dcf890 between ffn_norm-18 and ffn_up_sparse_cpu-18
link event 0x55ab10dcf700 between ffn_gate_par-18 and ffn_down_sparse_cpu-18
link event 0x55ab10dcf4a0 between ffn_norm-19 and ffn_up_sparse_cpu-19
link event 0x55ab0f7fbc20 between ffn_gate_par-19 and ffn_down_sparse_cpu-19
link event 0x55ab10dcefe0 between ffn_norm-20 and ffn_up_sparse_cpu-20
link event 0x55ab10dced80 between ffn_gate_par-20 and ffn_down_sparse_cpu-20
link event 0x55ab10dceb20 between ffn_norm-21 and ffn_up_sparse_cpu-21
link event 0x55ab10dce8c0 between ffn_gate_par-21 and ffn_down_sparse_cpu-21
link event 0x55ab10dce660 between ffn_norm-22 and ffn_up_sparse_cpu-22
link event 0x55ab10dce400 between ffn_gate_par-22 and ffn_down_sparse_cpu-22
link event 0x55ab10dce1a0 between ffn_norm-23 and ffn_up_sparse_cpu-23
link event 0x55ab10dcdf40 between ffn_gate_par-23 and ffn_down_sparse_cpu-23
link event 0x55ab10dcdce0 between ffn_norm-24 and ffn_up_sparse_cpu-24
link event 0x55ab10dcda80 between ffn_gate_par-24 and ffn_down_sparse_cpu-24
link event 0x55ab0faab4e0 between ffn_norm-25 and ffn_up_sparse_cpu-25
link event 0x55ab0faab280 between ffn_gate_par-25 and ffn_down_sparse_cpu-25
link event 0x55ab0faaabc0 between ffn_norm-26 and ffn_up_sparse_cpu-26
link event 0x55ab0faaaa30 between ffn_gate_par-26 and ffn_down_sparse_cpu-26
link event 0x55ab0faaab60 between ffn_norm-27 and ffn_up_sparse_cpu-27
link event 0x55ab0faaa900 between ffn_gate_par-27 and ffn_down_sparse_cpu-27
link event 0x55ab0faaa6a0 between ffn_norm-28 and ffn_up_sparse_cpu-28
link event 0x55ab0faaa1e0 between ffn_gate_par-28 and ffn_down_sparse_cpu-28
link event 0x55ab0faa9f80 between ffn_norm-29 and ffn_up_sparse_cpu-29
link event 0x55ab0faaa430 between ffn_gate_par-29 and ffn_down_sparse_cpu-29
link event 0x55ab20f67250 between ffn_norm-30 and ffn_up_sparse_cpu-30
link event 0x55ab20f674b0 between ffn_gate_par-30 and ffn_down_sparse_cpu-30
link event 0x55ab20f67710 between ffn_norm-31 and ffn_up_sparse_cpu-31
link event 0x55ab20f67970 between ffn_gate_par-31 and ffn_down_sparse_cpu-31
thread 0 waiting for event 0x55ab0f91d730 before computing node ffn_up_sparse_cpu-1
thread 0 waiting for event 0x55ab20f66fe0 before computing node ffn_down_sparse_cpu-1
thread 0 waiting for event 0x55ab0f7fb1a0 before computing node ffn_up_sparse_cpu-16
thread 0 waiting for event 0x55ab0ff2fc10 before computing node ffn_down_sparse_cpu-16
thread 0 waiting for event 0x55ab10da1fd0 before computing node ffn_up_sparse_cpu-17
thread 0 waiting for event 0x55ab10dcfaf0 before computing node ffn_down_sparse_cpu-17
thread 0 waiting for event 0x55ab10dcf890 before computing node ffn_up_sparse_cpu-18
thread 0 waiting for event 0x55ab10dcf700 before computing node ffn_down_sparse_cpu-18
thread 0 waiting for event 0x55ab10dcf4a0 before computing node ffn_up_sparse_cpu-19
thread 0 waiting for event 0x55ab0f7fbc20 before computing node ffn_down_sparse_cpu-19
thread 0 waiting for event 0x55ab10dcefe0 before computing node ffn_up_sparse_cpu-20
thread 0 waiting for event 0x55ab10dced80 before computing node ffn_down_sparse_cpu-20
thread 0 waiting for event 0x55ab10dceb20 before computing node ffn_up_sparse_cpu-21
thread 0 waiting for event 0x55ab10dce8c0 before computing node ffn_down_sparse_cpu-21
thread 0 waiting for event 0x55ab10dce660 before computing node ffn_up_sparse_cpu-22
thread 0 waiting for event 0x55ab10dce400 before computing node ffn_down_sparse_cpu-22
thread 0 waiting for event 0x55ab10dce1a0 before computing node ffn_up_sparse_cpu-23
thread 0 waiting for event 0x55ab10dcdf40 before computing node ffn_down_sparse_cpu-23
thread 0 waiting for event 0x55ab10dcdce0 before computing node ffn_up_sparse_cpu-24
thread 0 waiting for event 0x55ab10dcda80 before computing node ffn_down_sparse_cpu-24
thread 0 waiting for event 0x55ab0faab4e0 before computing node ffn_up_sparse_cpu-25
thread 0 waiting for event 0x55ab0faab280 before computing node ffn_down_sparse_cpu-25
thread 0 waiting for event 0x55ab0faaabc0 before computing node ffn_up_sparse_cpu-26
thread 0 waiting for event 0x55ab0faaaa30 before computing node ffn_down_sparse_cpu-26
thread 0 waiting for event 0x55ab0faaab60 before computing node ffn_up_sparse_cpu-27
thread 0 waiting for event 0x55ab0faaa900 before computing node ffn_down_sparse_cpu-27
thread 0 waiting for event 0x55ab0faaa6a0 before computing node ffn_up_sparse_cpu-28
thread 0 waiting for event 0x55ab0faaa1e0 before computing node ffn_down_sparse_cpu-28
thread 0 waiting for event 0x55ab0faa9f80 before computing node ffn_up_sparse_cpu-29
thread 0 waiting for event 0x55ab0faaa430 before computing node ffn_down_sparse_cpu-29
thread 0 waiting for event 0x55ab20f67250 before computing node ffn_up_sparse_cpu-30
thread 0 waiting for event 0x55ab20f674b0 before computing node ffn_down_sparse_cpu-30
thread 0 waiting for event 0x55ab20f67710 before computing node ffn_up_sparse_cpu-31    <- 每次都在这里后对ffn_up_sparse_cpu-31计算前的 eventSync 报错
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1".
0x00007f82ddc7142f in __GI___wait4 (pid=19102, stat_loc=0x0, options=0, usage=0x0) at ../sysdeps/unix/sysv/linux/wait4.c:30
30      ../sysdeps/unix/sysv/linux/wait4.c: No such file or directory.
#0  0x00007f82ddc7142f in __GI___wait4 (pid=19102, stat_loc=0x0, options=0, usage=0x0) at ../sysdeps/unix/sysv/linux/wait4.c:30
30      in ../sysdeps/unix/sysv/linux/wait4.c
#1  0x000055e5bbcb9c36 in ggml_print_backtrace () at /root/workspace/llama.cpp/ggml/src/ggml.c:194
194             waitpid(child_pid, NULL, 0);
#2  0x000055e5bbcb9d5d in ggml_abort (file=0x55e5bbd65df0 "/root/workspace/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu", line=106, fmt=0x55e5bbd65de5 "CUDA error") at /root/workspace/llama.cpp/ggml/src/ggml.c:215
215         ggml_print_backtrace();
#3  0x000055e5bb78b30b in ggml_cuda_error (stmt=0x55e5bbd68ac0 "cudaEventSynchronize((cudaEvent_t)event->context)", func=0x55e5bbd68a90 "ggml_backend_cuda_device_event_synchronize", file=0x55e5bbd65df0 "/root/workspace/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu", line=3502, msg=0x7f82ea88db00 "an illegal memory access was encountered") at /root/workspace/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:106
106         GGML_ABORT(GGML_CUDA_NAME " error");
#4  0x000055e5bb796ffd in ggml_backend_cuda_device_event_synchronize (dev=0x55e5edf90550, event=0x55e600151c40) at /root/workspace/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3502
3502        CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
#5  0x000055e5bbcd2138 in ggml_backend_event_synchronize (event=0x55e600151c40) at /root/workspace/llama.cpp/ggml/src/ggml-backend.cpp:439
439         event->device->iface.event_synchronize(event->device, event);
#6  0x000055e5bb6d5631 in ggml_graph_compute_thread (data=0x55e600355740) at /root/workspace/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:3863
3863                        ggml_backend_event_synchronize(parallel_event);
#7  0x000055e5bb6d68ff in ggml_graph_compute._omp_fn.0 () at /root/workspace/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:4165
4165                ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
#8  0x00007f82eac44a16 in GOMP_parallel () from /usr/lib/x86_64-linux-gnu/libgomp.so.1
#9  0x000055e5bb6d5c58 in ggml_graph_compute (cgraph=0x55e5eeb39bb8, cplan=0x7fffa57a8950) at /root/workspace/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:4156
```

## 他妈的搞不定，先把 CPU 后端的 sync 放在 splits loop 里面先了



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
