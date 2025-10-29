#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_sparkinfer_layer_cache ggml_sparkinfer_layer_cache;

struct ggml_backend_cuda_context;
struct ggml_tensor;

void reload_neurons(ggml_sparkinfer_layer_cache *      obj,
                    struct ggml_backend_cuda_context * ctx,
                    struct ggml_tensor *               dst);

#ifdef __cplusplus
}
#endif
