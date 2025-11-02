// this is a Sparkinfer function wrapper, create an API to call llama-sparkinfer function to avoid dependency issue

#pragma once 
#include "ggml.h" 

struct reload_plan_result {
    std::vector<int> groups_to_reload;
    std::vector<int> slots_for_evict;
};
typedef struct reload_plan_result reload_plan_result;

#ifdef __cplusplus  
extern "C" {  
#endif  

struct ggml_backend_cuda_context;  // forward declaration
typedef struct ggml_spif_context ggml_spif_context; 
  
bool ggml_spif_reload_plan(ggml_spif_context * ctx, ggml_tensor * tensor);  
bool ggml_spif_reload_exec(ggml_spif_context * ctx, ggml_tensor * tensor, struct ggml_backend_cuda_context * cuda_ctx);  
  
#ifdef __cplusplus  
}  
#endif
