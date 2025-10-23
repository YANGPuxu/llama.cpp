// this is a Sparkinfer function wrapper, create an API to call llama-sparkinfer function to avoid dependency issue

#pragma once 
#include "ggml.h" 

struct reload_plan_result {
    int * slot_to_evict;
    int * slot_to_load;
    size_t n_reload;

    reload_plan_result() : slot_to_evict(nullptr), slot_to_load(nullptr), n_reload(0) {}
};
typedef struct reload_plan_result reload_plan_result;
  
#ifdef __cplusplus  
extern "C" {  
#endif  

typedef struct ggml_spif_context ggml_spif_context; 
  
reload_plan_result * ggml_spif_reload_plan(ggml_spif_context * ctx, ggml_tensor * tensor);  
  
#ifdef __cplusplus  
}  
#endif
