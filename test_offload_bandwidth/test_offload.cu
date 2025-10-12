// test_offload.cu
// Compile:
/* 
nvcc -O2 -arch=sm_86 test_offload.cu -lnvToolsExt -o test_offload
./test_offload
*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>
#include <functional>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

const uint32_t colors[] = {
    0xff00ff00, 0xff0000ff, 0xffffff00,
    0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff
};
const int num_colors = sizeof(colors)/sizeof(uint32_t);
nvtxRangeId_t nvtx_outer_start(const char* name, int color_id=0) {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = colors[color_id % num_colors];
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    return nvtxRangeStartEx(&attr);
}
void nvtx_outer_end(nvtxRangeId_t id) {
    nvtxRangeEnd(id);
}

// Fixed total offload neurons
const int N_OFF = 5504;

using clk = std::chrono::high_resolution_clock;

float cuda_time_ms_for_func(std::function<void()> func, int repeat = 10, int warmup = 3, const char* nvtx_name = nullptr, int nvtx_color=0) {
    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup (not timed)
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // If nvtx_name provided, start outer NVTX (we'll wrap the measured repeats)
    nvtxRangeId_t rid = 0;
    if (nvtx_name) rid = nvtx_outer_start(nvtx_name, nvtx_color);

    std::vector<float> times;
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        func();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    if (nvtx_name) nvtx_outer_end(rid);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    return avg;
}

void append_csv(const std::string &fname, const std::string &line) {
    std::ofstream ofs(fname, std::ios::app);
    if (!ofs) return;
    ofs << line << std::endl;
    ofs.close();
}

int main(int argc, char** argv) {
    // Command line args:
    // argv[1] = n (neurons per expert)
    // argv[2] = len (per-neuron length)
    // argv[3] = streams (stream count for multi-stream)
    // argv[4] = repeat
    // argv[5] = warmup
    int n = 16;
    int len = 4096;
    int streams = 4;
    int repeat = 10;
    int warmup = 3;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) len = atoi(argv[2]);
    if (argc >= 4) streams = atoi(argv[3]);
    if (argc >= 5) repeat = atoi(argv[4]);
    if (argc >= 6) warmup = atoi(argv[5]);

    if (n <= 0 || len <= 0 || streams <= 0) {
        std::cerr << "Invalid args. n,len,streams must be >0\n";
        return 1;
    }

    const int64_t S_neuron = int64_t(len) * int64_t(sizeof(__half)); // bytes per neuron
    const int64_t N_off = N_OFF;
    const int64_t S_total = S_neuron * N_off;

    // compute number of experts
    int N_exp = (N_off + n - 1) / n; // ceil
    // For last expert, effective size may be smaller
    int last_expert_neurons = N_off - (N_exp - 1) * n;

    std::cout << "=== Offload experts test ===\n";
    std::cout << "Total neurons to offload: " << N_off << "\n";
    std::cout << "Per-neuron length: " << len << " (fp16)\n";
    std::cout << "Bytes per neuron: " << S_neuron << "\n";
    std::cout << "Neurons per expert (n): " << n << "\n";
    std::cout << "Experts count (N_exp): " << N_exp << " (last expert neurons = " << last_expert_neurons << ")\n";
    std::cout << "Total bytes to copy: " << (S_total / (1024.0*1024.0)) << " MB\n";
    std::cout << "Streams(default): " << streams << ", repeat: " << repeat << ", warmup: " << warmup << "\n";

    // Allocate pinned host buffer arranged as contiguous experts: [expert0][expert1]...
    __half* h_buf = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_buf, S_total));
    // Fill with some pattern
    for (int64_t i = 0; i < N_off * (int64_t)len; ++i) {
        float v = float(i % 1000) / 1000.0f;
        h_buf[i] = __float2half(v);
    }

    // Allocate device buffer (destination contiguous)
    __half* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_buf, S_total));

    // Create streams pool
    std::vector<cudaStream_t> stream_pool;
    stream_pool.resize(streams);
    for (int i = 0; i < streams; ++i) CHECK_CUDA(cudaStreamCreate(&stream_pool[i]));

    // Helper: compute pointer offsets
    auto host_expert_ptr = [&](int expert_idx) -> __half* {
        int64_t offset_neurons = int64_t(expert_idx) * int64_t(n);
        int64_t offset_elems = offset_neurons * int64_t(len);
        return h_buf + offset_elems;
    };
    auto dev_expert_ptr = [&](int expert_idx) -> __half* {
        int64_t offset_neurons = int64_t(expert_idx) * int64_t(n);
        int64_t offset_elems = offset_neurons * int64_t(len);
        return d_buf + offset_elems;
    };
    auto expert_size_bytes = [&](int expert_idx) -> int64_t {
        int eff_neurons = (expert_idx == N_exp - 1) ? last_expert_neurons : n;
        return int64_t(eff_neurons) * int64_t(len) * int64_t(sizeof(__half));
    };

    // ---------- Strategy A: batched_async_multi_stream (主方案) ----------
    auto batched_async_multi_stream = [&]() {
        // For each expert i, issue cudaMemcpyAsync of expert_size using stream_pool[i % streams]
        for (int i = 0; i < N_exp; ++i) {
            int s = i % streams;
            int64_t bytes = expert_size_bytes(i);
            CHECK_CUDA(cudaMemcpyAsync(dev_expert_ptr(i),
                                       host_expert_ptr(i),
                                       bytes,
                                       cudaMemcpyHostToDevice,
                                       stream_pool[s]));
        }
        // sync all streams
        for (int s = 0; s < streams; ++s) CHECK_CUDA(cudaStreamSynchronize(stream_pool[s]));
    };

    float t_batched_ms = cuda_time_ms_for_func(batched_async_multi_stream, repeat, warmup, "batched_async_multi_stream", 0);
    double bw_batched = (double(S_total) / (t_batched_ms / 1000.0)) / 1e9;

    std::cout << "[batched_async_multi_stream] avg_time_ms = " << t_batched_ms
              << ", BW = " << bw_batched << " GB/s\n";

    // ---------- Strategy B: naive_per_neuron_async ----------
    // Use single stream (stream_pool[0]) and do per-neuron async copies
    auto naive_per_neuron_async = [&]() {
        cudaStream_t s = stream_pool[0];
        for (int e = 0; e < N_exp; ++e) {
            int eff_neurons = (e == N_exp - 1) ? last_expert_neurons : n;
            int64_t bytes_per_neuron = S_neuron;
            // per-neuron copies within this expert
            __half* hptr = host_expert_ptr(e);
            __half* dptr = dev_expert_ptr(e);
            for (int ni = 0; ni < eff_neurons; ++ni) {
                CHECK_CUDA(cudaMemcpyAsync(dptr + int64_t(ni)*len,
                                           hptr + int64_t(ni)*len,
                                           bytes_per_neuron,
                                           cudaMemcpyHostToDevice,
                                           s));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(s));
    };

    float t_naive_ms = cuda_time_ms_for_func(naive_per_neuron_async, repeat, warmup, "naive_per_neuron_async", 1);
    double bw_naive = (double(S_total) / (t_naive_ms / 1000.0)) / 1e9;

    std::cout << "[naive_per_neuron_async] avg_time_ms = " << t_naive_ms
              << ", BW = " << bw_naive << " GB/s\n";

    // Append results to CSV
    {
        char buf[1024];
        // header if file not exists
        std::string csv = "test_offload.csv";
        std::ifstream ifs(csv);
        bool exists = ifs.good();
        ifs.close();
        if (!exists) {
            append_csv(csv, "strategy,n,len,streams,N_off,N_exp,total_MB,repeat,warmup,avg_ms,GBps");
        }
        // batched
        snprintf(buf, sizeof(buf), "batched_async_multi_stream,%d,%d,%d,%lld,%d,%.3f,%d,%d,%.6f,%.6f",
                 n, len, streams, (long long)N_off, N_exp, double(S_total)/(1024.0*1024.0), repeat, warmup, t_batched_ms, bw_batched);
        append_csv(csv, buf);
        // naive
        snprintf(buf, sizeof(buf), "naive_per_neuron_async,%d,%d,%d,%lld,%d,%.3f,%d,%d,%.6f,%.6f",
                 n, len, streams, (long long)N_off, N_exp, double(S_total)/(1024.0*1024.0), repeat, warmup, t_naive_ms, bw_naive);
        append_csv(csv, buf);
        std::cout << "Results appended to test_offload.csv\n";
    }

    // cleanup
    for (auto s : stream_pool) cudaStreamDestroy(s);
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));

    return 0;
}
