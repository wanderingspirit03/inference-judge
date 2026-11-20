// A small real GPU workload for testing ./try and ./profile on the card:
// a memory-bound copy kernel and a compute-heavy FMA kernel, each run 200
// times on 1 GiB.  Build: hipcc -O3 --offload-arch=gfx950 -o hip_probe hip_probe.cpp
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>

__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i];
}

__global__ void fma_kernel(float* __restrict__ x, size_t n, int iters) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], w = 1.0001f;
    for (int k = 0; k < iters; ++k) v = fmaf(v, w, 0.5f);
    x[i] = v;
}

int main() {
    const size_t bytes = 1ull << 30;
    const size_t n4 = bytes / sizeof(float4);
    float4 *a, *b;
    if (hipMalloc(&a, bytes) != hipSuccess || hipMalloc(&b, bytes) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
    hipMemset(a, 0, bytes);
    hipDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 200; ++r) copy_kernel<<<8192, 256>>>(a, b, n4);
    hipDeviceSynchronize();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("copy: %.1f GB/s (read+write)\n", 200.0 * 2 * bytes / s / 1e9);
    const size_t nf = (64ull << 20);
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 20; ++r) fma_kernel<<<(nf + 255) / 256, 256>>>((float*)a, nf, 2000);
    hipDeviceSynchronize();
    s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("fma: %.1f TFLOP/s (fp32)\n", 20.0 * nf * 2000 * 2 / s / 1e12);
    hipFree(a); hipFree(b);
    return 0;
}
