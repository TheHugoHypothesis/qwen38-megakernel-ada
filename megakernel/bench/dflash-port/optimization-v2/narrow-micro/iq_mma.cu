#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../native_iq_mma.cuh"
#include "../gguf.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_OK(call) do { const auto error = (call); if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); } while (0)

constexpr int K = 5120, N = 64, T = 8, U = K / 32;
constexpr size_t INPUT_BYTES = 2 * T * U * sizeof(int4) + T * U * sizeof(uint2);

__global__ void dots(const uint8_t * gate, int gs, int gt, const uint8_t * up, int us, int ut,
                     const int4 * xa, const int4 * xb, const uint2 * meta, float * output) {
    extern __shared__ __align__(16) uint8_t input[];
    auto * a = reinterpret_cast<int4 *>(input);
    auto * b = a + T * U;
    auto * m = reinterpret_cast<uint2 *>(b + T * U);
    for (int i = threadIdx.x; i < T * U; i += blockDim.x) { a[i] = xa[i]; b[i] = xb[i]; m[i] = meta[i]; }
    __syncthreads();
    const int warp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float g[2], u[2];
    iq_mma_dual(gate + size_t(warp * 8) * gs, gs, gt,
                up + size_t(warp * 8) * us, us, ut, a, b, m, U, g, u);
    const int row = warp * 8 + (lane >> 2), token = 2 * (lane & 3);
    for (int t = 0; t < 2; t++) {
        output[(token + t) * N + row] = g[t];
        output[(T + token + t) * N + row] = u[t];
    }
}

template<int GT, int UT, bool DP4A = false, bool NARROW = false>
__global__ void dots16(const uint8_t * gate, int gs, const uint8_t * up, int us,
                       const int4 * xa, const int4 * xb, const uint2 * meta, float * output) {
    extern __shared__ __align__(16) uint8_t input[];
    auto * a = reinterpret_cast<int4 *>(input); auto * b = a + T * U;
    auto * m = reinterpret_cast<uint2 *>(b + T * U);
    for (int i = threadIdx.x; i < T * U; i += blockDim.x) { a[i] = xa[i]; b[i] = xb[i]; m[i] = meta[i]; }
    __syncthreads();
    const int warp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if constexpr (DP4A) {
        float g[T], u[T];
        iq_dp4a_dual<T, U, GT, UT>(gate + size_t(warp) * gs, up + size_t(warp) * us, a, b, m, g, u);
        for (int t = 0; t < T; t++) {
            for (int shift = 16; shift; shift /= 2) {
                g[t] += __shfl_down_sync(0xffffffffu, g[t], shift);
                u[t] += __shfl_down_sync(0xffffffffu, u[t], shift);
            }
            if (lane == 0) { output[t * N + warp] = g[t]; output[(T + t) * N + warp] = u[t]; }
        }
    } else {
        float g[4], u[4];
        iq_mma16_dual<GT, UT, NARROW>(gate + size_t(warp * 16) * gs, gs, up + size_t(warp * 16) * us, us, a, b, m, U, g, u);
        for (int half = 0; half < 2; half++) for (int t = 0; t < 2; t++) {
            const int row = warp * 16 + (lane >> 2) + 8 * half, token = 2 * (lane & 3) + t;
            output[token * N + row] = g[2 * half + t];
            output[(T + token) * N + row] = u[2 * half + t];
        }
    }
}

template<int GT, int UT> static void launch_pair(int mode, const uint8_t * gate, int gs, const uint8_t * up, int us,
                                               const int4 * a, const int4 * b, const uint2 * m, float * out) {
    if (mode == 1) {
        CUDA_OK(cudaFuncSetAttribute(dots16<GT, UT>, cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_BYTES));
        dots16<GT, UT><<<N / 64, 128, INPUT_BYTES>>>(gate, gs, up, us, a, b, m, out);
    } else if (mode == 3) {
        CUDA_OK(cudaFuncSetAttribute(dots16<GT, UT, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_BYTES));
        dots16<GT, UT, false, true><<<N / 64, 128, INPUT_BYTES>>>(gate, gs, up, us, a, b, m, out);
    } else if constexpr ((GT == 18 || GT == 21 || GT == 23) && (UT == 18 || UT == 21 || UT == 23)) {
        CUDA_OK(cudaFuncSetAttribute(dots16<GT, UT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_BYTES));
        dots16<GT, UT, true><<<N / 4, 128, INPUT_BYTES>>>(gate, gs, up, us, a, b, m, out);
    }
}

template<int GT> static void launch_up(int ut, int mode, const uint8_t * gate, int gs, const uint8_t * up, int us,
                                     const int4 * a, const int4 * b, const uint2 * m, float * out) {
#define PAIR(IQ) case IQ: launch_pair<GT, IQ>(mode, gate, gs, up, us, a, b, m, out); break;
    switch (ut) { PAIR(16) PAIR(17) PAIR(18) PAIR(21) PAIR(22) PAIR(23) }
#undef PAIR
}

static void launch_typed(int gt, int ut, int mode, const uint8_t * gate, int gs, const uint8_t * up, int us,
                         const int4 * a, const int4 * b, const uint2 * m, float * out) {
#define GATE(IQ) case IQ: launch_up<IQ>(ut, mode, gate, gs, up, us, a, b, m, out); break;
    switch (gt) { GATE(16) GATE(17) GATE(18) GATE(21) GATE(22) GATE(23) }
#undef GATE
}

template<class V> static V * copy(const std::vector<V> & host) {
    V * device; CUDA_OK(cudaMalloc(&device, host.size() * sizeof(V)));
    CUDA_OK(cudaMemcpy(device, host.data(), host.size() * sizeof(V), cudaMemcpyHostToDevice)); return device;
}

struct Weight {
    int type, stride;
    uint8_t * device;
    std::vector<float> decoded;
};

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: iq_mma_test target.gguf");
        gguf::File file(argv[1]);
        std::vector<Weight> weights;
        for (int type : {16, 17, 18, 21, 22, 23}) {
            const gguf::Tensor * tensor = nullptr;
            for (const auto & entry : file.tensors) {
                const auto & t = entry.second;
                if (t.type == unsigned(type) && t.ne.size() == 2 && t.ne[0] == K && t.ne[1] >= 2 * N &&
                    (t.name.find("ffn_gate.weight") != std::string::npos || t.name.find("ffn_up.weight") != std::string::npos)) {
                    tensor = &t; break;
                }
            }
            if (!tensor) throw std::runtime_error("missing real FFN matrix for IQ type " + std::to_string(type));
            const int stride = gguf::type_row_bytes(type, K);
            std::vector<uint8_t> bytes(tensor->data, tensor->data + size_t(2 * N) * stride);
            Weight w{type, stride, copy(bytes), std::vector<float>(2 * N * K)};
            const auto * traits = ggml_get_type_traits(ggml_type(type));
            for (int row = 0; row < 2 * N; row++) traits->to_float(tensor->data + size_t(row) * stride, w.decoded.data() + row * K, K);
            weights.push_back(std::move(w));
        }
        CUDA_OK(cudaFuncSetAttribute(dots, cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_BYTES));
        float * output; CUDA_OK(cudaMalloc(&output, 2 * T * N * sizeof(float)));
        double worst_nrmse = 0, worst_abs = 0; int values = 0;
        for (int trial = 0; trial < 2; trial++) {
            std::vector<int4> a(T * U), b(T * U);
            std::vector<uint2> meta(T * U);
            std::vector<float> x(T * K);
            for (int t = 0; t < T; t++) for (int e = 0; e < K; e += 16) {
                const int unit = 8 * (e / 256) + 2 * ((e & 255) / 64) + ((e & 31) / 16), part = (e & 63) / 32;
                const __half scale = __float2half(trial ? .002f * (1 + (e / 16 + 3 * t) % 11) : 0.f);
                int sum = 0;
                for (int j = 0; j < 16; j++) {
                    const int q = (41 * (e + j) + 73 * t + 17) % 255 - 127;
                    reinterpret_cast<int8_t *>((part ? b : a).data())[size_t(t * U + unit) * 16 + j] = int8_t(q);
                    x[t * K + e + j] = q * __half2float(scale); sum += q;
                }
                reinterpret_cast<__half *>(&meta[t * U + unit].x)[part] = scale;
                reinterpret_cast<int16_t *>(&meta[t * U + unit].y)[part] = int16_t(sum);
            }
            auto * da = copy(a); auto * db = copy(b); auto * dm = copy(meta);
            std::vector<std::vector<double>> references;
            for (const auto & w : weights) {
                std::vector<double> ref(2 * T * N);
                for (int side = 0; side < 2; side++) for (int t = 0; t < T; t++) for (int r = 0; r < N; r++)
                    for (int e = 0; e < K; e++) ref[(side * T + t) * N + r] += double(w.decoded[(side * N + r) * K + e]) * x[t * K + e];
                references.push_back(std::move(ref));
            }
            for (size_t gi = 0; gi < weights.size(); gi++) for (size_t ui = 0; ui < weights.size(); ui++) {
                const auto & g = weights[gi]; const auto & u = weights[ui];
                for (int mode = 0; mode < 4; mode++) {
                if (mode == 2 && !((g.type == 18 || g.type == 21 || g.type == 23) && (u.type == 18 || u.type == 21 || u.type == 23))) continue;
                if (mode == 0) dots<<<N / 32, 128, INPUT_BYTES>>>(g.device, g.stride, g.type, u.device + size_t(N) * u.stride,
                                                  u.stride, u.type, da, db, dm, output);
                else launch_typed(g.type, u.type, mode, g.device, g.stride, u.device + size_t(N) * u.stride, u.stride, da, db, dm, output);
                CUDA_OK(cudaGetLastError());
                std::vector<float> result(2 * T * N);
                CUDA_OK(cudaMemcpy(result.data(), output, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double error2 = 0, ref2 = 0, maximum = 0;
                for (int i = 0; i < 2 * T * N; i++) {
                    if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite IQ MMA dot");
                    const double ref = i < T * N ? references[gi][i] : references[ui][i];
                    const double delta = result[i] - ref;
                    error2 += delta * delta; ref2 += ref * ref; maximum = std::max(maximum, std::abs(delta)); values++;
                }
                const double nrmse = ref2 ? std::sqrt(error2 / ref2) : std::sqrt(error2);
                const double rms = std::sqrt(ref2 / result.size());
                worst_nrmse = std::max(worst_nrmse, nrmse); worst_abs = std::max(worst_abs, maximum);
                if (nrmse > 1e-5 || maximum > 1e-4 * (1 + rms)) {
                    fprintf(stderr, "FAIL trial%d mode%d types%d/%d NRMSE%.9g max%.9g RMS%.9g\n", trial, mode, g.type, u.type, nrmse, maximum, rms);
                    throw std::runtime_error("IQ MMA differs from CPU GGML/double oracle");
                }
                }
            }
            CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dm));
        }
        for (auto & w : weights) CUDA_OK(cudaFree(w.device));
        CUDA_OK(cudaFree(output));
        printf("PASS %d IQ values, IMMA8/IMMA16/IMMA16-narrow 36 type pairs + DP4A 9 pairs, zero/nonzero Q16 inputs; max NRMSE %.9g abs %.9g\n", values, worst_nrmse, worst_abs);
    } catch (const std::exception & error) { fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
    return 0;
}
