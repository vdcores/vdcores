#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm

#include "context.cuh"
#include "type.cuh"

template<typename T>
__device__ __forceinline__ T silu_and_mul(T x, T mul) {
    return (x / (T(1) + expf(-x))) * mul;
}

template<typename data_t>
__device__ __forceinline__
typename F16Traits<data_t>::vec2_t
silu_and_mul(
    typename F16Traits<data_t>::vec2_t x,
    typename F16Traits<data_t>::vec2_t mul)
{
    using Tr = F16Traits<data_t>;
    // Convert to float2
    float2 xf  = Tr::to_float2(x);
    float2 mf  = Tr::to_float2(mul);

    // SiLU in FP32: x / (1 + exp(-x))
    float2 silu;
    silu.x = xf.x / (1.0f + expf(-xf.x));
    silu.y = xf.y / (1.0f + expf(-xf.y));

    // Multiply
    float2 out;
    out.x = silu.x * mf.x;
    out.y = silu.y * mf.y;

    // Convert back to half2
    return Tr::from_float2(out);
}

// make output size a argument for flexiblity now
template<uint64_t K,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_silu_K_half2_inplace(
    void *base,
    const MInst *st_insts,
    const int num_token,
    const int output_size,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    assert(K % 2 == 0 && "load not divisible by 2 for half2");
    assert(output_size % 2 == 0 && "store not divisible by 2 for half2");
    const int thread_id = threadIdx.x;
    const int numComputeThreads = numThreadsPerWarp * numComputeWarps;

    // gate mem will be used for output
    const int gate_addr_slot = m2c.template pop<0>();
    half2* __restrict__ gate_ptr = (half2*)slot_2_glob_ptr(st_insts, gate_addr_slot);
    const int up_addr_slot = m2c.template pop<0>();
    half2* __restrict__ up_ptr = (half2*)slot_2_glob_ptr(st_insts, up_addr_slot);

    #pragma unroll
    for (int token_id = 0; token_id < num_token; token_id++) {
        half2* input_ptr = gate_ptr + token_id * (K / 2);
        half2* mul_ptr = up_ptr + token_id * (K / 2);
        #pragma unroll
        for (int i = thread_id; i < output_size / 2; i += numComputeThreads) {
            half2 x = input_ptr[i];
            half2 mul = mul_ptr[i];
            half2 out = silu_and_mul<half>(x, mul);
            input_ptr[i] = out; 
        }
    }
    c2m.template push<31, true, false>(thread_id, gate_addr_slot);
    // pure load dont push back
    // c2m.template push<31, true, false>(thread_id, up_addr_slot);
}

template<uint64_t K,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_silu_K_half2(
    void *base,
    const MInst *st_insts,
    const int num_token,
    const int output_size,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    assert(K % 2 == 0 && "load not divisible by 2 for half2");
    assert(output_size % 2 == 0 && "store not divisible by 2 for half2");
    const int thread_id = threadIdx.x;
    const int numComputeThreads = numThreadsPerWarp * numComputeWarps;

    // gate mem will be used for output
    const int gate_addr_slot = m2c.template pop<0>();
    half2* __restrict__ gate_ptr = (half2*)slot_2_glob_ptr(st_insts, gate_addr_slot);
    const int up_addr_slot = m2c.template pop<0>();
    half2* __restrict__ up_ptr = (half2*)slot_2_glob_ptr(st_insts, up_addr_slot);
    const int out_addr_slot = m2c.template pop<0>();
    half2* __restrict__ out_ptr = (half2*)slot_2_glob_ptr(st_insts, out_addr_slot);

    #pragma unroll
    for (int token_id = 0; token_id < num_token; token_id++) {
        half2* input_ptr = gate_ptr + token_id * (K / 2);
        half2* mul_ptr = up_ptr + token_id * (K / 2);
        half2* output_ptr = out_ptr + token_id * (K / 2);
        #pragma unroll
        for (int i = thread_id; i < output_size / 2; i += numComputeThreads) {
            half2 x = input_ptr[i];
            half2 mul = mul_ptr[i];
            half2 out = silu_and_mul<half>(x, mul);
            output_ptr[i] = out; 
        }
    }
    // c2m.template push<31, true, false>(thread_id, gate_addr_slot);
    // c2m.template push<31, true, false>(thread_id, up_addr_slot);
    c2m.template push<31, true, false>(thread_id, out_addr_slot);
}

template<int K, typename Layout,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_silu_smem(
    const int N,
    const Layout& layout,
    void *base,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    using data_t = __nv_bfloat16;
    using fetch_t = __nv_bfloat162;

    const int slot_out = m2c.pop();
    fetch_t *sOut = (fetch_t *)get_slot_address(base, extract(slot_out));
    const int slot_gate = m2c.pop();
    fetch_t *sGate = (fetch_t *)get_slot_address(base, extract(slot_gate));
    const int slot_up = m2c.pop();
    fetch_t *sUp = (fetch_t *)get_slot_address(base, extract(slot_up));

    auto sO_vec = make_tensor(make_smem_ptr(sOut), layout);
    auto sG_vec = make_tensor(make_smem_ptr(sGate), layout);
    auto sU_vec = make_tensor(make_smem_ptr(sUp), layout);

    // unroll to fill the latency of smem load/store
    #pragma unroll
    for (int i = threadIdx.x; i < K / 2 * N; i += numComputeWarps * numThreadsPerWarp) {
        // each thread load 1 register in one non-unrolled loop
        const int in = i / (K / 2);
        const int ik = i % (K / 2);
        sO_vec(ik,in) = silu_and_mul<data_t>(sG_vec(ik,in), sU_vec(ik,in));
    }

    c2m.template push<0, true>(threadIdx.x, slot_out);
    // a write push ensures the threading order
    c2m.template push<0>(threadIdx.x, slot_gate | slot_up);
}


template<int K,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_silu_smem_1D(
    const int N,
    void *base,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    using data_t = __nv_bfloat16;
    using fetch_t = __nv_bfloat162;

    const int slot_out = m2c.pop();
    fetch_t *sOut = (fetch_t *)get_slot_address(base, extract(slot_out));
    const int slot_gate = m2c.pop();
    fetch_t *sGate = (fetch_t *)get_slot_address(base, extract(slot_gate));
    const int slot_up = m2c.pop();
    fetch_t *sUp = (fetch_t *)get_slot_address(base, extract(slot_up));

    // unroll to fill the latency of smem load/store
    #pragma unroll
    for (int i = threadIdx.x; i < K / 2 * N; i += numComputeWarps * numThreadsPerWarp) {
        // each thread load 1 register in one non-unrolled loop
        sOut[i] = silu_and_mul<data_t>(sGate[i], sUp[i]);
    }

    c2m.template push<0, true>(threadIdx.x, slot_out);
    // a write push ensures the threading order
    c2m.template push<0>(threadIdx.x, slot_gate | slot_up);
}
