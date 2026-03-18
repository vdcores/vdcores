#pragma once

#include <cstdint>
#include <cuda/ptx>

template<typename T> struct F16Traits;

template<> struct F16Traits<half> {
    using vec2_t = half2;
    static __device__ __forceinline__ float2 to_float2(half2 v)    { return __half22float2(v); }
    static __device__ __forceinline__ half2  from_float2(float2 v) { return __float22half2_rn(v); }
    static __device__ __forceinline__ float  to_float(half e)    { return __half2float(e); }
};

template<> struct F16Traits<__nv_bfloat16> {
    using vec2_t = __nv_bfloat162;
    static __device__ __forceinline__ float2        to_float2(__nv_bfloat162 v) { return __bfloat1622float2(v); }
    static __device__ __forceinline__ __nv_bfloat162 from_float2(float2 v)      { return __float22bfloat162_rn(v); }
    static __device__ __forceinline__ float          to_float(__nv_bfloat16 e){ return __bfloat162float(e); }
};

template<>
struct F16Traits<cutlass::bfloat16_t> : F16Traits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(cutlass::bfloat16_t e) {
        return F16Traits<__nv_bfloat16>::to_float(static_cast<__nv_bfloat16>(e));
    }
};