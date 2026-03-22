#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cute/algorithm/tensor_reduce.hpp>     // cute::reduce
#include <cute/algorithm/tensor_algorithms.hpp>     // cute::reduce
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/axpby.hpp> // cute::axpby
#include <cute/layout.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "context.cuh"
#include "type.cuh"

using namespace cute;

template<int NUM_HEAD, int HEAD_DIM, int N_COMPUTE_THREAD, typename data_t>
struct NormRope {
    using Tr = F16Traits<data_t>;
    using vec2_t = typename Tr::vec2_t;
    static constexpr int num_thread_per_token = (HEAD_DIM / 2 < N_COMPUTE_THREAD) ? HEAD_DIM / 2 : N_COMPUTE_THREAD;
    static constexpr int token_group_size = (N_COMPUTE_THREAD + num_thread_per_token - 1) / num_thread_per_token;
    static_assert(num_thread_per_token % 32 == 0, "to simplify warp-level reduce");
    static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be vectorizable");
    static_assert(N_COMPUTE_THREAD % num_thread_per_token == 0, "easier workload partition");
    static_assert(token_group_size <= 64, "to fit into reduce buffer");

    template<typename Tensor>
    __device__ __forceinline__
    static void fused_norm_and_rope(
        Tensor input,
        const int total_num_token, // this is across all heads in this SM
        const int token_glob_ofst,
        float* smem_reduce,
        const float epsilon, 
        const vec2_t* norm_weight,
        const vec2_t* rope_table,
        const bool need_rope
    ) {
        // assume input is shape [N * num_head, head_dim]
        const int warp_id = __compute_tid() / 32;
        const int lane_id = __compute_tid() % 32;
        const int ofst_in_token_group = __compute_tid() / num_thread_per_token;
        const int lane_in_one_token = __compute_tid() % num_thread_per_token;
        constexpr int num_warp_per_token = num_thread_per_token / 32;
        constexpr int half_vec_count = HEAD_DIM / 4;

        #pragma unroll
        for (int r = ofst_in_token_group; r < total_num_token; r += token_group_size) {
            // norm
            float sum = 0.0f;
            #pragma unroll
            for (int i = lane_in_one_token; i < num_thread_per_token; i += num_thread_per_token) {
                __nv_bfloat162 val = input(r, i);
                float2 val_f32 = __bfloat1622float2(val);
                sum += val_f32.x * val_f32.x + val_f32.y * val_f32.y;
            }

            // reduce within warp
            for (int offset = 32 / 2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
            }
            if (lane_id == 0)
                smem_reduce[warp_id] = sum;
            // each token group bar independently
            __sync_barrier<num_thread_per_token>(ofst_in_token_group);

            // final reduce by first lane in token group
            if (lane_in_one_token == 0) {
                #pragma unroll
                for (int i = 1; i < num_warp_per_token; i++)
                    sum += smem_reduce[i + ofst_in_token_group * num_warp_per_token];
                smem_reduce[ofst_in_token_group] = sum;
            }
            __sync_barrier<num_thread_per_token>(ofst_in_token_group);
            float rms_rcp = rsqrtf(smem_reduce[ofst_in_token_group] / float(HEAD_DIM) + epsilon);

            // Apply norm first so rotate_half can safely read both halves
            // from shared memory before any thread overwrites its partner lane.
            const int logical_token_id_in_glob = token_glob_ofst + r / NUM_HEAD;
            #pragma unroll
            for (int i = lane_in_one_token; i < num_thread_per_token; i += num_thread_per_token) {
                __nv_bfloat162 val = input(r, i);
                float2 val_f32 = __bfloat1622float2(val);

                // apply norm
                val_f32.x = val_f32.x * rms_rcp;
                val_f32.y = val_f32.y * rms_rcp;
                if (norm_weight != nullptr) {
                    float2 weight_f32 = __bfloat1622float2(norm_weight[i]);
                    val_f32.x = val_f32.x * weight_f32.x;
                    val_f32.y = val_f32.y * weight_f32.y;
                }
                input(r, i) = __float22bfloat162_rn(val_f32);
            }
            __sync_barrier<num_thread_per_token>(ofst_in_token_group);

            if (need_rope) {
                #pragma unroll
                for (int i = lane_in_one_token; i < half_vec_count; i += num_thread_per_token) {
                    float2 first_half = __bfloat1622float2(input(r, i));
                    float2 second_half = __bfloat1622float2(input(r, i + half_vec_count));
                    const int rope_row_base = HEAD_DIM / 2 * logical_token_id_in_glob;
                    float2 cos_pair = __bfloat1622float2(rope_table[rope_row_base + i]);
                    float2 sin_pair = __bfloat1622float2(rope_table[rope_row_base + half_vec_count + i]);

                    input(r, i) = __float22bfloat162_rn({
                        first_half.x * cos_pair.x - second_half.x * sin_pair.x,
                        first_half.y * cos_pair.y - second_half.y * sin_pair.y,
                    });
                    input(r, i + half_vec_count) = __float22bfloat162_rn({
                        second_half.x * cos_pair.x + first_half.x * sin_pair.x,
                        second_half.y * cos_pair.y + first_half.y * sin_pair.y,
                    });
                }
            }
            __sync_barrier<num_thread_per_token>(ofst_in_token_group);
        }
    }

    template<int GLOB_HEAD_DIM, typename Tensor>
    __device__ __forceinline__
    static void gemv_rope(
        Tensor input,
        const int num_token,
        const int hist_len, // assume all token has the same position for now
        const int k_glob_ofst,
        const vec2_t* rope_table
    ) {
        // assume input is shape [64 = H / 2, N]
        // TODO (zijian): assert above, especially if first dim span across heads
        // const int warp_id = __compute_tid() / 32;
        // const int lane_id = __compute_tid() % 32;
        const int ofst_in_token_group = __compute_tid() / num_thread_per_token;
        const int lane_in_one_token = __compute_tid() % num_thread_per_token;

        #pragma unroll
        for (int r = ofst_in_token_group; r < num_token; r += token_group_size) {
            #pragma unroll
            for (int i = lane_in_one_token; i < num_thread_per_token; i += num_thread_per_token) {
                __nv_bfloat162 val = input(i, r);
                float2 val_f32 = __bfloat1622float2(val);

                // apply rope
                const int rope_idx = GLOB_HEAD_DIM/2 * hist_len + i + k_glob_ofst/2;
                float2 cos_sin = __bfloat1622float2(rope_table[rope_idx]);
                float rotated_even = val_f32.x * cos_sin.x - val_f32.y * cos_sin.y;
                float rotated_odd = val_f32.x * cos_sin.y + val_f32.y * cos_sin.x;
                input(i, r) = __float22bfloat162_rn({rotated_even, rotated_odd});
            }
            __sync_barrier<num_thread_per_token>(ofst_in_token_group);
        }
    }
};

template<int N, typename M2CType, typename C2MType>
__device__ __forceinline__
void task_rope_interleaved(
    void *base, M2CType &m2c, C2MType &c2m
) {
    static_assert(N % 2 == 0, "N must be even for vectorized rope");
    using vec_t = __nv_bfloat162;

    int thread_id = threadIdx.x;

    int table_slot = m2c.pop();
    vec_t* table = (vec_t*)get_slot_address(base, extract(table_slot));
    int in_slot = m2c.pop();
    vec_t* in = (vec_t*)get_slot_address(base, extract(in_slot));
    int out_slot = m2c.pop();
    vec_t* out = (vec_t*)get_slot_address(base, extract(out_slot));

    const vec_t zero = make_bfloat162(0, 0);

    #pragma unroll
    for (int i = thread_id; i < N / 2; i += 128)
        out[i] = __hcmadd(in[i], table[i], zero);

    // TODO(zhiyuang): check if we need this fence
    // cuda::ptx::fence_proxy_async(cuda::ptx::space_shared_t{});
    c2m.template push<0, true>(thread_id, out_slot);
    // they cannot be merged, when one of them could be local
    c2m.push(thread_id, in_slot | table_slot);
}
