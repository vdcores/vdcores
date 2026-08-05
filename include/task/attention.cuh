#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/atom/copy_traits_sm100.hpp>
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
#include <cutlass/arch/barrier.h>

#include "context.cuh"
#include "rms_norm.cuh"
#include "rope.cuh"

#define TMP_QK 0
#define TMP_ROW_MAX 0
#define TMP_ROW_SUM 0
#define TMP_EXP_P 0
#define TMP_PV 0

namespace {

using namespace cute;

template <int M, int N, typename Layout_TV, typename Tensor_C>
__device__ __forceinline__ auto acc_get_mn_view(
    Layout_TV const& layout_tv,
    Tensor_C& tensor_fragC
) {
    auto layout_m = make_layout(
        make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<1>{}, Int<0>{}));
    auto tv2m = composition(layout_m, layout_tv);
    auto reg2m = coalesce(select<1>(tv2m));
    auto ns = nullspace(reg2m); // get the "kern" of the mapping
    auto tiled = logical_divide(coalesce(tensor_fragC), ns);
    return tiled;
}


template <int M, int N, typename Layout_TV>
__device__ __forceinline__ auto get_tv2m_layout(Layout_TV const& layout_tv) {
    auto layout_m = make_layout(
        make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<1>{}, Int<0>{}));
    auto tv2m = composition(layout_m, layout_tv);
    auto tv2m_coal = make_layout(
        coalesce(filter_zeros(select<0>(tv2m))),
        coalesce(filter_zeros(select<1>(tv2m)))
    );
    return tv2m_coal;
}

template <int Width, class Op, typename TensorT>
__device__ __forceinline__ void butterfly_reduce(TensorT &val, Op op) {
    constexpr int offset = Width / 2;
    #pragma unroll
    for (int r = 0; r < size(val); ++r) {
        #pragma unroll
        for (int c = offset; c > 0; c /= 2) {
            val(r) = op(val(r), __shfl_xor_sync(0xFFFFFFFF, val(r), c) );
        }
    }
}

template <int HEAD_DIM, int N_COMPUTE_THREAD, typename TensorVec, typename Vec2T>
__device__ __forceinline__ void rms_affine_rope_rows(
    TensorVec& input,
    const int total_num_token,
    float* smem_reduce,
    const float epsilon,
    const Vec2T* affine_weight,
    const Vec2T* rope_row,
    const bool need_norm,
    const bool need_rope
) {
    constexpr int num_thread_per_token = (HEAD_DIM / 2 < N_COMPUTE_THREAD) ? HEAD_DIM / 2 : N_COMPUTE_THREAD;
    constexpr int token_group_size = N_COMPUTE_THREAD / num_thread_per_token;
    constexpr int num_warp_per_token = num_thread_per_token / 32;
    static_assert(num_thread_per_token % 32 == 0, "to simplify warp-level reduce");
    static_assert(N_COMPUTE_THREAD % num_thread_per_token == 0, "token groups must be uniform");

    const int thread_id = __compute_tid();
    const int warp_id = thread_id / 32;
    const int lane_id = __compute_tid() % 32;
    const int token_group = thread_id / num_thread_per_token;
    const int lane_in_one_token = thread_id % num_thread_per_token;

    const int num_row_groups = (total_num_token + token_group_size - 1) / token_group_size;
    for (int row_group = 0; row_group < num_row_groups; ++row_group) {
        const int r = row_group * token_group_size + token_group;
        const bool row_valid = r < total_num_token;
        const int i = lane_in_one_token;
        float sum = 0.0f;
        if (need_norm) {
            if (row_valid) {
                auto val = __bfloat1622float2(input(r, i));
                sum += val.x * val.x + val.y * val.y;
            }

            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
            if (lane_id == 0)
                smem_reduce[warp_id] = sum;
            __sync_compute_group(N_COMPUTE_THREAD);

            if (lane_in_one_token == 0 && row_valid) {
                #pragma unroll
                for (int warp = 1; warp < num_warp_per_token; ++warp)
                    sum += smem_reduce[warp + token_group * num_warp_per_token];
                smem_reduce[N_COMPUTE_THREAD / 32 + token_group] = sum;
            }
            __sync_compute_group(N_COMPUTE_THREAD);
        }

        const float rms_rcp = need_norm
            ? rsqrtf(smem_reduce[N_COMPUTE_THREAD / 32 + token_group] / float(HEAD_DIM) + epsilon)
            : 1.0f;

        if (row_valid) {
            auto val = __bfloat1622float2(input(r, i));
            if (need_norm) {
                val.x *= rms_rcp;
                val.y *= rms_rcp;
                if (affine_weight != nullptr) {
                    auto weight = __bfloat1622float2(affine_weight[i]);
                    val.x *= weight.x;
                    val.y *= weight.y;
                }
            }
            if (need_rope && rope_row != nullptr) {
                auto cos_sin = __bfloat1622float2(rope_row[i]);
                const float rotated_even = val.x * cos_sin.x - val.y * cos_sin.y;
                const float rotated_odd = val.x * cos_sin.y + val.y * cos_sin.x;
                val = {rotated_even, rotated_odd};
            }
            input(r, i) = __float22bfloat162_rn(val);
        }
        __sync_compute_group(N_COMPUTE_THREAD);
    }
}

template <int HEAD_DIM, typename TensorVec, typename Vec2T>
__device__ __forceinline__ void rms_affine_rope_single_row(
    TensorVec& input,
    const int row_idx,
    float* smem_reduce,
    const float epsilon,
    const Vec2T* affine_weight,
    const Vec2T* rope_row,
    const bool need_norm,
    const bool need_rope
) {
    constexpr int vec_cols = HEAD_DIM / 2;
    const int thread_id = __compute_tid();
    const int lane_id = thread_id % 32;

    if (need_norm && thread_id < vec_cols) {
        auto val = __bfloat1622float2(input(row_idx, thread_id));
        float sum = val.x * val.x + val.y * val.y;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
        if (lane_id == 0)
            smem_reduce[thread_id / 32] = sum;
    }
    __sync_compute_group(128);

    if (need_norm && thread_id == 0)
        smem_reduce[0] += smem_reduce[1];
    __sync_compute_group(128);

    const float rms_rcp = need_norm ? rsqrtf(smem_reduce[0] / float(HEAD_DIM) + epsilon) : 1.0f;
    if (thread_id < vec_cols) {
        auto val = __bfloat1622float2(input(row_idx, thread_id));
        if (need_norm) {
            val.x *= rms_rcp;
            val.y *= rms_rcp;
            if (affine_weight != nullptr) {
                auto weight = __bfloat1622float2(affine_weight[thread_id]);
                val.x *= weight.x;
                val.y *= weight.y;
            }
        }
        if (need_rope && rope_row != nullptr) {
            auto cos_sin = __bfloat1622float2(rope_row[thread_id]);
            const float rotated_even = val.x * cos_sin.x - val.y * cos_sin.y;
            const float rotated_odd = val.x * cos_sin.y + val.y * cos_sin.x;
            val = {rotated_even, rotated_odd};
        }
        input(row_idx, thread_id) = __float22bfloat162_rn(val);
    }
    __sync_compute_group(128);
}


template <typename EngineS, typename LayoutS, typename TensorRowMax>
__device__ __forceinline__ void exp_scale(Tensor<EngineS, LayoutS> &acc_fragS, TensorRowMax& row_max) {
    // calculate P
    using accum_t = typename EngineS::value_type;

    #pragma unroll
    for (int r = 0; r < size<1>(acc_fragS); ++r) {
        // TODO(zijian): flashattention 3 says this:
        //  Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
        //  max * log_2(e)) This allows the compiler to use the ffma
        //  instruction instead of fadd and fmul separately.
        // 
        // currentlyy we just do exp(x - max)


        #pragma unroll
        for (int c = 0; c < size<0>(acc_fragS); ++c) {
            acc_fragS(c, r) = (accum_t)exp2f(acc_fragS(c, r) - row_max(r));
        }
    }
}

template<typename TensorS>
__device__ __forceinline__ void _mask(TensorS& acc_fragS, const int active_kv_len) {
    const int tid = threadIdx.x;
    const int ofst_in_group = tid % 4 * 2;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int i = 0; i < size<0>(acc_fragS); ++i) {
            int offset = i % 2 + ofst_in_group;
            int c = i / 2 * 8;
            if (c + offset >= active_kv_len) {
                acc_fragS(i, r) = -FLT_MAX;
            }
        }
    }
}

template <int tSRow, typename accum_t>
struct OnlineSoftmax {
    using TensorT = decltype(make_tensor<accum_t>(Shape<Int<tSRow>>{}));
    TensorT row_max, row_sum, scaler;
    __device__ __forceinline__ OnlineSoftmax() {
        clear(scaler);
        clear(row_sum);
        fill(row_max, -FLT_MAX);
    }

    template<typename TensorS>
    __device__ __forceinline__ void update2(TensorS& acc_fragS) {
        // cute::axpby(1.0f, scaler, 1.0f, row_sum);
        cute::transform(row_sum, scaler, row_sum, cute::multiplies{});
        cute::batch_reduce(acc_fragS, row_sum, cute::plus{});
    }

    template<typename TensorS, typename TensorO>
    __device__ __forceinline__ void update1(TensorS& acc_fragS, TensorO& acc_fragO) {
        // convert s to MN view before calling this function
        auto row_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, row_max_prev);

        // all reduce to get row max
        cute::batch_reduce(acc_fragS, row_max, cute::max_fn{});
        butterfly_reduce<4>(row_max, cute::max_fn{});
        
        // post correction for output fragments
        #pragma unroll
        for (int r = 0; r < tSRow; ++r) {
            accum_t score_scaler = (accum_t)exp2f(row_max_prev(r) - row_max(r));
            // row_sum(r) *= score_scaler;
            scaler(r) = score_scaler;

            #pragma unroll
            for (int c = 0; c < size<0>(acc_fragO); ++c) {
                acc_fragO(c, r) *= score_scaler;
            }
        }
        exp_scale(acc_fragS, row_max);
    }


    template<typename TensorO>
    __device__ __forceinline__ void post_correction(TensorO& acc_fragO) {
        #pragma unroll
        for (int r = 0; r < tSRow; ++r) {
            // TODO(zhiyuang): correct modification?
            float inv_sum = (row_sum(r) == 0) ? 1.f : 1.f / row_sum(r);
            #pragma unroll
            for (int c = 0; c < size<0>(acc_fragO); ++c) {
                acc_fragO(c, r) *= inv_sum;
            }
        }
    }
};

}

// SM100 decode attention keeps both the score and output accumulators in TMEM.
// Scores are drained to registers for online softmax, packed back into TMEM as
// BF16 probabilities, and consumed directly by the tensor-memory/shared-memory
// PV UMMA.  This mirrors the Blackwell FA4/FlashInfer dataflow while preserving
// VDCores' existing memory queues and shared-memory slot ABI.
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
template <int M, int N, typename TmemTensor, typename CoordTensor>
__device__ __forceinline__ void sm100_attention_rescale_tmem_rows(
    TmemTensor const& tmem_tensor,
    CoordTensor const& coord_tensor,
    const float scale,
    const int num_active_rows) {
    using namespace cute;
    const int tid = __compute_tid();
    const int active_warps = (num_active_rows + 15) / 16;
    if (tid < 2 * M && tid / 32 < active_warps) {
        using Load = SM100_TMEM_LOAD_16dp32b16x;
        using Store = SM100_TMEM_STORE_16dp32b16x;
        auto tiled_load = make_tmem_copy(Load{}, tmem_tensor);
        auto tiled_store = make_tmem_copy(Store{}, tmem_tensor);
        auto thr_load = tiled_load.get_slice(tid);
        auto thr_store = tiled_store.get_slice(tid);
        auto thread_tmem = thr_load.partition_S(tmem_tensor);
        auto thread_coord = thr_load.partition_D(coord_tensor);
        auto thread_tmem_store = thr_store.partition_D(tmem_tensor);
        auto r_values = make_tensor<float>(shape(thread_coord));
        copy(tiled_load, thread_tmem, r_values);
        #pragma unroll
        for (int i = 0; i < size(r_values); ++i) {
            r_values(i) *= scale;
        }
        copy(tiled_store, r_values, thread_tmem_store);
        cutlass::arch::fence_view_async_tmem_store();
    }
    __sync_compute_group(128);
}

template <int M, int N, typename TmemTensor, typename CoordTensor, typename SmemTensor>
__device__ __forceinline__ void sm100_attention_store_tmem_rows(
    TmemTensor const& tmem_tensor,
    CoordTensor const& coord_tensor,
    SmemTensor const& smem_tensor,
    const float inv_sum,
    const int num_active_rows) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    const int tid = __compute_tid();
    const int active_warps = (num_active_rows + 15) / 16;
    if (tid < 2 * M && tid / 32 < active_warps) {
        using Load = SM100_TMEM_LOAD_16dp32b16x;
        auto tiled_load = make_tmem_copy(Load{}, tmem_tensor);
        auto thr_load = tiled_load.get_slice(tid);
        auto thread_tmem = thr_load.partition_S(tmem_tensor);
        auto thread_coord = thr_load.partition_D(coord_tensor);
        auto thread_smem = thr_load.partition_D(smem_tensor);
        auto r_values = make_tensor<float>(shape(thread_coord));
        auto r_output = make_tensor<data_t>(shape(thread_smem));
        copy(tiled_load, thread_tmem, r_values);
        #pragma unroll
        for (int i = 0; i < size(r_values); ++i) {
            r_output(i) = data_t(r_values(i) * inv_sum);
        }
        copy(r_output, thread_smem);
    }
    __sync_compute_group(128);
    cutlass::arch::fence_view_async_shared();
}

template <int M, int N, typename TmemTensor, typename CoordTensor>
__device__ __forceinline__ void sm100_attention_store_tmem_rows_global(
    TmemTensor const& tmem_tensor,
    CoordTensor const& coord_tensor,
    cutlass::bfloat16_t *output,
    const float inv_sum,
    const int num_active_rows) {
    using namespace cute;
    const int tid = __compute_tid();
    const int active_warps = (num_active_rows + 15) / 16;
    if (tid < 2 * M && tid / 32 < active_warps) {
        using Load = SM100_TMEM_LOAD_16dp32b16x;
        using accum_t = float;
        using data_t = cutlass::bfloat16_t;
        using packed_t = cutlass::Array<data_t, 4>;
        auto tiled_load = make_tmem_copy(Load{}, tmem_tensor);
        auto thr_load = tiled_load.get_slice(tid);
        auto thread_tmem = thr_load.partition_S(tmem_tensor);
        auto thread_coord = thr_load.partition_D(coord_tensor);
        auto r_values = make_tensor<accum_t>(shape(thread_coord));
        copy(tiled_load, thread_tmem, r_values);
        static_assert(size(r_values) % 4 == 0,
                      "TMEM output fragments must form aligned BF16x4 packs");
        cutlass::NumericArrayConverter<data_t, accum_t, 4> convert;
        #pragma unroll
        for (int i = 0; i < size(r_values); i += 4) {
            const int row = int(get<0>(thread_coord(i + 0)));
            const int col = int(get<1>(thread_coord(i + 0)));
            if (row < num_active_rows) {
                cutlass::Array<accum_t, 4> values;
                values[0] = r_values(i + 0) * inv_sum;
                values[1] = r_values(i + 1) * inv_sum;
                values[2] = r_values(i + 2) * inv_sum;
                values[3] = r_values(i + 3) * inv_sum;
                *reinterpret_cast<packed_t *>(output + row * N + col) =
                    convert(values);
            }
        }
    }
}

template <int M, int KV, typename TmemTensor, typename CoordTensor,
          typename ProbTensor, typename ProbCoordTensor>
__device__ __forceinline__ void sm100_attention_softmax_tmem_rows(
    TmemTensor const& tmem_s,
    CoordTensor const& coord_s,
    ProbTensor const& tmem_p,
    ProbCoordTensor const& coord_p,
    const int block,
    const int num_kv_blocks,
    const int last_kv_active_token_len,
    const int num_active_q,
    int& logical_row,
    float& row_max,
    float& row_sum,
    float& correction) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr float kScoreScale = M_LOG2E / 11.313708498984761f;
    const int tid = __compute_tid();
    const int active_score_warps = (num_active_q + 15) / 16;
    correction = 0.0f;
    if (tid < 2 * M && tid / 32 < active_score_warps) {
        using ScoreLoad = SM100_TMEM_LOAD_16dp32b16x;
        auto tiled_s_load = make_tmem_copy(ScoreLoad{}, tmem_s);
        auto thr_s_load = tiled_s_load.get_slice(tid);
        auto thread_tmem_s = thr_s_load.partition_S(tmem_s);
        auto thread_coord_s = thr_s_load.partition_D(coord_s);
        const int row = int(get<0>(thread_coord_s(0)));
        logical_row = row;
        auto r_s = make_tensor<accum_t>(shape(thread_coord_s));
        copy(tiled_s_load, thread_tmem_s, r_s);

        float block_max = -FLT_MAX;
        #pragma unroll
        for (int i = 0; i < size(r_s); ++i) {
            const int col = int(get<1>(thread_coord_s(i)));
            float score = r_s(i) * kScoreScale;
            if (row >= num_active_q ||
                (block == num_kv_blocks - 1 && col >= last_kv_active_token_len)) {
                score = -FLT_MAX;
            }
            r_s(i) = score;
            block_max = fmaxf(block_max, score);
        }
        block_max = fmaxf(
            block_max,
            __shfl_xor_sync(0xFFFFFFFFU, block_max, 16));
        const float old_max = row_max;
        row_max = fmaxf(row_max, block_max);
        const bool row_valid = row < num_active_q && row_max != -FLT_MAX;
        correction = old_max == -FLT_MAX ? 0.0f : exp2f(old_max - row_max);
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < size(r_s); ++i) {
            const float probability = row_valid ? exp2f(r_s(i) - row_max) : 0.0f;
            r_s(i) = probability;
            block_sum += probability;
        }
        block_sum += __shfl_xor_sync(0xFFFFFFFFU, block_sum, 16);
        row_sum = row_sum * correction + block_sum;

        using ProbStore = SM100_TMEM_STORE_16dp32b16x;
        auto tiled_p_store = make_tmem_copy(ProbStore{}, tmem_p);
        auto thr_p_store = tiled_p_store.get_slice(tid);
        auto thread_coord_p = thr_p_store.partition_S(coord_p);
        auto thread_tmem_p = thr_p_store.partition_D(tmem_p);
        auto r_packed = make_tensor<uint32_t>(shape(thread_coord_p));
        auto r_packed_pairs = recast<cutlass::Array<data_t, 2>>(r_packed);
        cutlass::NumericArrayConverter<data_t, accum_t, 2> convert;
        const int lane_half = (tid >> 4) & 1;
        #pragma unroll
        for (int i = 0; i < size(r_packed_pairs); ++i) {
            const int chunk = i / 16;
            const int pair_in_quarter = i & 7;
            const int local_idx = chunk * 32 + 16 * lane_half + 2 * pair_in_quarter;
            const int send_idx = chunk * 32 + 16 * (1 - lane_half) + 2 * pair_in_quarter;
            const bool use_partner = ((i >> 3) & 1) != lane_half;
            const accum_t partner_0 = __shfl_xor_sync(
                0xFFFFFFFFU, r_s(send_idx), 16);
            const accum_t partner_1 = __shfl_xor_sync(
                0xFFFFFFFFU, r_s(send_idx + 1), 16);
            cutlass::Array<accum_t, 2> pair;
            pair[0] = use_partner ? partner_0 : r_s(local_idx);
            pair[1] = use_partner ? partner_1 : r_s(local_idx + 1);
            r_packed_pairs(i) = convert(pair);
        }
        copy(tiled_p_store, r_packed, thread_tmem_p);
        cutlass::arch::fence_view_async_tmem_store();
    }
    __sync_compute_group(128);
}

template <int M, int KV, typename TmemTensor, typename CoordTensor,
          typename ProbTensor>
__device__ __forceinline__ void sm100_attention_softmax_tmem_smem_fa4(
    TmemTensor const& tmem_s,
    CoordTensor const& coord_s,
    ProbTensor const& smem_p,
    float *score_stage,
    float *smem_reduce,
    const int block,
    const int num_kv_blocks,
    const int last_kv_active_token_len,
    const int num_active_q,
    int& logical_row,
    float& row_max,
    float& row_sum,
    float& correction) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr float kScoreScale = M_LOG2E / 11.313708498984761f;
    constexpr int kValuesPerLane = KV / 32;
    static_assert(KV == 64 || KV == 128);

    const int tid = __compute_tid();
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // One warp drains the QK accumulator.  The 16-DP TMEM mapping gives two
    // lanes to each row; stage only the live GQA rows in a compact linear
    // buffer so all four compute warps can participate in softmax.
    if (tid < 32) {
        using ScoreLoad = SM100_TMEM_LOAD_16dp32b16x;
        auto tiled_load = make_tmem_copy(ScoreLoad{}, tmem_s);
        auto thr_load = tiled_load.get_slice(lane_id);
        auto thread_tmem = thr_load.partition_S(tmem_s);
        auto thread_coord = thr_load.partition_D(coord_s);
        auto r_s = make_tensor<accum_t>(shape(thread_coord));
        copy(tiled_load, thread_tmem, r_s);
        #pragma unroll
        for (int i = 0; i < size(r_s); ++i) {
            const int row = int(get<0>(thread_coord(i)));
            const int col = int(get<1>(thread_coord(i)));
            if (row < num_active_q) {
                score_stage[row * KV + col] =
                    (block + 1 < num_kv_blocks ||
                     col < last_kv_active_token_len)
                    ? r_s(i) * kScoreScale : -FLT_MAX;
            }
        }
    }
    __sync_compute_group(128);

    accum_t scores[kValuesPerLane];
    accum_t block_max = -FLT_MAX;
    if (warp_id < num_active_q) {
        #pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
            scores[i] = score_stage[
                warp_id * KV + lane_id * kValuesPerLane + i];
            block_max = fmaxf(block_max, scores[i]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_max = fmaxf(
                block_max,
                __shfl_xor_sync(0xFFFFFFFFU, block_max, offset));
        }
    }

    accum_t block_sum = 0.0f;
    if (warp_id < num_active_q) {
        const accum_t old_max = block == 0
            ? -FLT_MAX : smem_reduce[warp_id];
        const accum_t old_sum = block == 0
            ? 0.0f : smem_reduce[4 + warp_id];
        const accum_t new_max = fmaxf(old_max, block_max);
        const accum_t row_correction = old_max == -FLT_MAX
            ? 0.0f : exp2f(old_max - new_max);
        #pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
            scores[i] = exp2f(scores[i] - new_max);
            block_sum += scores[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_xor_sync(
                0xFFFFFFFFU, block_sum, offset);
        }
        if (lane_id == 0) {
            smem_reduce[warp_id] = new_max;
            smem_reduce[4 + warp_id] =
                old_sum * row_correction + block_sum;
            smem_reduce[8 + warp_id] = row_correction;
        }
    }

    // Every score has been consumed into registers; the same special-slot
    // storage can now hold the swizzled BF16 probability tile for SS PV.
    __sync_compute_group(128);
    if (warp_id < num_active_q) {
        #pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
            const int col = lane_id * kValuesPerLane + i;
            // layout_p is hierarchically tiled; its flattened logical index
            // is M-major even though the physical shared layout is swizzled.
            smem_p(warp_id + M * col) = data_t(scores[i]);
        }
    }
    __sync_compute_group(128);
    cutlass::arch::fence_view_async_shared();

    // Restore the 16-DP row ownership expected by the output epilogue and
    // split-LSE publication.  Only warp 0 consumes these values afterwards.
    if (tid < 32) {
        logical_row = lane_id & 15;
        if (logical_row < num_active_q) {
            row_max = smem_reduce[logical_row];
            row_sum = smem_reduce[4 + logical_row];
            correction = smem_reduce[8 + logical_row];
        }
    } else {
        logical_row = num_active_q;
    }
}

template <int HEAD_DIM, int KV_BLOCK_SIZE = 64,
          bool SPLIT_KV = false, int MAX_SPLIT = 16,
          bool DIRECT_OUTPUT = false,
          bool FA4_SMEM_P = false,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_attention_fwd_sm100_decode(
    const int num_kv_blocks,
    const int split_idx,
    const int num_active_q,
    const int last_kv_active_token_len,
    const bool runtime_need_norm,
    const bool runtime_need_rope,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    void *base,
    float *smem_reduce,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    using vec2_t = typename F16Traits<data_t>::vec2_t;

    constexpr int M = 64;
    constexpr int KV = KV_BLOCK_SIZE;
    constexpr int O_N = HEAD_DIM;
    constexpr uint32_t P_OFFSET = 0;
    constexpr uint32_t O_OFFSET = 128;
    static_assert(HEAD_DIM == 128, "The SM100 decode path is specialized for Llama head_dim=128");
    static_assert(KV == 64 || KV == 128, "The SM100 decode path supports 64- or 128-token KV tiles");

    using QKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, KV, UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = std::conditional_t<
        FA4_SMEM_P,
        SM100_MMA_F16BF16_SS<
            data_t, data_t, accum_t, M, O_N,
            UMMA::Major::K, UMMA::Major::MN>,
        SM100_MMA_F16BF16_TS<
            data_t, data_t, accum_t, M, O_N,
            UMMA::Major::K, UMMA::Major::MN>>;

    const int tid = __compute_tid();
    auto tiled_qk = make_tiled_mma(QKAtom{});
    auto cta_qk = tiled_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto cta_pv = tiled_pv.get_slice(0);

    auto q_shape = partition_shape_A(tiled_qk, make_shape(Int<M>{}, Int<HEAD_DIM>{}));
    auto k_shape = partition_shape_B(tiled_qk, make_shape(Int<KV>{}, Int<HEAD_DIM>{}));
    auto v_shape = partition_shape_B(tiled_pv, make_shape(Int<O_N>{}, Int<KV>{}));
    auto p_shape = partition_shape_A(tiled_pv, make_shape(Int<M>{}, Int<KV>{}));
    auto layout_q = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<data_t>{}, q_shape);
    auto layout_k = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<data_t>{}, k_shape);
    auto layout_v = UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<data_t>{}, v_shape);
    auto layout_p = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<data_t>{}, p_shape);

    auto logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<KV>{}), make_stride(Int<KV>{}, Int<1>{})));
    auto coord_s = make_identity_tensor(make_shape(Int<M>{}, Int<KV>{}));
    auto cta_s = cta_qk.partition_C(logical_s);
    auto cta_coord_s = cta_qk.partition_C(coord_s);
    auto tmem_s = cta_qk.make_fragment_C(cta_s);
    tmem_s.data() = tmem_base_ptr;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<O_N>{}), make_stride(Int<O_N>{}, Int<1>{})));
    auto coord_o = make_identity_tensor(make_shape(Int<M>{}, Int<O_N>{}));
    auto cta_o = cta_pv.partition_C(logical_o);
    auto cta_coord_o = cta_pv.partition_C(coord_o);
    auto tmem_o = cta_pv.make_fragment_C(cta_o);
    tmem_o.data() = tmem_base_ptr + O_OFFSET;

    // The PV A operand aliases the QK accumulator columns.  A packed 16-bit
    // TMEM store writes two BF16 probabilities into each 32-bit score column.
    auto tmem_p_view = tmem_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));
    tmem_p_view.data() = tmem_base_ptr + P_OFFSET;
    auto coord_p_view = cta_coord_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));

    const bool use_fused_qk = runtime_need_norm || runtime_need_rope;
    int side_slot = 0;
    int k_store_slot = 0;
    const vec2_t *q_norm_weight = nullptr;
    const vec2_t *k_norm_weight = nullptr;
    const vec2_t *rope_row = nullptr;
    vec2_t *k_store_ptr = nullptr;
    if (use_fused_qk) {
        side_slot = m2c.template pop<0>();
        const vec2_t *packed = static_cast<const vec2_t *>(
            get_slot_address(base, extract(side_slot)));
        q_norm_weight = packed;
        k_norm_weight = packed + HEAD_DIM / 2;
        rope_row = packed + HEAD_DIM;
        k_store_slot = m2c.template pop<0>();
        k_store_ptr = static_cast<vec2_t *>(get_slot_address(base, extract(k_store_slot)));
    }

    const int q_slot = m2c.template pop<0>();
    data_t *q_ptr = static_cast<data_t *>(get_slot_address(base, extract(q_slot)));
    auto sQ = make_tensor(make_smem_ptr(q_ptr), layout_q);
    auto layout_q_vec = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<vec2_t>{},
        make_shape(Int<M>{}, Int<HEAD_DIM / 2>{}));
    auto sQ_vec = make_tensor(make_smem_ptr(reinterpret_cast<vec2_t *>(q_ptr)), layout_q_vec);
    if (use_fused_qk) {
        rms_affine_rope_rows<HEAD_DIM, 128>(
            sQ_vec, num_active_q, smem_reduce, 1.0e-6f,
            q_norm_weight, rope_row, runtime_need_norm, runtime_need_rope);
        // The two 64-thread row groups finish independently.  Join them before
        // UMMA can consume the full Q tile, then publish generic shared stores
        // to the asynchronous tensor-core proxy.
        __sync_compute_group(128);
        cutlass::arch::fence_view_async_shared();
    }
    auto frag_q = cta_qk.make_fragment_A(sQ);

    int logical_row = tid;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    int k_slot = m2c.template pop<0>();
    int o_slot = 0;
    for (int block = 0; block < num_kv_blocks; ++block) {
        data_t *k_ptr = static_cast<data_t *>(get_slot_address(base, extract(k_slot)));
        auto sK = make_tensor(make_smem_ptr(k_ptr), layout_k);
        auto frag_k = cta_qk.make_fragment_B(sK);

        if (use_fused_qk && block == num_kv_blocks - 1 && last_kv_active_token_len > 0) {
            auto layout_k_vec = tile_to_shape(
                GMMA::Layout_K_SW128_Atom<vec2_t>{},
                make_shape(Int<KV>{}, Int<HEAD_DIM / 2>{}));
            auto sK_vec = make_tensor(make_smem_ptr(reinterpret_cast<vec2_t *>(k_ptr)), layout_k_vec);
            const int row = last_kv_active_token_len - 1;
            rms_affine_rope_single_row<HEAD_DIM>(
                sK_vec, row, smem_reduce, 1.0e-6f,
                k_norm_weight, rope_row, runtime_need_norm, runtime_need_rope);
            cutlass::arch::fence_view_async_shared();
            if (tid < HEAD_DIM / 2) {
                k_store_ptr[tid] = sK_vec(row, tid);
            }
            __sync_compute_group(128);
        }

        tiled_qk.accumulate_ = UMMA::ScaleOut::Zero;
        if (tid < numThreadsPerWarp) {
            #pragma unroll
            for (int k_block = 0; k_block < size<2>(frag_q); ++k_block) {
                gemm(tiled_qk, frag_q(_, _, k_block), frag_k(_, _, k_block), tmem_s);
                tiled_qk.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }

        // K and V are consecutive in the memory queue. Consume V while QK
        // executes so the queue synchronization and any remaining TMA latency
        // are hidden behind tcgen05 rather than paid after softmax.
        const int v_slot = m2c.template pop<0>();
        data_t *v_ptr = static_cast<data_t *>(get_slot_address(base, extract(v_slot)));
        auto sV = make_tensor(make_smem_ptr(v_ptr), layout_v);
        auto frag_v = cta_pv.make_fragment_B(sV);
        if constexpr (FA4_SMEM_P) {
            if (num_kv_blocks == 1) {
                o_slot = m2c.template pop<0>();
            }
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;

        float correction = 0.0f;
        if constexpr (FA4_SMEM_P) {
            data_t *p_ptr = static_cast<data_t *>(
                get_slot_address(base, numSlots));
            auto sP = make_tensor(make_smem_ptr(p_ptr), layout_p);
            sm100_attention_softmax_tmem_smem_fa4<M, KV>(
                tmem_s, cta_coord_s, sP,
                reinterpret_cast<float *>(p_ptr), smem_reduce,
                block, num_kv_blocks,
                last_kv_active_token_len, num_active_q,
                logical_row, row_max, row_sum, correction);
        } else {
            auto dummy_p = make_tensor(
                make_smem_ptr(static_cast<data_t *>(nullptr)), layout_p);
            auto tmem_p = cta_pv.make_fragment_A(dummy_p);
            tmem_p.data() = tmem_base_ptr + P_OFFSET;
            sm100_attention_softmax_tmem_rows<M, KV>(
                tmem_s, cta_coord_s, tmem_p_view, coord_p_view,
                block, num_kv_blocks, last_kv_active_token_len, num_active_q,
                logical_row, row_max, row_sum, correction);
        }

        if (block > 0) {
            sm100_attention_rescale_tmem_rows<M, O_N>(
                tmem_o, cta_coord_o, correction, num_active_q);
        }

        c2m.push(tid, k_slot);

        tiled_pv.accumulate_ = block == 0 ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
        if (tid < numThreadsPerWarp) {
            if constexpr (FA4_SMEM_P) {
                data_t *p_ptr = static_cast<data_t *>(
                    get_slot_address(base, numSlots));
                auto sP = make_tensor(make_smem_ptr(p_ptr), layout_p);
                auto frag_p = cta_pv.make_fragment_A(sP);
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(frag_p); ++k_block) {
                    gemm(tiled_pv, frag_p(_, _, k_block),
                         frag_v(_, _, k_block), tmem_o);
                    tiled_pv.accumulate_ = UMMA::ScaleOut::One;
                }
            } else {
                auto dummy_p = make_tensor(
                    make_smem_ptr(static_cast<data_t *>(nullptr)), layout_p);
                auto tmem_p = cta_pv.make_fragment_A(dummy_p);
                tmem_p.data() = tmem_base_ptr + P_OFFSET;
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(tmem_p); ++k_block) {
                    gemm(tiled_pv, tmem_p(_, _, k_block),
                         frag_v(_, _, k_block), tmem_o);
                    tiled_pv.accumulate_ = UMMA::ScaleOut::One;
                }
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }

        // Likewise, pre-consume the next K tile while PV is in flight. The
        // single TMEM completion barrier remains ordered QK/PV/QK/PV.
        int next_k_slot = 0;
        if (block + 1 < num_kv_blocks) {
            next_k_slot = m2c.template pop<0>();
        } else if constexpr (!FA4_SMEM_P) {
            // The output allocation follows the final V tile in the memory
            // stream. Hide that queue transaction under the final PV UMMA.
            o_slot = m2c.template pop<0>();
        } else if (num_kv_blocks > 1) {
            o_slot = m2c.template pop<0>();
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;

        // Issue the direct epilogue before returning the input slots.  The
        // raw global stores can drain while the queue bookkeeping for V/Q is
        // performed, instead of sitting at the very end of the task.
        if constexpr (DIRECT_OUTPUT) {
            if (block + 1 == num_kv_blocks) {
                const float inv_sum =
                    tid < 2 * M && logical_row < num_active_q && row_sum > 0.0f
                    ? 1.0f / row_sum : 0.0f;
                data_t *o_ptr = static_cast<data_t *>(
                    slot_2_glob_ptr(st_insts, o_slot));
                sm100_attention_store_tmem_rows_global<M, O_N>(
                    tmem_o, cta_coord_o, o_ptr, inv_sum, num_active_q);
                // Raw-address allocations carry the special-slot index on
                // M2C, but C2M/ST consumes a slot mask.  Preserve the
                // production output barrier by publishing the corresponding
                // one-hot mask (the standalone harness has no output bar and
                // did not expose this distinction).
                c2m.template push<31, true, false>(
                    tid, 1U << o_slot);
            }
        }
        c2m.push(tid, v_slot);
        k_slot = next_k_slot;
    }

    c2m.push(tid, q_slot);
    if (use_fused_qk) {
        c2m.push(tid, side_slot);
        c2m.template push<0, true>(tid, k_store_slot);
    }

    if constexpr (!DIRECT_OUTPUT) {
        const float inv_sum =
            tid < 2 * M && logical_row < num_active_q && row_sum > 0.0f
            ? 1.0f / row_sum : 0.0f;
        data_t *o_ptr = static_cast<data_t *>(get_slot_address(base, extract(o_slot)));
        auto sO = make_tensor(
            make_smem_ptr(o_ptr),
            make_layout(make_shape(Int<M>{}, Int<O_N>{}), make_stride(Int<O_N>{}, Int<1>{})));
        auto cta_sO = cta_pv.partition_C(sO);
        sm100_attention_store_tmem_rows<M, O_N>(
            tmem_o, cta_coord_o, cta_sO, inv_sum, num_active_q);
        c2m.template push<0, true>(tid, o_slot);
    }

    if constexpr (SPLIT_KV) {
        // The score accumulator is kept in the log2 domain, matching the
        // exp2-based split reducer.  A 16-DP TMEM load assigns two threads to
        // each row; both publish the same scalar, so duplicate stores are
        // benign and avoid another synchronization/shuffle.
        const int lse_slot = m2c.template pop<0>();
        accum_t *lse_ptr = static_cast<accum_t *>(
            slot_2_glob_ptr(st_insts, lse_slot));
        if (tid < 2 * M && logical_row < num_active_q && row_sum > 0.0f) {
            lse_ptr[logical_row * MAX_SPLIT + split_idx] =
                row_max + log2f(row_sum);
        }
        __sync_compute_group(128);
        c2m.template push<31, true, false>(tid, 1U << lse_slot);
    }
}

// Low-latency GQA decode specialization for Blackwell.  Instead of padding the
// four local query heads to the M=64 dimension, transpose both GEMMs so the
// sequence/head dimension occupies M=128 and the query-head dimension occupies
// N=8.  This is the same swapped-A/B formulation used by CUTLASS' SM100 TGV GQA
// example: S = K * Q and O = V * P.  A 32-DP TMEM load gives every compute
// thread one sequence/output row, allowing all four compute warps to perform
// softmax and the TMEM correction in parallel.
template <bool SPLIT_KV = false, int MAX_SPLIT = 16,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_attention_fwd_sm100_decode_swap(
    const int num_kv_blocks,
    const int split_idx,
    const int num_active_q,
    const int last_kv_active_token_len,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    void *base,
    float *smem_reduce,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;

    constexpr int D = 128;
    constexpr int KV = 128;
    constexpr int Q = 8;
    constexpr int ACTIVE_Q = 4;
    constexpr uint32_t O_OFFSET = 32;
    constexpr int kTmemColumnBits = Q * sizeof_bits_v<accum_t>;
    static_assert(kTmemColumnBits == 256);

    using QKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, KV, Q,
        UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, D, Q,
        UMMA::Major::MN, UMMA::Major::K>;

    const int tid = __compute_tid();
    const int warp_id = tid / numThreadsPerWarp;
    const int lane_id = tid % numThreadsPerWarp;
    auto tiled_qk = make_tiled_mma(QKAtom{});
    auto cta_qk = tiled_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto cta_pv = tiled_pv.get_slice(0);

    auto k_shape = partition_shape_A(
        tiled_qk, make_shape(Int<KV>{}, Int<D>{}));
    auto q_shape = partition_shape_B(
        tiled_qk, make_shape(Int<Q>{}, Int<D>{}));
    auto v_shape = partition_shape_A(
        tiled_pv, make_shape(Int<D>{}, Int<KV>{}));
    auto layout_k = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, k_shape);
    auto layout_q = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, q_shape);
    auto layout_v = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, v_shape);

    auto shape_s = make_shape(Int<KV>{}, Int<Q>{});
    auto shape_p = make_shape(Int<Q>{}, Int<KV>{});
    auto layout_s = tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, shape_s,
        Step<_1, _2>{});
    auto layout_p = tile_to_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, shape_p,
        Step<_2, _1>{});

    // S and P deliberately alias the runtime's special scratch slot.  The two
    // layouts are transposed views with identical physical swizzle stacking.
    data_t *prob_ptr = static_cast<data_t *>(
        get_slot_address(base, numSlots));
    auto sS = make_tensor(make_smem_ptr(prob_ptr), layout_s);
    auto sP = make_tensor(make_smem_ptr(prob_ptr), layout_p);
    auto cta_s = cta_qk.partition_C(sS);
    auto cta_p = cta_pv.partition_B(sP);
    auto frag_p = cta_pv.make_fragment_B(cta_p);
    auto tmem_s = cta_qk.make_fragment_C(cta_s);
    tmem_s.data() = tmem_base_ptr;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<D>{}, Int<Q>{})));
    auto coord_o = make_identity_tensor(make_shape(Int<D>{}, Int<Q>{}));
    auto cta_o = cta_pv.partition_C(logical_o);
    auto cta_coord_o = cta_pv.partition_C(coord_o);
    auto tmem_o = cta_pv.make_fragment_C(cta_o);
    tmem_o.data() = tmem_base_ptr + O_OFFSET;

    auto score_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x,
                          kTmemColumnBits>();
    auto score_t2r = make_tmem_copy(score_load_op, tmem_s);
    auto score_thr = score_t2r.get_slice(tid);
    auto thread_tmem_s = score_thr.partition_S(tmem_s);
    auto thread_s = score_thr.partition_D(cta_s);
    auto r_scores = make_tensor<accum_t>(shape(thread_s));
    auto r_probs = make_tensor<data_t>(shape(thread_s));
    static_assert(size(r_scores) == Q,
                  "each 32-DP thread must own all eight query columns");

    auto output_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x,
                          kTmemColumnBits>();
    auto output_store_op =
        TMEM::op_repeater<SM100_TMEM_STORE_32dp32b1x,
                          kTmemColumnBits>();
    auto output_t2r = make_tmem_copy(output_load_op, tmem_o);
    auto output_r2t = make_tmem_copy(output_store_op, tmem_o);
    auto output_load_thr = output_t2r.get_slice(tid);
    auto output_store_thr = output_r2t.get_slice(tid);
    auto thread_tmem_o = output_load_thr.partition_S(tmem_o);
    auto thread_coord_o = output_load_thr.partition_D(cta_coord_o);
    auto thread_tmem_o_store = output_store_thr.partition_D(tmem_o);
    auto r_output = make_tensor<accum_t>(shape(thread_coord_o));
    static_assert(size(r_output) == Q,
                  "each 32-DP thread must own all eight output columns");

    const int q_slot = m2c.template pop<0>();
    data_t *q_ptr = static_cast<data_t *>(
        get_slot_address(base, extract(q_slot)));
    auto sQ = make_tensor(make_smem_ptr(q_ptr), layout_q);
    auto frag_q = cta_qk.make_fragment_B(sQ);

    accum_t row_max[ACTIVE_Q];
    accum_t row_sum[ACTIVE_Q];
    #pragma unroll
    for (int q = 0; q < ACTIVE_Q; ++q) {
        row_max[q] = -FLT_MAX;
        row_sum[q] = 0.0f;
    }

    int k_slot = m2c.template pop<0>();
    int o_slot = 0;
    constexpr accum_t kScoreScale = M_LOG2E / 11.313708498984761f;
    for (int block = 0; block < num_kv_blocks; ++block) {
        data_t *k_ptr = static_cast<data_t *>(
            get_slot_address(base, extract(k_slot)));
        auto sK = make_tensor(make_smem_ptr(k_ptr), layout_k);
        auto frag_k = cta_qk.make_fragment_A(sK);

        tiled_qk.accumulate_ = UMMA::ScaleOut::Zero;
        if (tid < numThreadsPerWarp) {
            #pragma unroll
            for (int k_block = 0;
                 k_block < size<2>(frag_k); ++k_block) {
                gemm(tiled_qk, frag_k(_, _, k_block),
                     frag_q(_, _, k_block), tmem_s);
                tiled_qk.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }

        // Consume V and the following K/output descriptor while QK executes.
        const int v_slot = m2c.template pop<0>();
        data_t *v_ptr = static_cast<data_t *>(
            get_slot_address(base, extract(v_slot)));
        auto sV = make_tensor(make_smem_ptr(v_ptr), layout_v);
        auto frag_v = cta_pv.make_fragment_A(sV);
        int next_k_slot = 0;
        if (block + 1 < num_kv_blocks) {
            next_k_slot = m2c.template pop<0>();
        } else {
            o_slot = m2c.template pop<0>();
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;

        copy(score_t2r, thread_tmem_s, r_scores);
        cutlass::arch::fence_view_async_tmem_load();
        __sync_compute_group(128);

        accum_t correction[ACTIVE_Q];
        const bool token_valid =
            block + 1 < num_kv_blocks || tid < last_kv_active_token_len;
        #pragma unroll
        for (int q = 0; q < ACTIVE_Q; ++q) {
            accum_t score = token_valid && q < num_active_q
                ? r_scores(q) * kScoreScale : -FLT_MAX;
            r_scores(q) = score;
            accum_t warp_max;
            asm volatile(
                "redux.sync.max.NaN.f32 %0, %1, 0xffffffff;\n"
                : "=f"(warp_max) : "f"(score));
            if (lane_id == 0) {
                smem_reduce[warp_id * ACTIVE_Q + q] = warp_max;
            }
        }
        __sync_compute_group(128);

        #pragma unroll
        for (int q = 0; q < ACTIVE_Q; ++q) {
            accum_t value = smem_reduce[q];
            #pragma unroll
            for (int warp = 1; warp < 4; ++warp) {
                value = fmaxf(
                    value, smem_reduce[warp * ACTIVE_Q + q]);
            }
            const accum_t new_max = fmaxf(row_max[q], value);
            correction[q] = row_max[q] == -FLT_MAX
                ? 0.0f : exp2f(row_max[q] - new_max);
            row_max[q] = new_max;
            r_scores(q) = q < num_active_q
                ? exp2f(r_scores(q) - new_max) : 0.0f;
        }

        #pragma unroll
        for (int q = 0; q < ACTIVE_Q; ++q) {
            accum_t warp_sum = r_scores(q);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_xor_sync(
                    0xFFFFFFFFU, warp_sum, offset);
            }
            if (lane_id == 0) {
                smem_reduce[warp_id * ACTIVE_Q + q] = warp_sum;
            }
        }
        __sync_compute_group(128);

        #pragma unroll
        for (int q = 0; q < ACTIVE_Q; ++q) {
            accum_t block_sum = smem_reduce[q];
            #pragma unroll
            for (int warp = 1; warp < 4; ++warp) {
                block_sum += smem_reduce[warp * ACTIVE_Q + q];
            }
            row_sum[q] = row_sum[q] * correction[q] + block_sum;
        }
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            r_probs(q) = data_t(r_scores(q));
            if (q >= ACTIVE_Q) {
                r_probs(q) = data_t(0.0f);
            }
        }

        // Store P through the S view.  The transposed P view aliases the same
        // bytes and is immediately consumable as the B operand of PV.
        copy(r_probs, thread_s);
        cutlass::arch::fence_view_async_shared();
        __sync_compute_group(128);

        if (block > 0) {
            copy(output_t2r, thread_tmem_o, r_output);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for (int q = 0; q < ACTIVE_Q; ++q) {
                r_output(q) *= correction[q];
            }
            copy(output_r2t, r_output, thread_tmem_o_store);
            cutlass::arch::fence_view_async_tmem_store();
            __sync_compute_group(128);
        }

        c2m.push(tid, k_slot);

        tiled_pv.accumulate_ = block == 0
            ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
        if (tid < numThreadsPerWarp) {
            #pragma unroll
            for (int k_block = 0; k_block < size<2>(frag_v); ++k_block) {
                gemm(tiled_pv, frag_v(_, _, k_block),
                     frag_p(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }

        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;

        if (block + 1 == num_kv_blocks) {
            copy(output_t2r, thread_tmem_o, r_output);
            cutlass::arch::fence_view_async_tmem_load();
            data_t *o_ptr = static_cast<data_t *>(
                slot_2_glob_ptr(st_insts, o_slot));
            #pragma unroll
            for (int i = 0; i < size(r_output); ++i) {
                const int row = int(get<0>(thread_coord_o(i)));
                const int q = int(get<1>(thread_coord_o(i)));
                if (q < num_active_q) {
                    accum_t output_scale = 1.0f;
                    if constexpr (!SPLIT_KV) {
                        output_scale = row_sum[q] > 0.0f
                            ? 1.0f / row_sum[q] : 0.0f;
                    }
                    o_ptr[q * D + row] =
                        data_t(r_output(i) * output_scale);
                }
            }
            c2m.template push<31, true, false>(tid, 1U << o_slot);
        }
        c2m.push(tid, v_slot);
        k_slot = next_k_slot;
    }

    c2m.push(tid, q_slot);

    if constexpr (SPLIT_KV) {
        const int lse_slot = m2c.template pop<0>();
        accum_t *lse_ptr = static_cast<accum_t *>(
            slot_2_glob_ptr(st_insts, lse_slot));
        if (tid < num_active_q && row_sum[tid] > 0.0f) {
            lse_ptr[(tid * 2 + 0) * MAX_SPLIT + split_idx] =
                row_max[tid];
            lse_ptr[(tid * 2 + 1) * MAX_SPLIT + split_idx] =
                row_sum[tid];
        }
        __sync_compute_group(128);
        c2m.template push<31, true, false>(tid, 1U << lse_slot);
    }
}

#endif

template <int HEAD_DIM,
          int Q_BLOCK_SIZE,
          int KV_BLOCK_SIZE,
          bool SPLIT_KV,
          int MAX_SPLIT,
          bool NEED_NORM, bool NEED_ROPE,
          typename AtomQK, typename AtomPV, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_attention_fwd_flash3_grouped(
    const int num_kv_blocks,
    const int split_idx,
    const int num_active_q, // to avoid overwriting other split_kv metadata buffer
    const int last_kv_active_token_len, // real kv tokens in the last block
    const int kv_start_idx, // global token-pos of first kv token, for prefill mask calculation
    const bool runtime_need_norm,
    const bool runtime_need_rope,
    void *base,
    float *smem_reduce,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    // Q: [HEAD_GROUP_SIZE, HEAD_DIM]
    // K, V: [SEQ_LEN, HEAD_DIM]

    using namespace cute;
    using AtomTrait = MMA_Traits<AtomQK>;
    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    using AtomTrait_PV = MMA_Traits<AtomPV>;
    using data_t_PV = typename AtomTrait_PV::ValTypeA;
    using accum_t_PV = typename AtomTrait_PV::ValTypeC;
    using Tr = F16Traits<data_t>;
    using vec2_t = typename Tr::vec2_t;

    static_assert(std::is_same<accum_t, accum_t_PV>::value, "accum type of QK and PV atom should be the same");

    constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});

    assert(blockDim.x >= 128 && "At least 128 threads are required for wgmma_m64n256k16");

    const int thread_id = threadIdx.x;

    auto tiled_mma_qk = make_tiled_mma(
        MMA_Atom<AtomQK>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<Q_BLOCK_SIZE>{}, Int<KV_BLOCK_SIZE>{}, Int<HEAD_DIM>{}) // tile along the M, N dims
    );
    auto tiled_mma_pv = make_tiled_mma(
        MMA_Atom<AtomPV>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<Q_BLOCK_SIZE>{}, Int<HEAD_DIM>{}, Int<KV_BLOCK_SIZE>{}) // tile along the M, N dims
    );
    // INTER (no-swizzle)
    // layout_sQ: Sw<0,4,3> o smem_ptr[16b](unset) o ((_8,_8),(_8,_16)):((_8,_64),(_1,_512))

    // SW32
    // layout_sQ: Sw<1,4,3> o smem_ptr[16b](unset) o ((_8,_8),(_16,_8)):((_16,_128),(_1,_1024))

    // SW64
    // layout_sQ: Sw<2,4,3> o smem_ptr[16b](unset) o ((_8,_8),(_32,_4)):((_32,_256),(_1,_2048))

    // SW128
    // layout_sQ: Sw<3,4,3> o smem_ptr[16b](unset) o ((_8,_8),(_64,_2)):((_64,_512),(_1,_4096))

    auto layout_sQ = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<HEAD_DIM>{}));
    auto layout_sK = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<KV_BLOCK_SIZE>{},Int<HEAD_DIM>{}));
    auto layout_sV = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<HEAD_DIM>{},Int<KV_BLOCK_SIZE>{}));
    auto layout_sP = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<KV_BLOCK_SIZE>{}));

    // TODO(zhiyuang): this is a non-swizzled layout for partial offloading.
    // try to add a flag to this one!

    // auto layout_sO = tile_to_shape(
    //     GMMA::Layout_K_SW128_Atom<data_t>{},
    //     make_shape(Int<Q_BLOCK_SIZE>{},Int<HEAD_DIM>{}));
    auto layout_sO = make_layout(
        make_shape(Int<Q_BLOCK_SIZE>{},Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, Int<1>{})
    );
    
    auto layout_sQ_vec = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<vec2_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<HEAD_DIM/2>{}));
    auto layout_sK_vec = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<vec2_t>{},
        make_shape(Int<KV_BLOCK_SIZE>{},Int<HEAD_DIM/2>{}));
    
// For Debug
    auto layout_sR = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<KV_BLOCK_SIZE>{}));
    auto layout_sPR = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<KV_BLOCK_SIZE>{}));
    auto layout_sPV = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{},Int<HEAD_DIM>{}));

    auto thr_mma_qk = tiled_mma_qk.get_slice(threadIdx.x);
    auto thr_mma_pv = tiled_mma_pv.get_slice(threadIdx.x);

    // S layout
    Tensor t_dummyS = make_tensor(
        make_smem_ptr((accum_t*)base),
        make_shape(Int<Q_BLOCK_SIZE>{}, Int<KV_BLOCK_SIZE>{})
    );

    // for each KV block, do
    // load K
    // 1. S = QK^T
    // 2. m_old = m, m = max(m_old, row_max(S))
    // 3. P = exp(S - m), l = exp(m_old - m) * l + row_sum(P)
    // load V
    // 4. O = diag(exp(m_old - m)) * O + PV

    const bool use_qwen_fused_qk = runtime_need_norm || runtime_need_rope;
    int slot_side_input = 0;
    int slot_k_store = 0;
    const vec2_t* q_norm_weight = nullptr;
    const vec2_t* k_norm_weight = nullptr;
    const vec2_t* rope_row = nullptr;
    vec2_t* sKStore_ptr = nullptr;
    if (use_qwen_fused_qk) {
        slot_side_input = m2c.template pop<0>();
        const vec2_t* packed_side_input = (const vec2_t*)get_slot_address(base, extract(slot_side_input));
        q_norm_weight = packed_side_input;
        k_norm_weight = packed_side_input + HEAD_DIM / 2;
        rope_row = packed_side_input + HEAD_DIM;
        slot_k_store = m2c.template pop<0>();
        sKStore_ptr = (vec2_t*)get_slot_address(base, extract(slot_k_store));
    }

    // load Qtile
    int slot_Q = m2c.template pop<0>();
    data_t* sQ_ptr = (data_t*)get_slot_address(base, extract(slot_Q));
    auto sQ = make_tensor(make_smem_ptr(sQ_ptr), layout_sQ);
    auto sQ_vec = make_tensor(make_smem_ptr((vec2_t*)sQ_ptr), layout_sQ_vec);
    if (use_qwen_fused_qk) {
        rms_affine_rope_rows<HEAD_DIM, 128>(
            sQ_vec,
            num_active_q,
            smem_reduce,
            1.0e-6f,
            q_norm_weight,
            rope_row,
            runtime_need_norm,
            runtime_need_rope
        );
    }
    auto frag_Q = thr_mma_qk.partition_fragment_A(sQ);
    // Keep scores in the log2 domain expected by the exp2-based softmax path
    // without rewriting the swizzled Q tile in shared memory.
    const accum_t score_scale = static_cast<accum_t>(M_LOG2E / sqrtf((float)HEAD_DIM));

    // O layout
    Tensor t_dummyO = make_tensor(make_smem_ptr((accum_t*)nullptr), layout_sO);
    // alloc O registers
    auto frag_O = thr_mma_pv.partition_fragment_C(t_dummyO);
    clear(frag_O);
    auto o_mn_view = acc_get_mn_view<Q_BLOCK_SIZE, HEAD_DIM>(
        tiled_mma_pv.get_layoutC_TV(),
        frag_O
    );
    OnlineSoftmax<size<1>(o_mn_view), accum_t> online_softmax;

    // fragment P
    auto t_dummyP = make_tensor(make_smem_ptr((data_t_PV*)nullptr), layout_sP);
    auto frag_P = thr_mma_pv.partition_fragment_A(t_dummyP);

    auto frag_S = thr_mma_qk.partition_fragment_C(t_dummyS);
    auto s_mn_view = acc_get_mn_view<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(
        tiled_mma_qk.get_layoutC_TV(),
        frag_S
    );
    int slot_V, slot_K, slot_oldK;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        clear(frag_S);

        // load K
        slot_K = m2c.template pop<0>();
        data_t* sK_ptr = (data_t*)get_slot_address(base, extract(slot_K));
        auto sK = make_tensor(make_smem_ptr(sK_ptr), layout_sK);
        auto frag_K = thr_mma_qk.partition_fragment_B(sK);
        auto sK_vec = make_tensor(make_smem_ptr((vec2_t*)sK_ptr), layout_sK_vec);
        if (use_qwen_fused_qk && kv_block_idx == num_kv_blocks - 1 && last_kv_active_token_len > 0) {
            const int current_k_row = last_kv_active_token_len - 1;
            rms_affine_rope_single_row<HEAD_DIM>(
                sK_vec,
                current_k_row,
                smem_reduce,
                1.0e-6f,
                k_norm_weight,
                rope_row,
                runtime_need_norm,
                runtime_need_rope
            );
            if (thread_id < HEAD_DIM / 2) {
                sKStore_ptr[thread_id] = sK_vec(current_k_row, thread_id);
            }
            __sync_compute_group(128);
        }

        // 1. S = QK^T
        warpgroup_arrive();
        gemm(tiled_mma_qk, frag_Q, frag_K, frag_S);
        cuda::ptx::fence_proxy_async();
        warpgroup_commit_batch();

        // -- PV async region starts
        if (kv_block_idx > 0) {
            c2m.push(thread_id, slot_V);
            c2m.push(thread_id, slot_oldK);
        }
        // -- PV async region ends

        // wait for both previous O and current K
        warpgroup_wait<0>();
        #pragma unroll
        for (int r = 0; r < size<1>(s_mn_view); ++r) {
            #pragma unroll
            for (int c = 0; c < size<0>(s_mn_view); ++c) {
                s_mn_view(c, r) *= score_scale;
            }
        }
        if (kv_block_idx == num_kv_blocks - 1) {
            // mask invalid positions for the last block
            _mask(s_mn_view, last_kv_active_token_len);
        }
        
#if TMP_QK
        int slot_TR = m2c.template pop<0>();
        data_t* sTR_ptr = (data_t*)get_slot_address(base, extract(slot_TR));
        auto sTR = make_tensor(make_smem_ptr(sTR_ptr), layout_sR);
        copy(frag_S,
             thr_mma_qk.partition_C(sTR));

        // sync?
        // cuda::ptx::fence_proxy_async();
        c2m.push(thread_id, slot_TR);
#endif
        // push back temp result

        // 2. 3
        // convert S layout to row-wise
        online_softmax.update1(s_mn_view, o_mn_view);
#if TMP_ROW_MAX
        int slot_m = m2c.template pop<0>();
        accum_t* m_ptr = (accum_t*)get_slot_address(base, extract(slot_m));
        auto rowwise_layout = get_tv2m_layout<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(
            tiled_mma_qk.get_layoutC_TV()
        );
        int offset = threadIdx.x % 4;
        accum_t* rm = m_ptr + offset * Q_BLOCK_SIZE;
        auto st_rowmax = make_tensor(make_smem_ptr(rm), rowwise_layout);
        copy(online_softmax.row_max, st_rowmax(threadIdx.x / 4, _));
        __sync_compute_group(128);
        c2m.push(thread_id, slot_m);
        // end debug push back temp result
#endif
        

#if TMP_ROW_SUM
        cute::plus plus_op;
        auto tmp_row_reduced = make_tensor<accum_t>(Int<size<1>(o_mn_view)>{});
        copy(online_softmax.row_sum, tmp_row_reduced);
        butterfly_reduce<4>(tmp_row_reduced, plus_op);
        int slot_s = m2c.template pop<0>();
        accum_t* s_ptr = (accum_t*)get_slot_address(base, extract(slot_s));
        auto s_rowwise_layout = get_tv2m_layout<Q_TILE_SIZE, KV_BLOCK_SIZE>(
            tiled_mma_qk.get_layoutC_TV()
        );
        int soffset = threadIdx.x % 4;
        accum_t* s_rm = s_ptr + soffset * Q_TILE_SIZE;
        auto st_rowsum = make_tensor(make_smem_ptr(s_rm), s_rowwise_layout);
        copy(tmp_row_reduced, st_rowsum(threadIdx.x / 4, _));
        __sync_compute_group(128);
        c2m.push(thread_id, slot_s);
#endif

        // might require data type conversion here
        // TODO(zhiyuang): parallelized copy?
        copy(frag_S, frag_P);

        // load V
        slot_V = m2c.template pop<0>();
        data_t_PV* sV_ptr = (data_t_PV*)get_slot_address(base, extract(slot_V));
        auto sV = make_tensor(make_smem_ptr(sV_ptr), layout_sV);
        auto frag_V = thr_mma_pv.partition_fragment_B(sV);

#if TMP_EXP_P
        int slot_sP = m2c.template pop<0>();
        data_t* sP_ptr = (data_t*)get_slot_address(base, extract(slot_sP));
        auto sP = make_tensor(make_smem_ptr(sP_ptr), layout_sPR);
        copy(frag_P,
             thr_mma_pv.partition_A(sP));
        // sync?
        // cuda::ptx::fence_proxy_async();
        c2m.push(thread_id, slot_sP);
#endif

        // 4. P @ V
        warpgroup_arrive();
        gemm(tiled_mma_pv, frag_O, frag_P, frag_V, frag_O);
        cuda::ptx::fence_proxy_async();
        warpgroup_commit_batch();

        slot_oldK = slot_K;
        // TODO(zhiyuang): use P fragment and use FP16?
        online_softmax.update2(s_mn_view);

#if TMP_PV
        int slot_sPV = m2c.template pop<0>();
        data_t* sPV_ptr = (data_t*)get_slot_address(base, extract(slot_sPV));
        auto sPV = make_tensor(make_smem_ptr(sPV_ptr), layout_sQ);
        copy(frag_O,
             thr_mma_pv.partition_C(sPV));
        // sync?
        // cuda::ptx::fence_proxy_async();
        c2m.template thread_push<32>(slot_sPV);
#endif
    }
    c2m.push(thread_id, slot_V);
    c2m.push(thread_id, slot_oldK);
    c2m.push(thread_id, slot_Q);
    if (use_qwen_fused_qk) {
        c2m.push(thread_id, slot_side_input);
        c2m.template push<0, true>(thread_id, slot_k_store);
    }
    
    // final correction
    // row sum is still thread local now
    butterfly_reduce<4>(online_softmax.row_sum, cute::plus{});
    // wait for last O
    warpgroup_wait<0>();
    online_softmax.post_correction(o_mn_view);

    const int slot_O = m2c.template pop<0>();
    data_t* sO_ptr = (data_t*)get_slot_address(base, extract(slot_O));
    auto sO = make_tensor(make_smem_ptr(sO_ptr), layout_sO);
    auto partition_sO = thr_mma_pv.partition_C(sO);
    copy(frag_O, partition_sO);
    c2m.template push<0, true>(thread_id, slot_O);

    if constexpr (SPLIT_KV) {
        const int slot_lse = m2c.template pop<0>();
        // assume sM_glob is of shape [H, N, G, max_split]
        // each SM will get its slice of [N, G, max_split] so no need to index the KV head dim
        accum_t* __restrict__ sLSE_ptr = (accum_t*)slot_2_glob_ptr(st_insts, slot_lse);
        constexpr int tSRow = decltype(size<1>(o_mn_view))::value;
        #pragma unroll
        for (int r = 0; r < tSRow; ++r) {
            const int q_row = (thread_id / 32) * 16 + (thread_id % 32) / 4 + r * 8;
            if (q_row < num_active_q) {
                const accum_t lse = online_softmax.row_max(r) + log2f(online_softmax.row_sum(r));
                sLSE_ptr[q_row * MAX_SPLIT + split_idx] = lse;
            }
        }
        __sync_compute_group(128);
        c2m.template push<31, true, false>(thread_id, 1 << slot_lse);
    }
}

template <int HEAD_DIM,
          int Q_BLOCK_SIZE,
          int KV_BLOCK_SIZE,
          bool SPLIT_KV,
          int MAX_SPLIT,
          bool NEED_NORM, bool NEED_ROPE,
          typename AtomQK, typename AtomPV, typename M2C_Type, typename C2M_Type,
          template <class> class LayoutQAtom = cute::GMMA::Layout_K_SW128_Atom,
          template <class> class LayoutKAtom = cute::GMMA::Layout_K_SW128_Atom,
          template <class> class LayoutVAtom = cute::GMMA::Layout_MN_SW128_Atom>
__device__ __forceinline__ void task_attention_fwd_flash3_grouped_mma(
    const int num_kv_blocks,
    const int split_idx,
    const int num_active_q,
    const int last_kv_active_token_len,
    const int kv_start_idx,
    const bool runtime_need_norm,
    const bool runtime_need_rope,
    void *base,
    float *smem_reduce,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    using namespace cute;
    using AtomTrait = MMA_Traits<AtomQK>;
    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    using AtomTrait_PV = MMA_Traits<AtomPV>;
    using data_t_PV = typename AtomTrait_PV::ValTypeA;
    using accum_t_PV = typename AtomTrait_PV::ValTypeC;
    using Tr = F16Traits<data_t>;
    using vec2_t = typename Tr::vec2_t;

    static_assert(std::is_same<accum_t, accum_t_PV>::value, "accum type of QK and PV atom should be the same");
    static_assert(std::is_same<data_t, data_t_PV>::value, "QK and PV operand types must match");

    constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});

    constexpr int MMA_M_PV = shape<0>(typename AtomTrait_PV::Shape_MNK{});
    constexpr int MMA_N_PV = shape<1>(typename AtomTrait_PV::Shape_MNK{});
    constexpr int MMA_K_PV = shape<2>(typename AtomTrait_PV::Shape_MNK{});

    static_assert(Q_BLOCK_SIZE % MMA_M == 0, "Q block must be divisible by the QK atom M size");
    static_assert(Q_BLOCK_SIZE % MMA_M_PV == 0, "Q block must be divisible by the PV atom M size");
    static_assert(KV_BLOCK_SIZE % MMA_N == 0, "KV block must be divisible by the QK atom N size");
    static_assert(HEAD_DIM % MMA_K == 0, "Head dim must be divisible by the QK atom K size");
    static_assert(HEAD_DIM % MMA_N_PV == 0, "Head dim must be divisible by the PV atom N size");
    static_assert(KV_BLOCK_SIZE % MMA_K_PV == 0, "KV block must be divisible by the PV atom K size");

    constexpr int numThreadsQK = 32 * (Q_BLOCK_SIZE / MMA_M);
    constexpr int numThreadsPV = 32 * (Q_BLOCK_SIZE / MMA_M_PV);
    static_assert(numThreadsQK == 128, "Only support a 128-thread QK compute group for now");
    static_assert(numThreadsPV == 128, "Only support a 128-thread PV compute group for now");

    const int thread_id = __compute_tid();

    auto tiled_mma_qk = make_tiled_mma(
        MMA_Atom<AtomQK>{},
        make_layout(make_shape(Int<Q_BLOCK_SIZE / MMA_M>{}, Int<1>{}, Int<1>{})),
        make_tile(Int<Q_BLOCK_SIZE>{}, Int<KV_BLOCK_SIZE>{}, Int<HEAD_DIM>{})
    );
    auto tiled_mma_pv = make_tiled_mma(
        MMA_Atom<AtomPV>{},
        make_layout(make_shape(Int<Q_BLOCK_SIZE / MMA_M_PV>{}, Int<1>{}, Int<1>{})),
        make_tile(Int<Q_BLOCK_SIZE>{}, Int<HEAD_DIM>{}, Int<KV_BLOCK_SIZE>{})
    );

    auto layout_sQ = tile_to_shape(
        LayoutQAtom<data_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{}, Int<HEAD_DIM>{}));
    auto layout_sK = tile_to_shape(
        LayoutKAtom<data_t>{},
        make_shape(Int<KV_BLOCK_SIZE>{}, Int<HEAD_DIM>{}));
    auto layout_sV = tile_to_shape(
        LayoutVAtom<data_t>{},
        make_shape(Int<HEAD_DIM>{}, Int<KV_BLOCK_SIZE>{}));
    auto layout_sP = make_layout(
        make_shape(Int<Q_BLOCK_SIZE>{}, Int<KV_BLOCK_SIZE>{}),
        make_stride(Int<KV_BLOCK_SIZE>{}, Int<1>{})
    );
    auto layout_sO = make_layout(
        make_shape(Int<Q_BLOCK_SIZE>{}, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, Int<1>{})
    );

    auto layout_sQ_vec = tile_to_shape(
        LayoutQAtom<vec2_t>{},
        make_shape(Int<Q_BLOCK_SIZE>{}, Int<HEAD_DIM / 2>{}));
    auto layout_sK_vec = tile_to_shape(
        LayoutKAtom<vec2_t>{},
        make_shape(Int<KV_BLOCK_SIZE>{}, Int<HEAD_DIM / 2>{}));

    auto thr_mma_qk = tiled_mma_qk.get_slice(thread_id);
    auto thr_mma_pv = tiled_mma_pv.get_slice(thread_id);

    auto t_dummyQ = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sQ);
    auto t_dummyK = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sK);
    auto t_dummyV = make_tensor(make_smem_ptr(static_cast<data_t_PV*>(nullptr)), layout_sV);
    auto t_dummyP = make_tensor(make_smem_ptr(static_cast<data_t_PV*>(nullptr)), layout_sP);
    auto t_dummyS = make_tensor(
        make_smem_ptr(static_cast<accum_t*>(nullptr)),
        Layout<Shape<Int<Q_BLOCK_SIZE>, Int<KV_BLOCK_SIZE>>, Stride<Int<1>, Int<Q_BLOCK_SIZE>>>{}
    );
    auto t_dummyO = make_tensor(
        make_smem_ptr(static_cast<accum_t_PV*>(nullptr)),
        layout_sO
    );

    auto frag_Q = thr_mma_qk.partition_fragment_A(t_dummyQ);
    auto frag_K = thr_mma_qk.partition_fragment_B(t_dummyK);
    auto frag_S = thr_mma_qk.partition_fragment_C(t_dummyS);
    auto frag_P = thr_mma_pv.partition_fragment_A(t_dummyP);
    auto frag_V = thr_mma_pv.partition_fragment_B(t_dummyV);
    auto frag_O = thr_mma_pv.partition_fragment_C(t_dummyO);

    auto s_mn_view = acc_get_mn_view<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(
        tiled_mma_qk.get_layoutC_TV(),
        frag_S
    );
    auto o_mn_view = acc_get_mn_view<Q_BLOCK_SIZE, HEAD_DIM>(
        tiled_mma_pv.get_layoutC_TV(),
        frag_O
    );

    clear(frag_O);

    const bool use_qwen_fused_qk = runtime_need_norm || runtime_need_rope;
    int slot_side_input = 0;
    int slot_k_store = 0;
    const vec2_t* q_norm_weight = nullptr;
    const vec2_t* k_norm_weight = nullptr;
    const vec2_t* rope_row = nullptr;
    vec2_t* sKStore_ptr = nullptr;
    if (use_qwen_fused_qk) {
        slot_side_input = m2c.template pop<0>();
        const vec2_t* packed_side_input = (const vec2_t*)get_slot_address(base, extract(slot_side_input));
        q_norm_weight = packed_side_input;
        k_norm_weight = packed_side_input + HEAD_DIM / 2;
        rope_row = packed_side_input + HEAD_DIM;
        slot_k_store = m2c.template pop<0>();
        sKStore_ptr = (vec2_t*)get_slot_address(base, extract(slot_k_store));
    }

    int slot_Q = m2c.template pop<0>();
    data_t* sQ_ptr = (data_t*)get_slot_address(base, extract(slot_Q));
    auto sQ = make_tensor(make_smem_ptr(sQ_ptr), layout_sQ);
    auto sQ_vec = make_tensor(make_smem_ptr((vec2_t*)sQ_ptr), layout_sQ_vec);
    if (use_qwen_fused_qk) {
        rms_affine_rope_rows<HEAD_DIM, 128>(
            sQ_vec,
            Q_BLOCK_SIZE,
            smem_reduce,
            1.0e-6f,
            q_norm_weight,
            rope_row,
            runtime_need_norm,
            runtime_need_rope
        );
    }
    copy(thr_mma_qk.partition_A(sQ), frag_Q);

    const accum_t score_scale = static_cast<accum_t>(M_LOG2E / sqrtf((float)HEAD_DIM));
    OnlineSoftmax<size<1>(o_mn_view), accum_t> online_softmax;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        clear(frag_S);

        const int slot_K = m2c.template pop<0>();
        data_t* sK_ptr = (data_t*)get_slot_address(base, extract(slot_K));
        auto sK = make_tensor(make_smem_ptr(sK_ptr), layout_sK);
        auto sK_vec = make_tensor(make_smem_ptr((vec2_t*)sK_ptr), layout_sK_vec);

        if (use_qwen_fused_qk && kv_block_idx == num_kv_blocks - 1 && last_kv_active_token_len > 0) {
            const int current_k_row = last_kv_active_token_len - 1;
            rms_affine_rope_single_row<HEAD_DIM>(
                sK_vec,
                current_k_row,
                smem_reduce,
                1.0e-6f,
                k_norm_weight,
                rope_row,
                runtime_need_norm,
                runtime_need_rope
            );
            if (thread_id < HEAD_DIM / 2) {
                sKStore_ptr[thread_id] = sK_vec(current_k_row, thread_id);
            }
            __sync_compute_group(128);
        }

        copy(thr_mma_qk.partition_B(sK), frag_K);
        gemm(tiled_mma_qk, frag_S, frag_Q, frag_K, frag_S);

        #pragma unroll
        for (int r = 0; r < size<1>(s_mn_view); ++r) {
            #pragma unroll
            for (int c = 0; c < size<0>(s_mn_view); ++c) {
                s_mn_view(c, r) *= score_scale;
            }
        }
        if (kv_block_idx == num_kv_blocks - 1) {
            _mask(s_mn_view, last_kv_active_token_len);
        }

        online_softmax.update1(s_mn_view, o_mn_view);
        online_softmax.update2(s_mn_view);

        auto sP = make_tensor(make_smem_ptr((data_t_PV*)sK_ptr), layout_sP);
        copy(frag_S, thr_mma_qk.partition_C(sP));
        __sync_compute_group(128);

        copy(thr_mma_pv.partition_A(sP), frag_P);

        const int slot_V = m2c.template pop<0>();
        data_t_PV* sV_ptr = (data_t_PV*)get_slot_address(base, extract(slot_V));
        auto sV = make_tensor(make_smem_ptr(sV_ptr), layout_sV);
        copy(thr_mma_pv.partition_B(sV), frag_V);

        gemm(tiled_mma_pv, frag_O, frag_P, frag_V, frag_O);

        __sync_compute_group(128);
        c2m.push(thread_id, slot_V);
        c2m.push(thread_id, slot_K);
    }

    c2m.push(thread_id, slot_Q);
    if (use_qwen_fused_qk) {
        c2m.push(thread_id, slot_side_input);
        c2m.template push<0, true>(thread_id, slot_k_store);
    }

    butterfly_reduce<4>(online_softmax.row_sum, cute::plus{});
    online_softmax.post_correction(o_mn_view);

    const int slot_O = m2c.template pop<0>();
    data_t* sO_ptr = (data_t*)get_slot_address(base, extract(slot_O));
    auto sO = make_tensor(make_smem_ptr(sO_ptr), layout_sO);
    copy(frag_O, thr_mma_pv.partition_C(sO));
    c2m.template push<0, true>(thread_id, slot_O);

    if constexpr (SPLIT_KV) {
        const int slot_lse = m2c.template pop<0>();
        accum_t* __restrict__ sLSE_ptr = (accum_t*)slot_2_glob_ptr(st_insts, slot_lse);
        constexpr int tSRow = decltype(size<1>(o_mn_view))::value;
        #pragma unroll
        for (int r = 0; r < tSRow; ++r) {
            const int q_row = (thread_id / 32) * 16 + (thread_id % 32) / 4 + r * 8;
            if (q_row < num_active_q) {
                const accum_t lse = online_softmax.row_max(r) + log2f(online_softmax.row_sum(r));
                sLSE_ptr[q_row * MAX_SPLIT + split_idx] = lse;
            }
        }
        __sync_compute_group(128);
        c2m.template push<31, true, false>(thread_id, 1 << slot_lse);
    }
}

template <int HEAD_DIM, int NUM_Q, int MAX_SPLIT,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_split_post_reduce2_raw_direct(
    void *base,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m) {
    static_assert(NUM_Q == 4);
    static_assert(HEAD_DIM == 128);

    using namespace cute;
    using data_t = __nv_bfloat16;
    using accum_t = float;
    using Tr = F16Traits<data_t>;
    using vec2_t = typename Tr::vec2_t;
    struct alignas(8) vec4_t {
        vec2_t lo;
        vec2_t hi;
    };

    constexpr int THREADS_PER_Q = 32;
    constexpr int ELEMS_PER_THREAD = (HEAD_DIM / 2) / THREADS_PER_Q;
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id % THREADS_PER_Q;
    const int my_q = thread_id / THREADS_PER_Q;
    const int my_i_base = lane_id * ELEMS_PER_THREAD;

    const int slot_meta = m2c.template pop<0>();
    const uint8_t* record = static_cast<const uint8_t *>(
        slot_2_glob_ptr(st_insts, slot_meta));
    vec2_t* output = reinterpret_cast<vec2_t *>(
        *reinterpret_cast<const uint64_t *>(record));
    const accum_t* metadata = reinterpret_cast<const accum_t *>(
        record + sizeof(uint64_t));

    accum_t lane_max = -FLT_MAX;
    accum_t lane_mass = 0.0f;
    if (lane_id < 2) {
        lane_max = metadata[(my_q * 2 + 0) * MAX_SPLIT + lane_id];
        lane_mass = metadata[(my_q * 2 + 1) * MAX_SPLIT + lane_id];
    }
    const accum_t peer_max = __shfl_xor_sync(
        0xFFFFFFFFU, lane_max, 1);
    accum_t lane_scale = 0.0f;
    accum_t lane_weighted_mass = 0.0f;
    if (lane_id < 2) {
        lane_scale = exp2f(lane_max - fmaxf(lane_max, peer_max));
        lane_weighted_mass = lane_scale * lane_mass;
    }
    const accum_t peer_weighted_mass = __shfl_xor_sync(
        0xFFFFFFFFU, lane_weighted_mass, 1);
    const accum_t lane_weight = lane_id < 2
        ? lane_scale / (lane_weighted_mass + peer_weighted_mass)
        : 0.0f;
    const accum_t weight0 = __shfl_sync(
        0xFFFFFFFFU, lane_weight, 0);
    const accum_t weight1 = __shfl_sync(
        0xFFFFFFFFU, lane_weight, 1);

    const int slot_split = m2c.template pop<0>();
    const vec2_t* split_ptr = static_cast<const vec2_t *>(
        get_slot_address(base, extract(slot_split)));
    constexpr int VEC2_PER_SPLIT = NUM_Q * (HEAD_DIM / 2);
    const int output_idx = my_q * (HEAD_DIM / 2) + my_i_base;
    const vec4_t packed0 = *reinterpret_cast<const vec4_t *>(
        split_ptr + output_idx);
    const vec4_t packed1 = *reinterpret_cast<const vec4_t *>(
        split_ptr + VEC2_PER_SPLIT + output_idx);
    const float2 partial0_lo = Tr::to_float2(packed0.lo);
    const float2 partial0_hi = Tr::to_float2(packed0.hi);
    const float2 partial1_lo = Tr::to_float2(packed1.lo);
    const float2 partial1_hi = Tr::to_float2(packed1.hi);
    const vec4_t packed_output = {
        Tr::from_float2({
            partial0_lo.x * weight0 + partial1_lo.x * weight1,
            partial0_lo.y * weight0 + partial1_lo.y * weight1,
        }),
        Tr::from_float2({
            partial0_hi.x * weight0 + partial1_hi.x * weight1,
            partial0_hi.y * weight0 + partial1_hi.y * weight1,
        }),
    };
    *reinterpret_cast<vec4_t *>(output + output_idx) = packed_output;

    __sync_compute_group(128);
    c2m.push(thread_id, slot_split);
    c2m.template push<31, true, false>(
        thread_id, 1U << slot_meta);
}

template <int HEAD_DIM,
          int NUM_Q,
          int KV_BLOCK_SIZE,
          int MAX_SPLIT,
          int THREADS_PER_Q,   // tuning knob: threads assigned to each Q row
          bool RAW_PARTIAL = false,
          bool DIRECT_OUTPUT = false,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_split_post_reduce(
    const int num_split,
    void *base,
    float* smem_reduce,
    const MInst *st_insts,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
    static_assert(NUM_Q * THREADS_PER_Q <= 128, "active threads must not exceed warpgroup size");
    static_assert((HEAD_DIM / 2) % THREADS_PER_Q == 0, "HEAD_DIM/2 must be divisible by THREADS_PER_Q");
    static_assert(THREADS_PER_Q == 32, "split reduction assigns one warp to each Q row");

    // ELEMS_PER_THREAD: consecutive vec2 columns each active thread accumulates.
    // ACTIVE_THREADS: threads [0, ACTIVE_THREADS) do work; the rest participate in syncs only.
    constexpr int ELEMS_PER_THREAD = (HEAD_DIM / 2) / THREADS_PER_Q;
    constexpr int ACTIVE_THREADS   = NUM_Q * THREADS_PER_Q;

    using namespace cute;
    using data_t  = __nv_bfloat16;
    using accum_t = float;
    using Tr      = F16Traits<data_t>;
    using vec2_t  = typename Tr::vec2_t;

    const int thread_id = threadIdx.x;
    const int lane_id   = thread_id % 32;
    const int my_q      = thread_id / THREADS_PER_Q;
    const int my_i_base = (thread_id % THREADS_PER_Q) * ELEMS_PER_THREAD;

    auto layout_sO = make_layout(
        make_shape(Int<NUM_Q>{}, Int<HEAD_DIM/2>{}),
        LayoutRight{});
    auto layout_split_O = make_layout(
        make_shape(num_split, Int<NUM_Q>{}, Int<HEAD_DIM/2>{}),
        LayoutRight{});
    // LSE lives in global memory — available immediately (raw address, no TMA wait).
    const int slot_lse = m2c.template pop<0>();
    const void* metadata_record = slot_2_glob_ptr(st_insts, slot_lse);
    const accum_t* __restrict__ sLSE_ptr =
        static_cast<const accum_t *>(metadata_record);
    vec2_t* direct_output_ptr = nullptr;
    if constexpr (DIRECT_OUTPUT) {
        direct_output_ptr = reinterpret_cast<vec2_t *>(
            *static_cast<const uint64_t *>(metadata_record));
        sLSE_ptr = reinterpret_cast<const accum_t *>(
            static_cast<const uint8_t *>(metadata_record) + sizeof(uint64_t));
    }

    // Phase 1: one lane owns one split's LSE. Warp reductions and broadcasts
    // replace the previous 32-way redundant exp2 work for every output lane.
    accum_t lane_max = -FLT_MAX;
    accum_t lane_mass = 0.0f;
    if constexpr (RAW_PARTIAL) {
        auto layout_meta = make_layout(
            make_shape(Int<NUM_Q>{}, Int<2>{}, Int<MAX_SPLIT>{}),
            LayoutRight{});
        auto gMeta = make_tensor(make_gmem_ptr(sLSE_ptr), layout_meta);
        if (lane_id < num_split) {
            lane_max = gMeta(my_q, 0, lane_id);
            lane_mass = gMeta(my_q, 1, lane_id);
        }
    } else {
        auto layout_lse = make_layout(
            make_shape(Int<NUM_Q>{}, Int<MAX_SPLIT>{}),
            LayoutRight{});
        auto gLSE = make_tensor(make_gmem_ptr(sLSE_ptr), layout_lse);
        if (lane_id < num_split) {
            lane_max = gLSE(my_q, lane_id);
            lane_mass = 1.0f;
        }
    }
    accum_t max_all;
    asm volatile(
        "redux.sync.max.NaN.f32 %0, %1, 0xffffffff;\n"
        : "=f"(max_all) : "f"(lane_max));
    const accum_t lane_scale = lane_id < num_split
        ? exp2f(lane_max - max_all) : 0.0f;
    accum_t sum_all = lane_scale * lane_mass;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_all += __shfl_xor_sync(0xFFFFFFFFU, sum_all, offset);
    }

    // Block on the TMA only now — overlap with phase 1 gives it extra time to complete.
    const int slot_split = m2c.template pop<0>();
    const vec2_t* __restrict__ split_O_ptr = (vec2_t*)get_slot_address(base, extract(slot_split));
    auto split_O = make_tensor(make_smem_ptr(split_O_ptr), layout_split_O);

    // Phase 2: accumulate. No rolling sp term — global max is fixed, inner loop is pure fma.
    float2 acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) acc[e] = {0.f, 0.f};

    if (thread_id < ACTIVE_THREADS) {
        #pragma unroll
        for (int s = 0; s < num_split; ++s) {
            const accum_t sn = __shfl_sync(0xFFFFFFFFU, lane_scale, s);
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                float2 oo = Tr::to_float2(split_O(s, my_q, my_i_base + e));
                acc[e].x += oo.x * sn;
                acc[e].y += oo.y * sn;
            }
        }
    }

    // One sync covers all splits. Idle threads just wait here.
    __sync_compute_group(128);
    c2m.push(thread_id, slot_split);

    // Normalize and either publish directly or write to the TMA output slot.
    const accum_t inv_sum = 1.f / sum_all;
    if constexpr (DIRECT_OUTPUT) {
        if (thread_id < ACTIVE_THREADS) {
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                direct_output_ptr[
                    my_q * (HEAD_DIM / 2) + my_i_base + e] =
                    Tr::from_float2(
                        {acc[e].x * inv_sum, acc[e].y * inv_sum});
            }
        }
        c2m.template push<31, true, false>(
            thread_id, 1U << slot_lse);
    } else {
        const int slot_final = m2c.template pop<0>();
        vec2_t* sF_ptr = static_cast<vec2_t *>(
            get_slot_address(base, extract(slot_final)));
        auto sF = make_tensor(make_smem_ptr(sF_ptr), layout_sO);
        if (thread_id < ACTIVE_THREADS) {
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                sF(my_q, my_i_base + e) = Tr::from_float2(
                    {acc[e].x * inv_sum, acc[e].y * inv_sum});
            }
        }
        __sync_compute_group(128);
        c2m.template push<0, true>(thread_id, slot_final);
    }
}
