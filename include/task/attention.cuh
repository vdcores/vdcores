#pragma once

#include <cmath>
#include <cuda/atomic>
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
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

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
template <class CopyOp, class TD, class DLayout>
__device__ __forceinline__ void sm100_attention_tmem_load_raw(
    uint32_t tmem_addr,
    Tensor<TD, DLayout>& dst) {
    static_assert(is_rmem<TD>::value, "raw TMEM load expects register dst");
    using reg_t = typename remove_extent<typename CopyOp::DRegisters>::type;
    auto r_dst = recast<reg_t>(dst);
    constexpr int kRegisters = extent<typename CopyOp::DRegisters>::value;
    CUTE_STATIC_ASSERT_V(size(r_dst) == Int<kRegisters>{});
    detail::explode(
        CopyOp::copy, &tmem_addr, seq<0>{}, r_dst, make_seq<kRegisters>{});
}

template <class CopyOp, class TS, class SLayout>
__device__ __forceinline__ void sm100_attention_tmem_store_raw(
    Tensor<TS, SLayout>& src,
    uint32_t tmem_addr) {
    static_assert(is_rmem<TS>::value, "raw TMEM store expects register src");
    using reg_t = typename remove_extent<typename CopyOp::SRegisters>::type;
    auto r_src = recast<reg_t>(src);
    constexpr int kRegisters = extent<typename CopyOp::SRegisters>::value;
    CUTE_STATIC_ASSERT_V(size(r_src) == Int<kRegisters>{});
    detail::explode(
        CopyOp::copy, r_src, make_seq<kRegisters>{}, &tmem_addr, seq<0>{});
}

template <int M, int N, bool Sync = true,
          typename TmemTensor, typename CoordTensor>
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
    if constexpr (Sync) {
        __sync_compute_group(128);
    }
}

template <int M, int N, bool Sync = true,
          typename TmemTensor, typename CoordTensor, typename SmemTensor>
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
    if constexpr (Sync) {
        __sync_compute_group(128);
    }
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

// DeepSeek-V4 keeps 64 query heads in the UMMA M dimension.  Preserve more
// of the FP32 softmax value than a single BF16 P tile by storing a BF16 high
// term and a BF16 residual term in separate TMEM regions.  PV consumes both
// terms before advancing the online accumulator.
template <int M, int KV, typename TmemTensor, typename CoordTensor,
          typename ProbTensor, typename ProbCoordTensor>
__device__ __forceinline__ void sm100_dsv4_softmax_tmem_rows_hi_lo(
    TmemTensor const& tmem_s,
    CoordTensor const& coord_s,
    ProbTensor const& tmem_p_hi,
    ProbTensor const& tmem_p_lo,
    ProbCoordTensor const& coord_p,
    const float *sink,
    const int block,
    const int num_kv_blocks,
    const int last_kv_active_token_len,
    int& logical_row,
    float& row_max,
    float& row_sum,
    float& correction) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr float kScoreScale =
        M_LOG2E * 0.04419417382415922f;  // log2(e) / sqrt(512)
    const int tid = __compute_tid();
    correction = 0.0f;
    if (tid < 2 * M) {
        using ScoreLoad = SM100_TMEM_LOAD_16dp32b16x;
        auto tiled_s_load = make_tmem_copy(ScoreLoad{}, tmem_s);
        auto thr_s_load = tiled_s_load.get_slice(tid);
        auto thread_tmem_s = thr_s_load.partition_S(tmem_s);
        auto thread_coord_s = thr_s_load.partition_D(coord_s);
        const int row = int(get<0>(thread_coord_s(0)));
        logical_row = row;
        if (block == 0) {
            row_max = sink[row] * M_LOG2E;
            row_sum = 1.0f;
        }
        auto r_s = make_tensor<accum_t>(shape(thread_coord_s));
        copy(tiled_s_load, thread_tmem_s, r_s);

        float block_max = -FLT_MAX;
#pragma unroll 1
        for (int i = 0; i < size(r_s); ++i) {
            const int col = int(get<1>(thread_coord_s(i)));
            float score = r_s(i) * kScoreScale;
            if (block == num_kv_blocks - 1 &&
                col >= last_kv_active_token_len) {
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
        correction = exp2f(old_max - row_max);
        float block_sum = 0.0f;
#pragma unroll 1
        for (int i = 0; i < size(r_s); ++i) {
            const float probability = exp2f(r_s(i) - row_max);
            r_s(i) = probability;
            block_sum += probability;
        }
        block_sum += __shfl_xor_sync(
            0xFFFFFFFFU, block_sum, 16);
        row_sum = row_sum * correction + block_sum;

        using ProbStore = SM100_TMEM_STORE_16dp32b16x;
        auto tiled_p_store = make_tmem_copy(ProbStore{}, tmem_p_hi);
        auto thr_p_store = tiled_p_store.get_slice(tid);
        auto thread_coord_p = thr_p_store.partition_S(coord_p);
        auto thread_tmem_hi = thr_p_store.partition_D(tmem_p_hi);
        auto thread_tmem_lo = thr_p_store.partition_D(tmem_p_lo);
        auto r_hi = make_tensor<uint32_t>(shape(thread_coord_p));
        auto r_lo = make_tensor<uint32_t>(shape(thread_coord_p));
        auto r_hi_pairs = recast<cutlass::Array<data_t, 2>>(r_hi);
        auto r_lo_pairs = recast<cutlass::Array<data_t, 2>>(r_lo);
        cutlass::NumericArrayConverter<data_t, accum_t, 2> convert;
        const int lane_half = (tid >> 4) & 1;
#pragma unroll 1
        for (int i = 0; i < size(r_hi_pairs); ++i) {
            const int chunk = i / 16;
            const int pair_in_quarter = i & 7;
            const int local_idx =
                chunk * 32 + 16 * lane_half + 2 * pair_in_quarter;
            const int send_idx =
                chunk * 32 + 16 * (1 - lane_half) + 2 * pair_in_quarter;
            const bool use_partner = ((i >> 3) & 1) != lane_half;
            const accum_t partner_0 = __shfl_xor_sync(
                0xFFFFFFFFU, r_s(send_idx), 16);
            const accum_t partner_1 = __shfl_xor_sync(
                0xFFFFFFFFU, r_s(send_idx + 1), 16);
            cutlass::Array<accum_t, 2> pair;
            pair[0] = use_partner ? partner_0 : r_s(local_idx);
            pair[1] = use_partner ? partner_1 : r_s(local_idx + 1);
            const cutlass::Array<data_t, 2> high = convert(pair);
            cutlass::Array<accum_t, 2> residual;
            residual[0] = pair[0] - static_cast<accum_t>(high[0]);
            residual[1] = pair[1] - static_cast<accum_t>(high[1]);
            r_hi_pairs(i) = high;
            r_lo_pairs(i) = convert(residual);
        }
        copy(tiled_p_store, r_hi, thread_tmem_hi);
        copy(tiled_p_store, r_lo, thread_tmem_lo);
        cutlass::arch::fence_view_async_tmem_store();
    }
    __sync_compute_group(128);
}

// All 64 DeepSeek-V4 heads share one 512-wide KV cache.  Stream four native
// K128 TMA tiles for QK, keep scores/probabilities and four output chunks in
// TMEM, and stream four MN-major V tiles for PV.  Inputs and outputs stay in
// allocator-owned shared slots; STU remains the only global-output owner.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_contiguous_attention_512_umma(
    int rows,
    int output_tile_code,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t& tmem_mma_phase,
    void *smem_base,
    M2CQueue& m2c,
    C2MQueue& c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr int M = 64;
    constexpr int KV = 128;
    constexpr int D_TILE = 128;
    constexpr int D_TILES = 4;
    constexpr int QK_WAVE_TILES = 2;
    constexpr uint32_t P_HI_OFFSET = 0;
    constexpr uint32_t P_LO_OFFSET = 64;
    constexpr uint32_t O_OFFSET = 128;
    constexpr uint32_t O_STRIDE = 128;
    const bool output_sharded = output_tile_code != 0;

    using QKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, KV,
        UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = SM100_MMA_F16BF16_TS<
        data_t, data_t, accum_t, M, D_TILE,
        UMMA::Major::K, UMMA::Major::MN>;

    const int tid = __compute_tid();
    auto tiled_qk = make_tiled_mma(QKAtom{});
    auto cta_qk = tiled_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto cta_pv = tiled_pv.get_slice(0);

    auto q_shape = partition_shape_A(
        tiled_qk, make_shape(Int<M>{}, Int<D_TILE>{}));
    auto k_shape = partition_shape_B(
        tiled_qk, make_shape(Int<KV>{}, Int<D_TILE>{}));
    auto v_shape = partition_shape_B(
        tiled_pv, make_shape(Int<D_TILE>{}, Int<KV>{}));
    auto p_shape = partition_shape_A(
        tiled_pv, make_shape(Int<M>{}, Int<KV>{}));
    auto layout_q = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, q_shape);
    auto layout_k = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, k_shape);
    auto layout_v = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, v_shape);
    auto layout_p = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, p_shape);

    auto logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<KV>{}),
                    make_stride(Int<KV>{}, Int<1>{})));
    auto coord_s = make_identity_tensor(make_shape(Int<M>{}, Int<KV>{}));
    auto cta_s = cta_qk.partition_C(logical_s);
    auto cta_coord_s = cta_qk.partition_C(coord_s);
    auto tmem_s = cta_qk.make_fragment_C(cta_s);
    tmem_s.data() = tmem_base_ptr;

    auto tmem_p_hi = tmem_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));
    auto tmem_p_lo = tmem_p_hi;
    tmem_p_hi.data() = tmem_base_ptr + P_HI_OFFSET;
    tmem_p_lo.data() = tmem_base_ptr + P_LO_OFFSET;
    auto coord_p = cta_coord_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));
    auto dummy_p = make_tensor(
        make_smem_ptr(static_cast<data_t *>(nullptr)), layout_p);
    auto frag_p_hi = cta_pv.make_fragment_A(dummy_p);
    auto frag_p_lo = cta_pv.make_fragment_A(dummy_p);
    frag_p_hi.data() = tmem_base_ptr + P_HI_OFFSET;
    frag_p_lo.data() = tmem_base_ptr + P_LO_OFFSET;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                    make_stride(Int<D_TILE>{}, Int<1>{})));
    auto coord_o = make_identity_tensor(
        make_shape(Int<M>{}, Int<D_TILE>{}));
    auto cta_o = cta_pv.partition_C(logical_o);
    auto cta_coord_o = cta_pv.partition_C(coord_o);

    tiled_qk.accumulate_ = UMMA::ScaleOut::Zero;
    for (int wave = 0; wave < D_TILES / QK_WAVE_TILES; ++wave) {
        int q_slots[QK_WAVE_TILES];
        data_t *q_ptrs[QK_WAVE_TILES];
        int k_slots[QK_WAVE_TILES];
        data_t *k_ptrs[QK_WAVE_TILES];
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            q_slots[tile] = m2c.template pop<0>();
            q_ptrs[tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(q_slots[tile])));
        }
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            k_slots[tile] = m2c.template pop<0>();
            k_ptrs[tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(k_slots[tile])));
        }
        if (tid < numThreadsPerWarp) {
#pragma unroll 1
            for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
                auto sQ = make_tensor(make_smem_ptr(q_ptrs[tile]), layout_q);
                auto sK = make_tensor(make_smem_ptr(k_ptrs[tile]), layout_k);
                auto frag_q = cta_qk.make_fragment_A(sQ);
                auto frag_k = cta_qk.make_fragment_B(sK);
#pragma unroll 1
                for (int k_block = 0; k_block < size<2>(frag_q); ++k_block) {
                    gemm(tiled_qk, frag_q(_, _, k_block),
                         frag_k(_, _, k_block), tmem_s);
                    tiled_qk.accumulate_ = UMMA::ScaleOut::One;
                }
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            c2m.push(tid, q_slots[tile] | k_slots[tile]);
        }
    }

    int v_slots[D_TILES];
    data_t *v_ptrs[D_TILES];
    const int active_v_tiles = output_sharded ? 1 : D_TILES;
#pragma unroll 1
    for (int tile = 0; tile < active_v_tiles; ++tile) {
        v_slots[tile] = m2c.template pop<0>();
        v_ptrs[tile] = static_cast<data_t *>(
            get_slot_address(smem_base, extract(v_slots[tile])));
    }
    const int sink_slots = m2c.template pop<0>();
    const auto *sink = static_cast<const float *>(
        get_slot_address(smem_base, extract(sink_slots)));

    int logical_row = tid;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float correction = 0.0f;
    sm100_dsv4_softmax_tmem_rows_hi_lo<M, KV>(
        tmem_s, cta_coord_s, tmem_p_hi, tmem_p_lo, coord_p,
        sink, 0, 1, rows, logical_row,
        row_max, row_sum, correction);
    c2m.push(tid, sink_slots);

    const float inv_sum =
        logical_row < M && row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    // Four-SM mode duplicates QK/softmax but assigns one D128 PV/store tile
    // to each SM. The branch is uniform for all 128 compute threads. It adds
    // no compute-group or cross-SM barrier; the ordinary C2M writeback edge
    // remains the sole shared-to-global publication mechanism.
    if (output_sharded) {
        if (tid < numThreadsPerWarp) {
            auto sV = make_tensor(make_smem_ptr(v_ptrs[0]), layout_v);
            auto frag_v = cta_pv.make_fragment_B(sV);
            auto tmem_o = cta_pv.make_fragment_C(cta_o);
            tmem_o.data() = tmem_base_ptr + O_OFFSET;
            tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(frag_p_hi); ++k_block) {
                gemm(tiled_pv, frag_p_hi(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(frag_p_lo); ++k_block) {
                gemm(tiled_pv, frag_p_lo(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, v_slots[0]);

        const int output_slots = m2c.template pop<0>();
        auto *output_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(output_slots)));
        auto sO = make_tensor(
            make_smem_ptr(output_ptr),
            make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                        make_stride(Int<D_TILE>{}, Int<1>{})));
        auto cta_sO = cta_pv.partition_C(sO);
        auto tmem_o = cta_pv.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + O_OFFSET;
        sm100_attention_store_tmem_rows<M, D_TILE, false>(
            tmem_o, cta_coord_o, cta_sO, inv_sum, M);
        c2m.template push<31, true, false>(tid, output_slots);
        return;
    }

    if (tid < numThreadsPerWarp) {
#pragma unroll 1
        for (int tile = 0; tile < D_TILES - 1; ++tile) {
            auto sV = make_tensor(make_smem_ptr(v_ptrs[tile]), layout_v);
            auto frag_v = cta_pv.make_fragment_B(sV);
            auto tmem_o = cta_pv.make_fragment_C(cta_o);
            tmem_o.data() = tmem_base_ptr + O_OFFSET + tile * O_STRIDE;
            tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
            for (int k_block = 0; k_block < size<2>(frag_p_hi); ++k_block) {
                gemm(tiled_pv, frag_p_hi(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0; k_block < size<2>(frag_p_lo); ++k_block) {
                gemm(tiled_pv, frag_p_lo(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
        }
        cutlass::arch::umma_arrive(tmem_mma_barrier);
    }
    cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
    tmem_mma_phase ^= 1;
#pragma unroll 1
    for (int tile = 0; tile < D_TILES - 1; ++tile) {
        c2m.push(tid, v_slots[tile]);
    }

    const int first_output_slots = m2c.template pop<0>();
    auto *first_output_ptr = static_cast<data_t *>(
        get_slot_address(smem_base, extract(first_output_slots)));
    auto first_sO = make_tensor(
        make_smem_ptr(first_output_ptr),
        make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                    make_stride(Int<D_TILE>{}, Int<1>{})));
    auto first_cta_sO = cta_pv.partition_C(first_sO);
    auto first_tmem_o = cta_pv.make_fragment_C(cta_o);
    first_tmem_o.data() = tmem_base_ptr + O_OFFSET;
    sm100_attention_store_tmem_rows<M, D_TILE>(
        first_tmem_o, cta_coord_o, first_cta_sO, inv_sum, M);
    c2m.template push<31, true, false>(tid, first_output_slots);

    if (tid < numThreadsPerWarp) {
        auto sV = make_tensor(make_smem_ptr(v_ptrs[D_TILES - 1]), layout_v);
        auto frag_v = cta_pv.make_fragment_B(sV);
        auto tmem_o = cta_pv.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + O_OFFSET;
        tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
        for (int k_block = 0; k_block < size<2>(frag_p_hi); ++k_block) {
            gemm(tiled_pv, frag_p_hi(_, _, k_block),
                 frag_v(_, _, k_block), tmem_o);
            tiled_pv.accumulate_ = UMMA::ScaleOut::One;
        }
#pragma unroll 1
        for (int k_block = 0; k_block < size<2>(frag_p_lo); ++k_block) {
            gemm(tiled_pv, frag_p_lo(_, _, k_block),
                 frag_v(_, _, k_block), tmem_o);
            tiled_pv.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(tmem_mma_barrier);
    }
    cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
    tmem_mma_phase ^= 1;
    c2m.push(tid, v_slots[D_TILES - 1]);

#pragma unroll 1
    for (int tile = 1; tile < D_TILES; ++tile) {
        const int output_slots = m2c.template pop<0>();
        auto *output_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(output_slots)));
        auto sO = make_tensor(
            make_smem_ptr(output_ptr),
            make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                        make_stride(Int<D_TILE>{}, Int<1>{})));
        auto cta_sO = cta_pv.partition_C(sO);
        auto tmem_o = cta_pv.make_fragment_C(cta_o);
        const int tmem_tile = tile == D_TILES - 1 ? 0 : tile;
        tmem_o.data() = tmem_base_ptr + O_OFFSET + tmem_tile * O_STRIDE;
        sm100_attention_store_tmem_rows<M, D_TILE, false>(
            tmem_o, cta_coord_o, cta_sO, inv_sum, M);
        c2m.template push<31, true, false>(tid, output_slots);
    }
}

// Normalize a dense K128 prefix and a live K<=32 tail in one softmax pass.
// Both score fragments stay in TMEM until the shared-memory high/residual P
// tiles are written, so no score or partial-output copy crosses a task stage.
template <int M, int PREFIX_KV, int TAIL_KV,
          typename PrefixTmemTensor, typename PrefixCoordTensor,
          typename TailTmemTensor, typename TailCoordTensor,
          typename PrefixProbTensor, typename TailProbTensor>
__device__ __forceinline__ void
sm100_dsv4_softmax_tmem_smem_prefix_tail_hi_lo(
    PrefixTmemTensor const& prefix_scores,
    PrefixCoordTensor const& prefix_coords,
    TailTmemTensor const& tail_scores,
    TailCoordTensor const& tail_coords,
    PrefixProbTensor const& prefix_p_hi,
    PrefixProbTensor const& prefix_p_lo,
    TailProbTensor const& tail_p_hi,
    TailProbTensor const& tail_p_lo,
    const float *sink,
    const int tail_tokens,
    int& logical_row,
    float& row_sum) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr float kScoreScale =
        M_LOG2E * 0.04419417382415922f;
    const int tid = __compute_tid();

    using PrefixLoad = SM100_TMEM_LOAD_16dp32b16x;
    auto prefix_load = make_tmem_copy(PrefixLoad{}, prefix_scores);
    auto prefix_thread = prefix_load.get_slice(tid);
    auto thread_prefix_scores =
        prefix_thread.partition_S(prefix_scores);
    auto thread_prefix_coords =
        prefix_thread.partition_D(prefix_coords);
    auto r_prefix = make_tensor<accum_t>(shape(thread_prefix_coords));
    copy(prefix_load, thread_prefix_scores, r_prefix);

    using TailLoad = SM100_TMEM_LOAD_16dp32b16x;
    auto tail_load = make_tmem_copy(TailLoad{}, tail_scores);
    auto tail_thread = tail_load.get_slice(tid);
    auto thread_tail_scores = tail_thread.partition_S(tail_scores);
    auto thread_tail_coords = tail_thread.partition_D(tail_coords);
    auto r_tail = make_tensor<accum_t>(shape(thread_tail_coords));
    copy(tail_load, thread_tail_scores, r_tail);

    logical_row = int(get<0>(thread_prefix_coords(0)));
    float row_max = sink[logical_row] * M_LOG2E;
#pragma unroll 1
    for (int i = 0; i < size(r_prefix); ++i) {
        r_prefix(i) *= kScoreScale;
        row_max = fmaxf(row_max, r_prefix(i));
    }
#pragma unroll 1
    for (int i = 0; i < size(r_tail); ++i) {
        const int col = int(get<1>(thread_tail_coords(i)));
        r_tail(i) = col < tail_tokens
            ? r_tail(i) * kScoreScale : -FLT_MAX;
        row_max = fmaxf(row_max, r_tail(i));
    }
    row_max = fmaxf(
        row_max, __shfl_xor_sync(0xFFFFFFFFU, row_max, 16));

    const float sink_prob =
        exp2f(sink[logical_row] * M_LOG2E - row_max);
    float local_sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < size(r_prefix); ++i) {
        r_prefix(i) = exp2f(r_prefix(i) - row_max);
        local_sum += r_prefix(i);
    }
#pragma unroll 1
    for (int i = 0; i < size(r_tail); ++i) {
        r_tail(i) = exp2f(r_tail(i) - row_max);
        local_sum += r_tail(i);
    }
    row_sum = sink_prob + local_sum +
        __shfl_xor_sync(0xFFFFFFFFU, local_sum, 16);

    cutlass::NumericArrayConverter<data_t, accum_t, 1> convert;
#pragma unroll 1
    for (int i = 0; i < size(r_prefix); ++i) {
        const int row = int(get<0>(thread_prefix_coords(i)));
        const int col = int(get<1>(thread_prefix_coords(i)));
        cutlass::Array<accum_t, 1> value{{r_prefix(i)}};
        const auto high = convert(value);
        cutlass::Array<accum_t, 1> residual{{
            r_prefix(i) - static_cast<accum_t>(high[0])}};
        prefix_p_hi(row + M * col) = high[0];
        prefix_p_lo(row + M * col) = convert(residual)[0];
    }
#pragma unroll 1
    for (int i = 0; i < size(r_tail); ++i) {
        const int row = int(get<0>(thread_tail_coords(i)));
        const int col = int(get<1>(thread_tail_coords(i)));
        cutlass::Array<accum_t, 1> value{{r_tail(i)}};
        const auto high = convert(value);
        cutlass::Array<accum_t, 1> residual{{
            r_tail(i) - static_cast<accum_t>(high[0])}};
        tail_p_hi(row + M * col) = high[0];
        tail_p_lo(row + M * col) = convert(residual)[0];
    }

    // The same 128 compute threads publish all P fragments. Join once before
    // the issuer warp consumes them through the asynchronous tensor proxy.
    __sync_compute_group(128);
    cutlass::arch::fence_view_async_shared();
}

// Contexts 129--160 use a native K128 prefix plus a native K32 tail. Q is
// retained only through both QK products, then its allocator slots are reused
// in place as shared P storage. Prefix/tail V pairs stream one output tile at
// a time while four FP32 output tiles remain in TMEM.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_contiguous_attention_512_umma_tail32(
    int rows,
    int output_tile_code,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t& tmem_mma_phase,
    void *smem_base,
    M2CQueue& m2c,
    C2MQueue& c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr int M = 64;
    constexpr int PREFIX_KV = 128;
    constexpr int TAIL_KV = 32;
    constexpr int D_TILE = 128;
    constexpr int D_TILES = 4;
    constexpr int QK_WAVE_TILES = 2;
    constexpr uint32_t TAIL_SCORE_OFFSET = PREFIX_KV;
    const bool output_sharded = output_tile_code != 0;
    const int first_output_tile = output_sharded
        ? output_tile_code - 1 : 0;
    const int output_tile_count = output_sharded ? 1 : D_TILES;

    using PrefixQKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, PREFIX_KV,
        UMMA::Major::K, UMMA::Major::K>;
    using TailQKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, TAIL_KV,
        UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, D_TILE,
        UMMA::Major::K, UMMA::Major::MN>;

    const int tid = __compute_tid();
    const int tail_tokens = rows - PREFIX_KV;
    auto prefix_qk = make_tiled_mma(PrefixQKAtom{});
    auto prefix_qk_cta = prefix_qk.get_slice(0);
    auto tail_qk = make_tiled_mma(TailQKAtom{});
    auto tail_qk_cta = tail_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto pv_cta = tiled_pv.get_slice(0);

    auto prefix_q_shape = partition_shape_A(
        prefix_qk, make_shape(Int<M>{}, Int<D_TILE>{}));
    auto prefix_k_shape = partition_shape_B(
        prefix_qk, make_shape(Int<PREFIX_KV>{}, Int<D_TILE>{}));
    auto tail_q_shape = partition_shape_A(
        tail_qk, make_shape(Int<M>{}, Int<D_TILE>{}));
    auto tail_k_shape = partition_shape_B(
        tail_qk, make_shape(Int<TAIL_KV>{}, Int<D_TILE>{}));
    auto prefix_v_shape = partition_shape_B(
        tiled_pv, make_shape(Int<D_TILE>{}, Int<PREFIX_KV>{}));
    auto tail_v_shape = partition_shape_B(
        tiled_pv, make_shape(Int<D_TILE>{}, Int<TAIL_KV>{}));
    auto prefix_p_shape = partition_shape_A(
        tiled_pv, make_shape(Int<M>{}, Int<PREFIX_KV>{}));
    auto tail_p_shape = partition_shape_A(
        tiled_pv, make_shape(Int<M>{}, Int<TAIL_KV>{}));

    auto prefix_q_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, prefix_q_shape);
    auto prefix_k_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, prefix_k_shape);
    auto tail_q_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, tail_q_shape);
    auto tail_k_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, tail_k_shape);
    auto prefix_v_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, prefix_v_shape);
    auto tail_v_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, tail_v_shape);
    auto prefix_p_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, prefix_p_shape);
    auto tail_p_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW32_Atom<data_t>{}, tail_p_shape);

    auto prefix_logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<PREFIX_KV>{}),
                    make_stride(Int<PREFIX_KV>{}, Int<1>{})));
    auto prefix_coord_s = make_identity_tensor(
        make_shape(Int<M>{}, Int<PREFIX_KV>{}));
    auto prefix_cta_s = prefix_qk_cta.partition_C(prefix_logical_s);
    auto prefix_cta_coord_s =
        prefix_qk_cta.partition_C(prefix_coord_s);
    auto prefix_tmem_s = prefix_qk_cta.make_fragment_C(prefix_cta_s);
    prefix_tmem_s.data() = tmem_base_ptr;

    auto tail_logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<TAIL_KV>{}),
                    make_stride(Int<TAIL_KV>{}, Int<1>{})));
    auto tail_coord_s = make_identity_tensor(
        make_shape(Int<M>{}, Int<TAIL_KV>{}));
    auto tail_cta_s = tail_qk_cta.partition_C(tail_logical_s);
    auto tail_cta_coord_s = tail_qk_cta.partition_C(tail_coord_s);
    auto tail_tmem_s = tail_qk_cta.make_fragment_C(tail_cta_s);
    tail_tmem_s.data() = tmem_base_ptr + TAIL_SCORE_OFFSET;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                    make_stride(Int<D_TILE>{}, Int<1>{})));
    auto coord_o = make_identity_tensor(
        make_shape(Int<M>{}, Int<D_TILE>{}));
    auto cta_o = pv_cta.partition_C(logical_o);
    auto cta_coord_o = pv_cta.partition_C(coord_o);

    int q_slots[D_TILES];
    data_t *q_ptrs[D_TILES];
    prefix_qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
    for (int wave = 0; wave < D_TILES / QK_WAVE_TILES; ++wave) {
        int k_slots[QK_WAVE_TILES];
        data_t *k_ptrs[QK_WAVE_TILES];
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            const int q_tile = wave * QK_WAVE_TILES + tile;
            q_slots[q_tile] = m2c.template pop<0>();
            q_ptrs[q_tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(q_slots[q_tile])));
        }
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            k_slots[tile] = m2c.template pop<0>();
            k_ptrs[tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(k_slots[tile])));
        }
        if (tid < numThreadsPerWarp) {
#pragma unroll 1
            for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
                const int q_tile = wave * QK_WAVE_TILES + tile;
                auto sQ = make_tensor(
                    make_smem_ptr(q_ptrs[q_tile]), prefix_q_layout);
                auto sK = make_tensor(
                    make_smem_ptr(k_ptrs[tile]), prefix_k_layout);
                auto frag_q = prefix_qk_cta.make_fragment_A(sQ);
                auto frag_k = prefix_qk_cta.make_fragment_B(sK);
#pragma unroll 1
                for (int k_block = 0;
                     k_block < size<2>(frag_q); ++k_block) {
                    gemm(prefix_qk, frag_q(_, _, k_block),
                         frag_k(_, _, k_block), prefix_tmem_s);
                    prefix_qk.accumulate_ = UMMA::ScaleOut::One;
                }
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            c2m.push(tid, k_slots[tile]);
        }
    }

    int tail_k_slots[D_TILES];
    data_t *tail_k_ptrs[D_TILES];
#pragma unroll 1
    for (int tile = 0; tile < D_TILES; ++tile) {
        tail_k_slots[tile] = m2c.template pop<0>();
        tail_k_ptrs[tile] = static_cast<data_t *>(
            get_slot_address(smem_base, extract(tail_k_slots[tile])));
    }
    tail_qk.accumulate_ = UMMA::ScaleOut::Zero;
    if (tid < numThreadsPerWarp) {
#pragma unroll
        for (int tile = 0; tile < D_TILES; ++tile) {
            auto sQ = make_tensor(
                make_smem_ptr(q_ptrs[tile]), tail_q_layout);
            auto sK = make_tensor(
                make_smem_ptr(tail_k_ptrs[tile]), tail_k_layout);
            auto frag_q = tail_qk_cta.make_fragment_A(sQ);
            auto frag_k = tail_qk_cta.make_fragment_B(sK);
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(frag_q); ++k_block) {
                gemm(tail_qk, frag_q(_, _, k_block),
                     frag_k(_, _, k_block), tail_tmem_s);
                tail_qk.accumulate_ = UMMA::ScaleOut::One;
            }
        }
        cutlass::arch::umma_arrive(tmem_mma_barrier);
    }
    cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
    tmem_mma_phase ^= 1;
#pragma unroll 1
    for (int tile = 0; tile < D_TILES; ++tile) {
        c2m.push(tid, tail_k_slots[tile]);
    }

    const int sink_slots = m2c.template pop<0>();
    const auto *sink = static_cast<const float *>(
        get_slot_address(smem_base, extract(sink_slots)));
    auto prefix_p_hi = make_tensor(
        make_smem_ptr(q_ptrs[0]), prefix_p_layout);
    auto prefix_p_lo = make_tensor(
        make_smem_ptr(q_ptrs[1]), prefix_p_layout);
    auto tail_p_hi = make_tensor(
        make_smem_ptr(q_ptrs[2]), tail_p_layout);
    auto tail_p_lo = make_tensor(
        make_smem_ptr(q_ptrs[2] + M * TAIL_KV),
        tail_p_layout);
    int logical_row = tid;
    float row_sum = 0.0f;
    sm100_dsv4_softmax_tmem_smem_prefix_tail_hi_lo<
        M, PREFIX_KV, TAIL_KV>(
        prefix_tmem_s, prefix_cta_coord_s,
        tail_tmem_s, tail_cta_coord_s,
        prefix_p_hi, prefix_p_lo, tail_p_hi, tail_p_lo,
        sink, tail_tokens, logical_row, row_sum);
    c2m.push(tid, sink_slots | q_slots[3]);

    auto prefix_frag_p_hi = pv_cta.make_fragment_A(prefix_p_hi);
    auto prefix_frag_p_lo = pv_cta.make_fragment_A(prefix_p_lo);
    auto tail_frag_p_hi = pv_cta.make_fragment_A(tail_p_hi);
    auto tail_frag_p_lo = pv_cta.make_fragment_A(tail_p_lo);
#pragma unroll 1
    for (int output_index = 0;
         output_index < output_tile_count; ++output_index) {
        const int tile = first_output_tile + output_index;
        const int prefix_v_slots = m2c.template pop<0>();
        auto *prefix_v_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(prefix_v_slots)));
        const int tail_v_slots = m2c.template pop<0>();
        auto *tail_v_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(tail_v_slots)));
        auto prefix_v = make_tensor(
            make_smem_ptr(prefix_v_ptr), prefix_v_layout);
        auto tail_v = make_tensor(
            make_smem_ptr(tail_v_ptr), tail_v_layout);
        auto prefix_frag_v = pv_cta.make_fragment_B(prefix_v);
        auto tail_frag_v = pv_cta.make_fragment_B(tail_v);
        auto tmem_o = pv_cta.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + tile * D_TILE;
        tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
        if (tid < numThreadsPerWarp) {
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(prefix_frag_p_hi); ++k_block) {
                gemm(tiled_pv, prefix_frag_p_hi(_, _, k_block),
                     prefix_frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(prefix_frag_p_lo); ++k_block) {
                gemm(tiled_pv, prefix_frag_p_lo(_, _, k_block),
                     prefix_frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(tail_frag_p_hi); ++k_block) {
                gemm(tiled_pv, tail_frag_p_hi(_, _, k_block),
                     tail_frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0;
                 k_block < size<2>(tail_frag_p_lo); ++k_block) {
                gemm(tiled_pv, tail_frag_p_lo(_, _, k_block),
                     tail_frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, prefix_v_slots | tail_v_slots);
    }
    c2m.push(tid, q_slots[0] | q_slots[1] | q_slots[2]);

    const float inv_sum =
        logical_row < M && row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
#pragma unroll 1
    for (int output_index = 0;
         output_index < output_tile_count; ++output_index) {
        const int tile = first_output_tile + output_index;
        const int output_slots = m2c.template pop<0>();
        auto *output_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(output_slots)));
        auto sO = make_tensor(
            make_smem_ptr(output_ptr),
            make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                        make_stride(Int<D_TILE>{}, Int<1>{})));
        auto cta_sO = pv_cta.partition_C(sO);
        auto tmem_o = pv_cta.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + tile * D_TILE;
        // Each output owns a disjoint TMEM region.  The C2M writeback queue
        // already collects all 128 compute-thread arrivals after their
        // shared-proxy fence, so an additional compute-group join here has
        // no producer/consumer edge to enforce.
        sm100_attention_store_tmem_rows<M, D_TILE, false>(
            tmem_o, cta_coord_o, cta_sO, inv_sum, M);
        c2m.template push<31, true, false>(tid, output_slots);
    }
}

// A split-KV producer owns one contiguous K32 shard while all 64 attention
// heads occupy UMMA's M dimension.  The producer emits a locally normalized
// BF16 partial plus (max, mass) metadata.  A later per-head reducer merges the
// shards, adds the attention sink, applies inverse RoPE, and writes native O_a
// FP8 records directly.
template <int M, int KV, typename ScoreTensor, typename CoordTensor,
          typename ProbTensor>
__device__ __forceinline__ void sm100_dsv4_softmax_split32_hi_lo(
    ScoreTensor const& scores,
    CoordTensor const& coords,
    ProbTensor const& p_hi,
    ProbTensor const& p_lo,
    const int active_tokens,
    int& logical_row,
    float& row_max,
    float& row_sum) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr float kScoreScale =
        M_LOG2E * 0.04419417382415922f;
    const int tid = __compute_tid();

    using ScoreLoad = SM100_TMEM_LOAD_16dp32b16x;
    auto tiled_load = make_tmem_copy(ScoreLoad{}, scores);
    auto thread_load = tiled_load.get_slice(tid);
    auto thread_scores = thread_load.partition_S(scores);
    auto thread_coords = thread_load.partition_D(coords);
    auto values = make_tensor<accum_t>(shape(thread_coords));
    copy(tiled_load, thread_scores, values);

    logical_row = int(get<0>(thread_coords(0)));
    row_max = -FLT_MAX;
#pragma unroll 1
    for (int i = 0; i < size(values); ++i) {
        const int col = int(get<1>(thread_coords(i)));
        values(i) = col < active_tokens
            ? values(i) * kScoreScale : -FLT_MAX;
        row_max = fmaxf(row_max, values(i));
    }
    row_max = fmaxf(
        row_max, __shfl_xor_sync(0xFFFFFFFFU, row_max, 16));

    float local_sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < size(values); ++i) {
        values(i) = exp2f(values(i) - row_max);
        local_sum += values(i);
    }
    row_sum = local_sum +
        __shfl_xor_sync(0xFFFFFFFFU, local_sum, 16);

    cutlass::NumericArrayConverter<data_t, accum_t, 1> convert;
#pragma unroll 1
    for (int i = 0; i < size(values); ++i) {
        const int row = int(get<0>(thread_coords(i)));
        const int col = int(get<1>(thread_coords(i)));
        cutlass::Array<accum_t, 1> source{{values(i)}};
        const auto high = convert(source);
        cutlass::Array<accum_t, 1> residual{{
            values(i) - static_cast<accum_t>(high[0])}};
        p_hi(row + M * col) = high[0];
        p_lo(row + M * col) = convert(residual)[0];
    }
    __sync_compute_group(128);
    cutlass::arch::fence_view_async_shared();
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_attention_split32_umma_sm100(
    int active_tokens,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t& tmem_mma_phase,
    void *smem_base,
    M2CQueue& m2c,
    C2MQueue& c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr int M = 64;
    constexpr int KV = 32;
    constexpr int D_TILE = 128;
    constexpr int D_TILES = 4;
    constexpr int QK_WAVE_TILES = 2;
    constexpr uint32_t O_OFFSET = 128;

    using QKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, KV,
        UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, D_TILE,
        UMMA::Major::K, UMMA::Major::MN>;

    const int tid = __compute_tid();
    auto tiled_qk = make_tiled_mma(QKAtom{});
    auto qk_cta = tiled_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto pv_cta = tiled_pv.get_slice(0);

    auto q_shape = partition_shape_A(
        tiled_qk, make_shape(Int<M>{}, Int<D_TILE>{}));
    auto k_shape = partition_shape_B(
        tiled_qk, make_shape(Int<KV>{}, Int<D_TILE>{}));
    auto v_shape = partition_shape_B(
        tiled_pv, make_shape(Int<D_TILE>{}, Int<KV>{}));
    auto p_shape = partition_shape_A(
        tiled_pv, make_shape(Int<M>{}, Int<KV>{}));
    auto q_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, q_shape);
    auto k_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW32_Atom<data_t>{}, k_shape);
    auto v_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, v_shape);
    auto p_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW32_Atom<data_t>{}, p_shape);

    auto logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<KV>{}),
                    make_stride(Int<KV>{}, Int<1>{})));
    auto coord_s = make_identity_tensor(make_shape(Int<M>{}, Int<KV>{}));
    auto cta_s = qk_cta.partition_C(logical_s);
    auto cta_coord_s = qk_cta.partition_C(coord_s);
    auto tmem_s = qk_cta.make_fragment_C(cta_s);
    tmem_s.data() = tmem_base_ptr;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                    make_stride(Int<D_TILE>{}, Int<1>{})));
    auto coord_o = make_identity_tensor(make_shape(Int<M>{}, Int<D_TILE>{}));
    auto cta_o = pv_cta.partition_C(logical_o);
    auto cta_coord_o = pv_cta.partition_C(coord_o);

    int q_slots[D_TILES];
    data_t *q_ptrs[D_TILES];
    tiled_qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
    for (int wave = 0; wave < D_TILES / QK_WAVE_TILES; ++wave) {
        int k_slots[QK_WAVE_TILES];
        data_t *k_ptrs[QK_WAVE_TILES];
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            const int q_tile = wave * QK_WAVE_TILES + tile;
            q_slots[q_tile] = m2c.template pop<0>();
            q_ptrs[q_tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(q_slots[q_tile])));
        }
#pragma unroll 1
        for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
            k_slots[tile] = m2c.template pop<0>();
            k_ptrs[tile] = static_cast<data_t *>(
                get_slot_address(smem_base, extract(k_slots[tile])));
        }
        if (tid < numThreadsPerWarp) {
#pragma unroll 1
            for (int tile = 0; tile < QK_WAVE_TILES; ++tile) {
                const int q_tile = wave * QK_WAVE_TILES + tile;
                auto sQ = make_tensor(make_smem_ptr(q_ptrs[q_tile]), q_layout);
                auto sK = make_tensor(make_smem_ptr(k_ptrs[tile]), k_layout);
                auto frag_q = qk_cta.make_fragment_A(sQ);
                auto frag_k = qk_cta.make_fragment_B(sK);
#pragma unroll 1
                for (int k_block = 0; k_block < size<2>(frag_q); ++k_block) {
                    gemm(tiled_qk, frag_q(_, _, k_block),
                         frag_k(_, _, k_block), tmem_s);
                    tiled_qk.accumulate_ = UMMA::ScaleOut::One;
                }
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, k_slots[0] | k_slots[1]);
    }

    auto p_hi = make_tensor(make_smem_ptr(q_ptrs[0]), p_layout);
    auto p_lo = make_tensor(make_smem_ptr(q_ptrs[0] + M * KV), p_layout);
    int logical_row = 0;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    sm100_dsv4_softmax_split32_hi_lo<M, KV>(
        tmem_s, cta_coord_s, p_hi, p_lo, active_tokens,
        logical_row, row_max, row_sum);
    c2m.push(tid, q_slots[1] | q_slots[2] | q_slots[3]);

    auto frag_p_hi = pv_cta.make_fragment_A(p_hi);
    auto frag_p_lo = pv_cta.make_fragment_A(p_lo);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
#pragma unroll 1
    for (int tile = 0; tile < D_TILES; ++tile) {
        const int v_slots = m2c.template pop<0>();
        auto *v_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(v_slots)));
        auto sV = make_tensor(make_smem_ptr(v_ptr), v_layout);
        auto frag_v = pv_cta.make_fragment_B(sV);
        auto tmem_o = pv_cta.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + O_OFFSET;
        tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
        if (tid < numThreadsPerWarp) {
#pragma unroll 1
            for (int k_block = 0; k_block < size<2>(frag_p_hi); ++k_block) {
                gemm(tiled_pv, frag_p_hi(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
#pragma unroll 1
            for (int k_block = 0; k_block < size<2>(frag_p_lo); ++k_block) {
                gemm(tiled_pv, frag_p_lo(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, v_slots);

        const int output_slots = m2c.template pop<0>();
        auto *output_ptr = static_cast<data_t *>(
            get_slot_address(smem_base, extract(output_slots)));
        auto sO = make_tensor(
            make_smem_ptr(output_ptr),
            make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                        make_stride(Int<D_TILE>{}, Int<1>{})));
        auto cta_sO = pv_cta.partition_C(sO);
        sm100_attention_store_tmem_rows<M, D_TILE, false>(
            tmem_o, cta_coord_o, cta_sO, inv_sum, M);
        c2m.template push<31, true, false>(tid, output_slots);
    }
    c2m.push(tid, q_slots[0]);

    const int metadata_slots = m2c.template pop<0>();
    auto *metadata = static_cast<float *>(
        get_slot_address(smem_base, extract(metadata_slots)));
    if ((tid & 16) == 0) {
        metadata[logical_row * 2] = row_max;
        metadata[logical_row * 2 + 1] = row_sum;
    }
    __sync_compute_group(128);
    c2m.template push<31, true, false>(tid, metadata_slots);
}

// B64 BF16 split producer used by the FlashMLA-style resident path.  Scores
// become one BF16 probability tile in TMEM, so the ordinary 128-thread
// compute warpgroup performs QK and PV without a dequant/transform warpgroup.
template <typename ScoreTensor, typename CoordTensor,
          typename ProbTensor, typename ProbCoordTensor>
__device__ __forceinline__ void sm100_dsv4_softmax_split64_tmem(
    ScoreTensor const& scores,
    CoordTensor const& coords,
    ProbTensor const& probabilities,
    ProbCoordTensor const& probability_coords,
    const int active_tokens,
    int& logical_row,
    float& row_max,
    float& row_sum) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    constexpr int M = 64;
    constexpr float kScoreScale =
        M_LOG2E * 0.04419417382415922f;
    const int tid = __compute_tid();

    if (tid < 2 * M) {
        using ScoreLoad = SM100_TMEM_LOAD_16dp32b16x;
        auto tiled_load = make_tmem_copy(ScoreLoad{}, scores);
        auto thread_load = tiled_load.get_slice(tid);
        auto thread_scores = thread_load.partition_S(scores);
        auto thread_coords = thread_load.partition_D(coords);
        auto values = make_tensor<accum_t>(shape(thread_coords));
        copy(tiled_load, thread_scores, values);
        // TMEM loads retire asynchronously with respect to the issuing
        // thread.  Drain the load before consuming its register fragment or
        // allowing a later UMMA/kernel teardown to reuse the same TMEM.
        cutlass::arch::fence_view_async_tmem_load();

        logical_row = int(get<0>(thread_coords(0)));
        row_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < size(values); ++i) {
            const int col = int(get<1>(thread_coords(i)));
            values(i) *= kScoreScale;
            if (col < active_tokens) {
                row_max = fmaxf(row_max, values(i));
            }
        }
        row_max = fmaxf(
            row_max, __shfl_xor_sync(0xFFFFFFFFU, row_max, 16));

        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < size(values); ++i) {
            const int col = int(get<1>(thread_coords(i)));
            values(i) = col < active_tokens
                ? exp2f(values(i) - row_max) : 0.0f;
            local_sum += values(i);
        }
        row_sum = local_sum +
            __shfl_xor_sync(0xFFFFFFFFU, local_sum, 16);

        using ProbStore = SM100_TMEM_STORE_16dp32b16x;
        auto tiled_store = make_tmem_copy(ProbStore{}, probabilities);
        auto thread_store = tiled_store.get_slice(tid);
        auto thread_probability_coords =
            thread_store.partition_S(probability_coords);
        auto thread_probabilities =
            thread_store.partition_D(probabilities);
        auto packed = make_tensor<uint32_t>(shape(thread_probability_coords));
        auto pairs = recast<cutlass::Array<data_t, 2>>(packed);
        cutlass::NumericArrayConverter<data_t, accum_t, 2> convert;
        const int lane_half = (tid >> 4) & 1;
#pragma unroll
        for (int i = 0; i < size(pairs); ++i) {
            const int chunk = i / 16;
            const int pair_in_quarter = i & 7;
            const int local_index =
                chunk * 32 + 16 * lane_half + 2 * pair_in_quarter;
            const int peer_index =
                chunk * 32 + 16 * (1 - lane_half) + 2 * pair_in_quarter;
            const bool use_peer = ((i >> 3) & 1) != lane_half;
            const accum_t peer_0 = __shfl_xor_sync(
                0xFFFFFFFFU, values(peer_index), 16);
            const accum_t peer_1 = __shfl_xor_sync(
                0xFFFFFFFFU, values(peer_index + 1), 16);
            cutlass::Array<accum_t, 2> pair;
            pair[0] = use_peer ? peer_0 : values(local_index);
            pair[1] = use_peer ? peer_1 : values(local_index + 1);
            pairs(i) = convert(pair);
        }
        copy(tiled_store, packed, thread_probabilities);
        cutlass::arch::fence_view_async_tmem_store();
    }
    // The probability tile is produced by four warps but consumed by the
    // single UMMA-issuing warp.  Ordinary bar.sync does not transfer the
    // tcgen05 proxy view by itself.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __sync_compute_group(128);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_attention_split64_umma_sm100(
    int sm_id,
    int active_tokens,
    int ring_port_mask,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t& tmem_mma_phase,
    uint32_t& internal_ring_full_phase_mask,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m,
    uint64_t *g_events) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr int M = 64;
    constexpr int KV = 64;
    constexpr int D_TILE = 128;
    constexpr int D_TILES = 4;
    constexpr uint32_t P_OFFSET = 0;
    constexpr uint32_t O_OFFSET = 128;

    using QKAtom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, KV,
        UMMA::Major::K, UMMA::Major::K>;
    using PVAtom = SM100_MMA_F16BF16_TS<
        data_t, data_t, accum_t, M, D_TILE,
        UMMA::Major::K, UMMA::Major::MN>;

    const int tid = __compute_tid();
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
#define DAE_DSV4_ATTN_EVENT(event_id)                                      \
    do {                                                                   \
        if (tid == 0) {                                                    \
            asm volatile("" ::: "memory");                               \
            const uint64_t dae_attn_timer =                                \
                cuda::ptx::get_sreg_globaltimer();                         \
            const uint64_t dae_attn_cycles = clock64();                    \
            reinterpret_cast<volatile uint64_t *>(g_events)[               \
                sm_id * numProfileEvents +                                 \
                detailProfileEventBase + (event_id)] =                    \
                (uint64_t(uint32_t(dae_attn_cycles)) << 32) |               \
                uint32_t(dae_attn_timer);                                  \
            asm volatile("" ::: "memory");                               \
        }                                                                  \
    } while (false)
#else
#define DAE_DSV4_ATTN_EVENT(event_id) do { } while (false)
    (void)sm_id;
    (void)g_events;
#endif
    DAE_DSV4_ATTN_EVENT(2);
    auto tiled_qk = make_tiled_mma(QKAtom{});
    auto qk_cta = tiled_qk.get_slice(0);
    auto tiled_pv = make_tiled_mma(PVAtom{});
    auto pv_cta = tiled_pv.get_slice(0);
    auto q_shape = partition_shape_A(
        tiled_qk, make_shape(Int<M>{}, Int<D_TILE>{}));
    auto k_shape = partition_shape_B(
        tiled_qk, make_shape(Int<KV>{}, Int<D_TILE>{}));
    auto v_shape = partition_shape_B(
        tiled_pv, make_shape(Int<D_TILE>{}, Int<KV>{}));
    auto p_shape = partition_shape_A(
        tiled_pv, make_shape(Int<M>{}, Int<KV>{}));
    auto q_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, q_shape);
    auto k_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, k_shape);
    auto v_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_MN_SW128_Atom<data_t>{}, v_shape);
    auto p_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, p_shape);

    auto logical_s = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<KV>{}),
                    make_stride(Int<KV>{}, Int<1>{})));
    auto coord_s = make_identity_tensor(make_shape(Int<M>{}, Int<KV>{}));
    auto cta_s = qk_cta.partition_C(logical_s);
    auto cta_coord_s = qk_cta.partition_C(coord_s);
    auto tmem_s = qk_cta.make_fragment_C(cta_s);
    tmem_s.data() = tmem_base_ptr;
    auto tmem_p_view = tmem_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));
    tmem_p_view.data() = tmem_base_ptr + P_OFFSET;
    auto coord_p_view = cta_coord_s.compose(
        make_layout(make_shape(Int<M>{}, Int<KV / 2>{})));
    auto dummy_p = make_tensor(
        make_smem_ptr(static_cast<data_t *>(nullptr)), p_layout);
    auto tmem_p = pv_cta.make_fragment_A(dummy_p);
    tmem_p.data() = tmem_base_ptr + P_OFFSET;

    auto logical_o = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                    make_stride(Int<D_TILE>{}, Int<1>{})));
    auto coord_o = make_identity_tensor(make_shape(Int<M>{}, Int<D_TILE>{}));
    auto cta_o = pv_cta.partition_C(logical_o);
    auto cta_coord_o = pv_cta.partition_C(coord_o);

    const int ring_slots = m2c.template pop<0>();
    DAE_DSV4_ATTN_EVENT(0);
    auto *ring = static_cast<data_t *>(
        get_slot_address(smem_base, extract(ring_slots)));
    const int q_slots = m2c.template pop<0>();
    DAE_DSV4_ATTN_EVENT(1);
    auto *q = static_cast<data_t *>(
        get_slot_address(smem_base, extract(q_slots)));

    auto *ring_full = reinterpret_cast<TxBarrier *>(
        tmem_mma_barrier + internalRingFullBarrierBase);
    auto *ring_empty = reinterpret_cast<TxBarrier *>(
        tmem_mma_barrier + internalRingEmptyBarrierBase);
    for (int port = 0; port < 2; ++port) {
        if (ring_port_mask & (1 << port)) {
            constexpr int kStage = 0;
            const int phase_index = port * internalRingStages + kStage;
            const uint32_t phase =
                (internal_ring_full_phase_mask >> phase_index) & 1U;
            if (tid < numThreadsPerWarp) {
                ring_full[phase_index].wait(phase);
            }
            internal_ring_full_phase_mask ^= 1U << phase_index;
            DAE_DSV4_ATTN_EVENT(38 + port);
        }
    }
    DAE_DSV4_ATTN_EVENT(3);

    // A reusable position image is sized for its largest decode context.
    // Splits beyond the current candidate count still consume their ordinary
    // LDU/STU messages, but publish the exact softmax identity partial.  This
    // preserves the queue and reducer contracts without issuing useless UMMA.
    if (active_tokens == 0) {
        c2m.push(tid, q_slots);
        const int metadata_slots = m2c.template pop<0>();
        const ComputeRawAddressSlots raw_slots{st_insts};
        auto *metadata = raw_slots.template get<float>(metadata_slots);
        if (tid < M) {
            cuda::atomic_ref<float, cuda::thread_scope_device>(
                metadata[tid * 2]
            ).store(-FLT_MAX, cuda::memory_order_release);
            cuda::atomic_ref<float, cuda::thread_scope_device>(
                metadata[tid * 2 + 1]
            ).store(0.0f, cuda::memory_order_release);
        }
#pragma unroll
        for (int tile = 0; tile < D_TILES; ++tile) {
            if (tile + 1 == D_TILES) {
                if (tid == 0) {
                    ring_empty[0].arrive();
                }
                c2m.push(tid, ring_slots);
            }
            const int output_slots = m2c.template pop<0>();
            auto *output = static_cast<data_t *>(
                get_slot_address(smem_base, extract(output_slots)));
            for (int item = tid; item < M * D_TILE; item += 128) {
                output[item] = data_t(0.0f);
            }
            __sync_compute_group(128);
            cutlass::arch::fence_view_async_shared();
            c2m.template push<31, true, false>(tid, output_slots);
        }
        return;
    }

    tiled_qk.accumulate_ = UMMA::ScaleOut::Zero;
    if (tid < numThreadsPerWarp) {
        // QK remains one four-tile loop so the full inference image does not
        // clone the complete CuTe/UMMA issue body four times.
#pragma unroll 1
        for (int tile = 0; tile < D_TILES; ++tile) {
            auto sQ = make_tensor(
                make_smem_ptr(q + tile * M * D_TILE), q_layout);
            auto sK = make_tensor(
                make_smem_ptr(ring + tile * KV * D_TILE), k_layout);
            auto frag_q = qk_cta.make_fragment_A(sQ);
            auto frag_k = qk_cta.make_fragment_B(sK);
#pragma unroll
            for (int k_block = 0; k_block < size<2>(frag_q); ++k_block) {
                gemm(tiled_qk, frag_q(_, _, k_block),
                     frag_k(_, _, k_block), tmem_s);
                tiled_qk.accumulate_ = UMMA::ScaleOut::One;
            }
        }
        cutlass::arch::umma_arrive(tmem_mma_barrier);
    }

    cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
    tmem_mma_phase ^= 1;
    // tcgen05.commit (umma_arrive) supplies the producer-side ordering.  A
    // consumer-side fence is still required before the other compute warps
    // may load the completed score tile from TMEM.
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    c2m.push(tid, q_slots);
    DAE_DSV4_ATTN_EVENT(4);

    // Metadata is a small raw vector.  Consume its dedicated mailbox before
    // the four allocator-owned partial-tile destinations.
    const int metadata_slots = m2c.template pop<0>();
    const ComputeRawAddressSlots raw_slots{st_insts};
    auto *metadata = raw_slots.template get<float>(metadata_slots);
    DAE_DSV4_ATTN_EVENT(5);

    int logical_row = 0;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    sm100_dsv4_softmax_split64_tmem(
        tmem_s, cta_coord_s, tmem_p_view, coord_p_view,
        active_tokens, logical_row, row_max, row_sum);
    DAE_DSV4_ATTN_EVENT(6);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    if ((tid & 16) == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device>(
            metadata[logical_row * 2]
        ).store(row_max, cuda::memory_order_release);
        cuda::atomic_ref<float, cuda::thread_scope_device>(
            metadata[logical_row * 2 + 1]
        ).store(row_sum, cuda::memory_order_release);
    }
    // Metadata rows are disjoint raw global stores.  Do not stop PV issue on
    // them: the first PV materialization joins all four compute warps before
    // either output group (and therefore either reducer) can be published.
    DAE_DSV4_ATTN_EVENT(20);

#pragma unroll 1
    for (int tile = 0; tile < D_TILES; ++tile) {
        // Publish the second output group first.  Its reducer owns the final
        // D128 tile and therefore also performs inverse RoPE; starting that
        // longer group while group 0 is still being materialized removes it
        // from the producer/reducer tail.
        const int output_tile = tile ^ 2;
        auto sV = make_tensor(
            make_smem_ptr(
                ring + KV * D_TILE * D_TILES +
                    output_tile * KV * D_TILE),
            v_layout);
        auto frag_v = pv_cta.make_fragment_B(sV);
        auto tmem_o = pv_cta.make_fragment_C(cta_o);
        tmem_o.data() = tmem_base_ptr + O_OFFSET;
        tiled_pv.accumulate_ = UMMA::ScaleOut::Zero;
        if (tid < numThreadsPerWarp) {
#pragma unroll
            for (int k_block = 0; k_block < size<2>(tmem_p); ++k_block) {
                gemm(tiled_pv, tmem_p(_, _, k_block),
                     frag_v(_, _, k_block), tmem_o);
                tiled_pv.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        DAE_DSV4_ATTN_EVENT(7 + 3 * tile);
        if (tile + 1 == D_TILES) {
            if (tid == 0) {
                ring_empty[0].arrive();
            }
            c2m.push(tid, ring_slots);
        }

        const int output_slots = m2c.template pop<0>();
        DAE_DSV4_ATTN_EVENT(30 + tile);
        auto *output = static_cast<data_t *>(
            get_slot_address(smem_base, extract(output_slots)));
        auto sO = make_tensor(
            make_smem_ptr(output),
            make_layout(make_shape(Int<M>{}, Int<D_TILE>{}),
                        make_stride(Int<D_TILE>{}, Int<1>{})));
        if (tid < 2 * M) {
            using OutputLoad = SM100_TMEM_LOAD_16dp32b16x;
            auto tiled_load = make_tmem_copy(OutputLoad{}, tmem_o);
            auto thread_load = tiled_load.get_slice(tid);
            auto thread_tmem = thread_load.partition_S(tmem_o);
            auto cta_output = pv_cta.partition_C(sO);
            auto thread_output = thread_load.partition_D(cta_output);
            auto values = make_tensor<accum_t>(shape(thread_output));
            copy(tiled_load, thread_tmem, values);
            cutlass::arch::fence_view_async_tmem_load();
            DAE_DSV4_ATTN_EVENT(34 + tile);
            auto converted = make_tensor<data_t>(shape(thread_output));
            static_assert(size(values) % 2 == 0,
                          "TMEM output fragments must form BF16x2 packs");
            auto converted_pairs =
                recast<cutlass::Array<data_t, 2>>(converted);
            cutlass::NumericArrayConverter<data_t, accum_t, 2> convert;
#pragma unroll
            for (int index = 0; index < size(converted_pairs); ++index) {
                cutlass::Array<accum_t, 2> pair;
                pair[0] = values(2 * index) * inv_sum;
                pair[1] = values(2 * index + 1) * inv_sum;
                converted_pairs(index) = convert(pair);
            }
            copy(converted, thread_output);
        }
        DAE_DSV4_ATTN_EVENT(8 + 3 * tile);
        // All four warps drain the same output tile.  Wait for every
        // tcgen05.ld before warp 0 reuses O_OFFSET for the next PV UMMA.
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __sync_compute_group(128);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        cutlass::arch::fence_view_async_shared();
        c2m.template push<31, true, false>(tid, output_slots);
        DAE_DSV4_ATTN_EVENT(9 + 3 * tile);
    }
    DAE_DSV4_ATTN_EVENT(19);
    DAE_DSV4_ATTN_EVENT(21);
#undef DAE_DSV4_ATTN_EVENT
}

template <int ScalePack, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_attention_context1_fp8_sm100(
    int head,
    int output_group,
    void *smem_base,
    void *task_scratch,
    const MInst *st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m) {
    using namespace cute;
    using Fp8 = cutlass::float_e4m3_t;
    using Scale = cutlass::float_ue8m0_t;
    using Accum = float;
    constexpr int kHeadDim = 512;
    constexpr int kTileM = 128;
    constexpr int kTileN = 8;
    constexpr int kTileK = 128;
    constexpr int kTotalTiles = 4;
    constexpr int kTiles = kTotalTiles;
    constexpr int kScaleVector = 32;
    constexpr int kBBytes = kTileN * kTileK;
    constexpr int kBTileBytes = 2048;
    constexpr float kScoreScale = M_LOG2E / 22.627416997969522f;
    static_assert(ScalePack == 2);
    const bool normalize_q = (output_group & 4) != 0;

    const int tid = __compute_tid();
    const int lane = tid & 31;
    const int warp = tid >> 5;
    auto *shared = static_cast<float *>(task_scratch);
    const ComputeRawAddressSlots raw_slots{st_insts};

    // The barriered raw record is the sole dynamic producer dependency. Q and
    // the current KV row live at stable addresses for the resident launch;
    // the layer-specific sink is the following tiny LDU copy.
    const int record_slot = m2c.template pop<0>();
    const auto *record = raw_slots.template get<const uint64_t>(record_slot);
    const auto *q = reinterpret_cast<const __nv_bfloat16 *>(record[0]) +
        head * kHeadDim;
    const auto *kv = reinterpret_cast<const __nv_bfloat16 *>(record[1]);
    const auto *table = reinterpret_cast<const float *>(record[2]);
    const int sink_slot = m2c.template pop<0>();
    const auto *sink = static_cast<const float *>(
        get_slot_address(smem_base, extract(sink_slot)));

    // One row needs one Q.K score. In the direct-BF16 split-K path, fold the
    // otherwise separate Q RMS/RoPE stage into this task and round to BF16 at
    // the same semantic boundary before taking the dot product.
    float q_values[kTotalTiles];
#pragma unroll
    for (int tile = 0; tile < kTotalTiles; ++tile) {
        q_values[tile] = __bfloat162float(q[tile * kTileK + tid]);
    }
    if (normalize_q) {
        float square_sum = 0.0f;
#pragma unroll
        for (int tile = 0; tile < kTotalTiles; ++tile) {
            square_sum = fmaf(q_values[tile], q_values[tile], square_sum);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            square_sum += __shfl_down_sync(
                0xFFFFFFFFU, square_sum, offset);
        }
        if (lane == 0) {
            shared[warp] = square_sum;
        }
        __sync_compute_group(128);
        if (tid == 0) {
            shared[33] = rsqrtf(
                (shared[0] + shared[1] + shared[2] + shared[3]) /
                    float(kHeadDim) +
                0x1.0cp-20f);
        }
        __sync_compute_group(128);
#pragma unroll
        for (int tile = 0; tile < kTotalTiles; ++tile) {
            q_values[tile] *= shared[33];
        }
        if (tid >= 64) {
            const int pair = (tid - 64) >> 1;
            const float partner = __shfl_xor_sync(
                0xFFFFFFFFU, q_values[kTotalTiles - 1], 1);
            const float cosine = table[pair * 2];
            const float sine = table[pair * 2 + 1];
            q_values[kTotalTiles - 1] = (tid & 1)
                ? partner * sine + q_values[kTotalTiles - 1] * cosine
                : q_values[kTotalTiles - 1] * cosine - partner * sine;
        }
#pragma unroll
        for (int tile = 0; tile < kTotalTiles; ++tile) {
            q_values[tile] = __bfloat162float(
                __float2bfloat16(q_values[tile]));
        }
    }

    // All four compute warps cover K512 and one head task emits all four
    // output tiles, so neither Q.K nor the LDU messages are duplicated.
    float dot = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kTotalTiles; ++tile) {
        const int index = tile * kTileK + tid;
        dot = fmaf(
            q_values[tile],
            __bfloat162float(kv[index]),
            dot);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
    }
    if (lane == 0) {
        shared[warp] = dot;
    }
    __sync_compute_group(128);
    if (tid == 0) {
        const float score_log2 =
            (shared[0] + shared[1] + shared[2] + shared[3]) * kScoreScale;
        const float sink_log2 = sink[head & 3] * M_LOG2E;
        const float maximum = fmaxf(score_log2, sink_log2);
        const float numerator = exp2f(score_log2 - maximum);
        shared[32] = numerator /
            (numerator + exp2f(sink_log2 - maximum));
    }
    __sync_compute_group(128);
    c2m.push(tid, sink_slot);

    float values[kTiles];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        values[tile] = shared[32] * __bfloat162float(
            kv[tile * kTileK + tid]);
    }
    if (tid >= 64) {
        const int pair = (tid - 64) >> 1;
        const float partner =
            __shfl_xor_sync(0xFFFFFFFFU, values[kTiles - 1], 1);
        const float cosine = table[pair * 2];
        const float sine = table[pair * 2 + 1];
        values[kTiles - 1] = (tid & 1)
            ? values[kTiles - 1] * cosine - partner * sine
            : values[kTiles - 1] * cosine + partner * sine;
    }

    const int output_slot = m2c.template pop<0>();
    auto *output = raw_slots.template get<uint8_t>(output_slot) +
        head * kTotalTiles * kBTileBytes;

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        float maximum = fabsf(values[tile]);
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
        }
        if (lane == 0) {
            shared[tile * 4 + warp] = maximum;
        }
    }
    __sync_compute_group(128);
    if (tid < kTiles) {
        const int tile = tid;
        const float maximum = fmaxf(
            fmaxf(shared[tile * 4], shared[tile * 4 + 1]),
            fmaxf(shared[tile * 4 + 2], shared[tile * 4 + 3]));
        const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
        const uint32_t requested_bits = __float_as_uint(requested);
        const uint32_t exponent_bits = (requested_bits >> 23) & 0xFFU;
        const uint32_t fraction_bits = requested_bits & 0x7FFFFFU;
        int exponent;
        if (exponent_bits == 0) {
            const int highest_bit = 31 - __clz(fraction_bits);
            const int floor_exponent = highest_bit - 149;
            exponent = floor_exponent +
                ((fraction_bits & (fraction_bits - 1)) != 0);
        } else if (exponent_bits == 0xFFU) {
            exponent = 127;
        } else {
            exponent = int(exponent_bits) - 127 +
                (fraction_bits != 0);
        }
        exponent = max(-127, min(127, exponent));
        const uint32_t scale_bits = exponent >= -126
            ? uint32_t(exponent + 127) << 23
            : 1U << 22;
        shared[16 + tile] = __uint_as_float(scale_bits);
    }
    __sync_compute_group(128);

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        auto *tile_output = output + tile * kBTileBytes;
        const Fp8 quantized = values[tile] == 0.0f
            ? Fp8(0.0f)
            : Fp8(fminf(
                fmaxf(values[tile] / shared[16 + tile], -448.0f),
                448.0f));
        const int source_chunk = tid / 16;
        const int byte_in_chunk = tid % 16;
#pragma unroll
        for (int row = 0; row < kTileN; ++row) {
            const int destination_chunk = source_chunk ^ row;
            reinterpret_cast<Fp8 *>(tile_output)[
                row * kTileK + destination_chunk * 16 + byte_in_chunk] =
                quantized;
        }
    }

    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
    using Atom = SM100_MMA_MXF8F6F4_SS<
        Fp8, Fp8, Accum, Scale, kTileM, kTileN,
        UMMA::Major::K, UMMA::Major::K>;
    using TiledMma = decltype(make_tiled_mma(Atom{}));
    using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;
    using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
    const auto logical_sfb = ScaleConfig::tile_atom_to_shape_SFB(
        ScaleProblemShape{});
    if (tid < kTileN * ScalePack) {
        const int row = tid / ScalePack;
        const int sf = tid % ScalePack;
        const int dst = int(logical_sfb(row, sf * kScaleVector));
#pragma unroll
        for (int group = 0; group < kTotalTiles / ScalePack; ++group) {
            auto *packed_scale = reinterpret_cast<Scale *>(
                output + group * ScalePack * kBTileBytes + kBBytes);
            packed_scale[dst] = Scale(
                shared[16 + group * ScalePack + sf]);
        }
    }
    c2m.template push<31, true, false>(tid, 1U << output_slot);
}

template <int ScalePack, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_attention_split_reduce_fp8_sm100(
    int sm_id,
    int num_splits,
    int head,
    int output_group,
    void *task_scratch,
    const MInst *st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m,
    uint64_t *g_events) {
    using namespace cute;
    using Fp8 = cutlass::float_e4m3_t;
    using Scale = cutlass::float_ue8m0_t;
    using Accum = float;
    constexpr int kHeads = 64;
    constexpr int kTileM = 128;
    constexpr int kTileN = 8;
    constexpr int kTileK = 128;
    constexpr int kTotalTiles = 4;
    constexpr int kTiles = 2;
    constexpr int kScaleVector = 32;
    constexpr int kBBytes = kTileN * kTileK;
    constexpr int kBTileBytes = 2048;
    const int tid = __compute_tid();
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
#define DAE_DSV4_REDUCE_EVENT(event_id)                                    \
    do {                                                                   \
        if (tid == 0) {                                                    \
            g_events[sm_id * numProfileEvents +                           \
                     detailProfileEventBase + (event_id)] =               \
                cuda::ptx::get_sreg_globaltimer();                         \
        }                                                                  \
    } while (false)
#else
#define DAE_DSV4_REDUCE_EVENT(event_id) do { } while (false)
    (void)sm_id;
    (void)g_events;
#endif
    DAE_DSV4_REDUCE_EVENT(22);
    static_assert(ScalePack == 2);
    static_assert(kTiles == ScalePack);
    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
    using Atom = SM100_MMA_MXF8F6F4_SS<
        Fp8, Fp8, Accum, Scale, kTileM, kTileN,
        UMMA::Major::K, UMMA::Major::K>;
    using TiledMma = decltype(make_tiled_mma(Atom{}));
    using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;
    using LayoutSFB = decltype(
        ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
    using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
    const auto logical_sfb =
        ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});

    // The barriered record is the reducer's sole input message.  Its four
    // pointers name [partials, metadata, sink, inverse-RoPE table]; LDU owns
    // the producer dependency and publishes this message only after every
    // producer STU has completed its partial stores. Max/sum metadata bypasses
    // STU, so its producer uses device-release stores and the loads below use
    // device-acquire semantics instead of borrowing STU's release sequence.
    const int record_slot = m2c.template pop<0>();
    DAE_DSV4_REDUCE_EVENT(23);
    const ComputeRawAddressSlots raw_slots{st_insts};
    const auto *record = raw_slots.template get<const uint64_t>(record_slot);
    const auto *partials = reinterpret_cast<const __nv_bfloat16 *>(record[0]);
    const auto *metadata = reinterpret_cast<const float *>(record[1]);
    const auto *sink = reinterpret_cast<const float *>(record[2]);
    const auto *table = reinterpret_cast<const float *>(record[3]);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int first_tile = output_group * kTiles;
    auto *shared = static_cast<float *>(task_scratch);


    if (warp == 0) {
        const float sink_log2 = sink[head] * M_LOG2E;
        float local_max = -FLT_MAX;
        float local_mass = 0.0f;
        if (lane < num_splits) {
            local_max = cuda::atomic_ref<
                float, cuda::thread_scope_device
            >(*const_cast<float *>(
                metadata + (lane * kHeads + head) * 2
            )).load(cuda::memory_order_acquire);
            local_mass = cuda::atomic_ref<
                float, cuda::thread_scope_device
            >(*const_cast<float *>(
                metadata + (lane * kHeads + head) * 2 + 1
            )).load(cuda::memory_order_acquire);
        }
        if (num_splits == 2) {
            const float split_max = lane < 2 ? local_max : -FLT_MAX;
            const float max_candidate = lane == 0
                ? fmaxf(split_max, sink_log2) : split_max;
            const float global_max = fmaxf(
                max_candidate,
                __shfl_xor_sync(0xFFFFFFFFU, max_candidate, 1));
            float split_contribution = 0.0f;
            if (lane < 2) {
                split_contribution =
                    exp2f(split_max - global_max) * local_mass;
            }
            float denominator_part = split_contribution;
            if (lane == 0) {
                denominator_part += exp2f(sink_log2 - global_max);
            }
            const float denominator = denominator_part +
                __shfl_xor_sync(
                    0xFFFFFFFFU, denominator_part, 1);
            if (lane < 2) {
                shared[32 + lane] =
                    split_contribution / denominator;
            }
        } else {
            float global_max = lane == 0 ? sink_log2 : -FLT_MAX;
            if (lane < num_splits) {
                global_max = fmaxf(global_max, local_max);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                global_max = fmaxf(
                    global_max,
                    __shfl_down_sync(0xFFFFFFFFU, global_max, offset));
            }
            global_max = __shfl_sync(0xFFFFFFFFU, global_max, 0);

            float split_contribution = 0.0f;
            if (lane < num_splits) {
                split_contribution =
                    exp2f(local_max - global_max) * local_mass;
            }
            float contribution = split_contribution;
            if (lane == 0) {
                contribution += exp2f(sink_log2 - global_max);
            }
            float denominator = contribution;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                denominator += __shfl_down_sync(
                    0xFFFFFFFFU, denominator, offset);
            }
            const float inv_denominator = 1.0f /
                __shfl_sync(0xFFFFFFFFU, denominator, 0);
            if (lane < num_splits) {
                shared[32 + lane] =
                    split_contribution * inv_denominator;
            }
        }
    }
    __sync_compute_group(128);
    DAE_DSV4_REDUCE_EVENT(24);

    float values[kTiles] = {};
    if (num_splits == 2) {
        const auto *partial_0 = partials +
            head * kTotalTiles * kTileK;
        const auto *partial_1 = partial_0 +
            kHeads * kTotalTiles * kTileK;
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
            const int offset = (first_tile + tile) * kTileK + tid;
            values[tile] =
                shared[32] * __bfloat162float(partial_0[offset]) +
                shared[33] * __bfloat162float(partial_1[offset]);
        }
    } else {
#pragma unroll 1
        for (int split = 0; split < num_splits; ++split) {
            const auto *partial = partials +
                (split * kHeads + head) * kTotalTiles * kTileK;
            const float weight = shared[32 + split];
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                values[tile] +=
                    weight * __bfloat162float(
                        partial[(first_tile + tile) * kTileK + tid]);
            }
        }
    }

    if (output_group == 1 && tid >= 64) {
        const int pair = (tid - 64) >> 1;
        const float partner =
            __shfl_xor_sync(0xFFFFFFFFU, values[kTiles - 1], 1);
        const float cosine = table[pair * 2];
        const float sine = table[pair * 2 + 1];
        values[kTiles - 1] = (tid & 1)
            ? values[kTiles - 1] * cosine - partner * sine
            : values[kTiles - 1] * cosine + partner * sine;
    }
    DAE_DSV4_REDUCE_EVENT(25);

    const int output_slot = m2c.template pop<0>();
    DAE_DSV4_REDUCE_EVENT(26);
    auto *output = raw_slots.template get<uint8_t>(output_slot) +
        (head * kTotalTiles + first_tile) * kBTileBytes;

    // Reduce both local tile maxima together.  Keeping the merge coefficients
    // at shared[32..] leaves shared[0..19] available for this phase without
    // a producer/consumer reuse join.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        float maximum = fabsf(values[tile]);
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
        }
        if (lane == 0) {
            shared[tile * 4 + warp] = maximum;
        }
    }
    __sync_compute_group(128);
    DAE_DSV4_REDUCE_EVENT(27);
    if (tid < kTiles) {
        const int tile = tid;
        const float maximum = fmaxf(
            fmaxf(shared[tile * 4], shared[tile * 4 + 1]),
            fmaxf(shared[tile * 4 + 2], shared[tile * 4 + 3]));
        const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
        const uint32_t requested_bits = __float_as_uint(requested);
        const uint32_t exponent_bits = (requested_bits >> 23) & 0xFFU;
        const uint32_t fraction_bits = requested_bits & 0x7FFFFFU;
        int exponent;
        if (exponent_bits == 0) {
            const int highest_bit = 31 - __clz(fraction_bits);
            const int floor_exponent = highest_bit - 149;
            exponent = floor_exponent +
                ((fraction_bits & (fraction_bits - 1)) != 0);
        } else if (exponent_bits == 0xFFU) {
            exponent = 127;
        } else {
            exponent = int(exponent_bits) - 127 +
                (fraction_bits != 0);
        }
        exponent = max(-127, min(127, exponent));
        const uint32_t scale_bits = exponent >= -126
            ? uint32_t(exponent + 127) << 23
            : 1U << 22;
        shared[16 + tile] = __uint_as_float(scale_bits);
    }
    __sync_compute_group(128);
    DAE_DSV4_REDUCE_EVENT(28);

    // Emit all data tiles after the shared scales are ready.  Padding bytes
    // in each native record are deliberately unspecified and are not stored.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        auto *tile_output = output + tile * kBTileBytes;
        const Fp8 quantized = values[tile] == 0.0f
            ? Fp8(0.0f)
            : Fp8(fminf(
                fmaxf(values[tile] / shared[16 + tile], -448.0f),
                448.0f));
        const int source_chunk = tid / 16;
        const int byte_in_chunk = tid % 16;
#pragma unroll
        for (int row = 0; row < kTileN; ++row) {
            const int destination_chunk = source_chunk ^ row;
            reinterpret_cast<Fp8 *>(tile_output)[
                row * kTileK + destination_chunk * 16 + byte_in_chunk] =
                quantized;
        }
    }
    if (tid < kTileN * ScalePack) {
        const int row = tid / ScalePack;
        const int sf = tid % ScalePack;
        auto *packed_scale = reinterpret_cast<Scale *>(
            output + kBBytes);
        const int dst = int(logical_sfb(row, sf * kScaleVector));
        packed_scale[dst] = Scale(shared[16 + sf]);
    }
    // Raw-address M2C messages carry a slot index; STU consumes a one-hot
    // completion token.  The C2M barrier also joins all 128 direct stores
    // before STU releases any downstream dependency.
    c2m.template push<31, true, false>(tid, 1U << output_slot);
    DAE_DSV4_REDUCE_EVENT(29);
#undef DAE_DSV4_REDUCE_EVENT
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
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128,
                  "The SM100 decode path supports head_dim=64 or 128");
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
                get_slot_address(base, attentionScratchSlot));
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
                    get_slot_address(base, attentionScratchSlot));
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
    constexpr int kScoreTmemColumnBits =
        ACTIVE_Q * sizeof_bits_v<accum_t>;
    constexpr int kOutputTmemColumnBits =
        ACTIVE_Q * sizeof_bits_v<accum_t>;
    static_assert(kScoreTmemColumnBits == 128);
    static_assert(kOutputTmemColumnBits == 128);

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
        get_slot_address(base, attentionScratchSlot));
#if defined(DAE_PACKED_SWAP_ATTENTION_SCRATCH)
    // QK scores and P have disjoint lifetimes.  The score transpose is first
    // loaded completely into registers, then the same 2-KiB region is reused
    // for the BF16 probability tile.
    accum_t *score_stage = reinterpret_cast<accum_t *>(prob_ptr);
#else
    accum_t *score_stage = reinterpret_cast<accum_t *>(
        prob_ptr + cosize(layout_s));
#endif
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
    auto cta_o = cta_pv.partition_C(logical_o);
    auto tmem_o = cta_pv.make_fragment_C(cta_o);
    tmem_o.data() = tmem_base_ptr + O_OFFSET;

    auto score_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x,
                          kScoreTmemColumnBits>();
    const uint32_t score_tmem_addr = raw_pointer_cast(tmem_s.data());
    auto r_scores = make_tensor<accum_t>(Int<ACTIVE_Q>{});

    auto output_load_op =
        TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x,
                          kOutputTmemColumnBits>();
    auto output_store_op =
        TMEM::op_repeater<SM100_TMEM_STORE_32dp32b1x,
                          kOutputTmemColumnBits>();
    const uint32_t output_tmem_addr = raw_pointer_cast(tmem_o.data());
    auto r_output = make_tensor<accum_t>(Int<ACTIVE_Q>{});

    const int q_slot = m2c.template pop<0>();
    data_t *q_ptr = static_cast<data_t *>(
        get_slot_address(base, extract(q_slot)));
    auto sQ = make_tensor(make_smem_ptr(q_ptr), layout_q);
    auto frag_q = cta_qk.make_fragment_B(sQ);

    accum_t row_max = -FLT_MAX;
    accum_t row_sum = 0.0f;

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

        sm100_attention_tmem_load_raw<decltype(score_load_op)>(
            score_tmem_addr, r_scores);
        cutlass::arch::fence_view_async_tmem_load();
        const bool token_valid =
            block + 1 < num_kv_blocks || tid < last_kv_active_token_len;
        #pragma unroll
        for (int q = 0; q < ACTIVE_Q; ++q) {
            score_stage[q * KV + tid] = token_valid && q < num_active_q
                ? r_scores(q) * kScoreScale : -FLT_MAX;
        }
        __sync_compute_group(128);
        c2m.push(tid, k_slot);

        // Transpose the softmax work in shared memory: one warp owns one live
        // query column and each lane consumes four sequence positions.  This
        // replaces four CTA-wide reductions with one warp reduction per query.
        accum_t scores[KV / numThreadsPerWarp];
        accum_t block_max = -FLT_MAX;
        #pragma unroll
        for (int i = 0; i < KV / numThreadsPerWarp; ++i) {
            const int token = lane_id + i * numThreadsPerWarp;
            scores[i] = score_stage[warp_id * KV + token];
            block_max = fmaxf(block_max, scores[i]);
        }
#if defined(DAE_PACKED_SWAP_ATTENTION_SCRATCH)
        __sync_compute_group(128);
#endif
        accum_t warp_max;
        asm volatile(
            "redux.sync.max.NaN.f32 %0, %1, 0xffffffff;\n"
            : "=f"(warp_max) : "f"(block_max));
        const bool query_valid =
            warp_id < num_active_q && warp_max != -FLT_MAX;
        const accum_t new_max = query_valid
            ? fmaxf(row_max, warp_max) : -FLT_MAX;
        const accum_t correction = row_max == -FLT_MAX
            ? 0.0f : exp2f(row_max - new_max);
        row_max = new_max;

        accum_t block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < KV / numThreadsPerWarp; ++i) {
            scores[i] = query_valid
                ? exp2f(scores[i] - new_max) : 0.0f;
            block_sum += scores[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_xor_sync(
                0xFFFFFFFFU, block_sum, offset);
        }
        row_sum = row_sum * correction + block_sum;

        // The live and padded P columns are disjoint across warps.  Keep the
        // padded UMMA columns zero while publishing the four live columns.
        #pragma unroll
        for (int i = 0; i < KV / numThreadsPerWarp; ++i) {
            const int token = lane_id + i * numThreadsPerWarp;
            sS(token, warp_id) = data_t(scores[i]);
            sS(token, warp_id + ACTIVE_Q) = data_t(0.0f);
        }
        if (lane_id == 0) {
            smem_reduce[warp_id] = row_max;
            smem_reduce[ACTIVE_Q + warp_id] = row_sum;
            smem_reduce[2 * ACTIVE_Q + warp_id] = correction;
        }

        cutlass::arch::fence_view_async_shared();
        __sync_compute_group(128);

        if (block > 0) {
            sm100_attention_tmem_load_raw<decltype(output_load_op)>(
                output_tmem_addr, r_output);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for (int q = 0; q < ACTIVE_Q; ++q) {
                r_output(q) *= smem_reduce[2 * ACTIVE_Q + q];
            }
            sm100_attention_tmem_store_raw<decltype(output_store_op)>(
                r_output, output_tmem_addr);
            cutlass::arch::fence_view_async_tmem_store();
            __sync_compute_group(128);
        }

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
            sm100_attention_tmem_load_raw<decltype(output_load_op)>(
                output_tmem_addr, r_output);
            cutlass::arch::fence_view_async_tmem_load();
            data_t *o_ptr = static_cast<data_t *>(
                slot_2_glob_ptr(st_insts, o_slot));
            #pragma unroll
            for (int q = 0; q < ACTIVE_Q; ++q) {
                if (q < num_active_q) {
                    accum_t output_scale = 1.0f;
                    if constexpr (!SPLIT_KV) {
                        const accum_t total_mass =
                            smem_reduce[ACTIVE_Q + q];
                        output_scale = total_mass > 0.0f
                            ? 1.0f / total_mass : 0.0f;
                    }
                    o_ptr[q * D + tid] =
                        data_t(r_output(q) * output_scale);
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
        if (tid < num_active_q &&
            smem_reduce[ACTIVE_Q + tid] > 0.0f) {
            lse_ptr[(tid * 2 + 0) * MAX_SPLIT + split_idx] =
                smem_reduce[tid];
            lse_ptr[(tid * 2 + 1) * MAX_SPLIT + split_idx] =
                smem_reduce[ACTIVE_Q + tid];
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
