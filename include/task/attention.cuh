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
            acc_fragS(c, r) = (accum_t)expf(acc_fragS(c, r) - row_max(r));
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
            accum_t score_scaler = (accum_t)expf(row_max_prev(r) - row_max(r));
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

    // load Qtile
    int slot_Q = m2c.template pop<0>();
    data_t* sQ_ptr = (data_t*)get_slot_address(base, extract(slot_Q));
    auto sQ = make_tensor(make_smem_ptr(sQ_ptr), layout_sQ);
    auto frag_Q = thr_mma_qk.partition_fragment_A(sQ);

    // TODO(zhiyuang): implement scale here for now. could be fused with gemv?
    // TODO(zhiyuang): vecorized scale?
    const data_t scale_fp16 = static_cast<data_t>(1.0f / sqrtf((float)HEAD_DIM));
    {
        auto sQ_part = thr_mma_qk.partition_A(sQ);
        #pragma unroll
        for (int i = 0; i < size(sQ_part); ++i) {
            sQ_part(i) = sQ_part(i) * scale_fp16;
        }
    }
    __sync_compute_group(128);

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
                const accum_t lse = online_softmax.row_max(r) + logf(online_softmax.row_sum(r));
                sLSE_ptr[q_row * MAX_SPLIT + split_idx] = lse;
            }
        }
        __sync_compute_group(128);
        c2m.template push<31, true, false>(thread_id, 1 << slot_lse);
    }
}

template <int HEAD_DIM,
          int NUM_Q,
          int KV_BLOCK_SIZE,
          int MAX_SPLIT,
          int THREADS_PER_Q,   // tuning knob: threads assigned to each Q row
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
    const int my_q      = thread_id / THREADS_PER_Q;
    const int my_i_base = (thread_id % THREADS_PER_Q) * ELEMS_PER_THREAD;

    auto layout_sO = make_layout(
        make_shape(Int<NUM_Q>{}, Int<HEAD_DIM/2>{}),
        LayoutRight{});
    auto layout_split_O = make_layout(
        make_shape(num_split, Int<NUM_Q>{}, Int<HEAD_DIM/2>{}),
        LayoutRight{});
    auto layout_lse = make_layout(
        make_shape(Int<NUM_Q>{}, Int<MAX_SPLIT>{}),
        LayoutRight{});

    // LSE lives in global memory — available immediately (raw address, no TMA wait).
    const int slot_lse = m2c.template pop<0>();
    const accum_t* __restrict__ sLSE_ptr = (accum_t*)slot_2_glob_ptr(st_insts, slot_lse);
    auto gLSE = make_tensor(make_gmem_ptr(sLSE_ptr), layout_lse);

    // Phase 1: preprocess LSE while the split-O TMA load is still in flight.
    // Each thread caches num_split scale factors (one scalar per split, not per element).
    accum_t sn_arr[MAX_SPLIT];
    accum_t sum_all = 0.f;
    if (thread_id < ACTIVE_THREADS) {
        accum_t max_all = -FLT_MAX;
        for (int s = 0; s < num_split; ++s)
            max_all = fmaxf(max_all, gLSE(my_q, s));
        for (int s = 0; s < num_split; ++s) {
            sn_arr[s] = expf(gLSE(my_q, s) - max_all);
            sum_all  += sn_arr[s];
        }
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
        for (int s = 0; s < num_split; ++s) {
            const accum_t sn = sn_arr[s];
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

    // Normalize and write to output smem.
    const int slot_final = m2c.template pop<0>();
    vec2_t* sF_ptr = (vec2_t*)get_slot_address(base, extract(slot_final));
    auto sF = make_tensor(make_smem_ptr(sF_ptr), layout_sO);

    if (thread_id < ACTIVE_THREADS) {
        const accum_t inv_sum = 1.f / sum_all;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            sF(my_q, my_i_base + e) = Tr::from_float2({acc[e].x * inv_sum, acc[e].y * inv_sum});
        }
    }

    __sync_compute_group(128);
    c2m.template push<0, true>(thread_id, slot_final);
}