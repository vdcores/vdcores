#pragma once

// Compile-time GDAKI specialization for PoolInst dispatch.  NCCL owns the
// communicator, registered window, QPs, keys, and credit state; this helper
// changes only WQE construction so one RDMA WRITE can gather independently
// addressed activation rows into a contiguous remote interval.
#if DAE_POOL_SLICE_RAW_SGL

#include <nccl_device/gin/gdaki/gin_gdaki.h>

#include <cstddef>
#include <cstdint>

#ifdef __CUDA_ARCH__

static __device__ __forceinline__ void* pool_gin_gdaki_wqe_segment(
    doca_gpu_dev_verbs_qp* qp,
    uint64_t base_wqe,
    uint32_t segment) {
  constexpr uint32_t segment_bytes = 16;
  constexpr uint32_t wqebb_bytes = 64;
  const uint32_t byte_offset = segment * segment_bytes;
  auto* wqebb = reinterpret_cast<uint8_t*>(
      doca_gpu_dev_verbs_get_wqe_ptr(
          qp, base_wqe + byte_offset / wqebb_bytes));
  return wqebb + byte_offset % wqebb_bytes;
}

static __device__ __forceinline__ uint32_t
pool_gin_gdaki_sgl_wqebbs(uint32_t source_segments) {
  // RC WRITE consists of control, remote-address, then N local data segments.
  return (2 + source_segments + 3) / 4;
}

static __device__ __forceinline__ uint64_t pool_gin_window_address(
    const void* address) {
  using nccl::utility::loadConst;
  return 4096ULL * loadConst(
      &dae_pool_gin_transport_state.window->ginOffset4K) +
      pool_gin_offset(address);
}

// A group is reserved once on one NCCL-owned QP, built cooperatively by one
// warp, and submitted with one doorbell.  Each data WQE is followed by an
// inline progress write on the same RC QP.  Therefore progress K names exactly
// the first K contiguous destination segments without a system-wide fence or
// a transport-wide completion sweep.
static __device__ __noinline__ void pool_gin_gdaki_sgl_put_rows_warp(
    uint8_t* destination,
    uint64_t* ready,
    const uint8_t* token_pool,
    const uint32_t* target_rows,
    uint32_t row_begin,
    uint32_t row_end,
    uint32_t row_bytes,
    uint32_t write_chunk_rows,
    int target_pe,
    const int* barriers,
    uint32_t write_barrier,
    uint64_t sequence,
    uint32_t lane) {
  using nccl::utility::loadConst;
  const uint32_t row_count = row_end - row_begin;

  // Acquire only writer chunks named by this group. Groups whose chunks are
  // already ready proceed independently on other PoolInst CTAs/QPs.
  for (uint32_t packed_row = lane;
       packed_row < row_count;
       packed_row += 32) {
    const uint32_t source_row = target_rows[row_begin + packed_row];
    const uint32_t chunk = source_row / write_chunk_rows;
    while (!pool_slice_barrier_ready(barriers + write_barrier + chunk))
      __nanosleep(barrierPollSleepCycles);
  }
  __syncwarp();

  const PoolGin gin = pool_gin();
  auto* gdaki =
      &reinterpret_cast<ncclGinGdakiGPUContext*>(gin._ginHandle)[
          gin.contextId];
  doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + target_pe;
  auto* memory_handle = reinterpret_cast<ncclGinGdakiMemHandle*>(
      loadConst(
          &dae_pool_gin_transport_state.window->ginWins[
              gin.connectionId]));
  const uint32_t local_key = loadConst(&memory_handle->lkey);
  const uint32_t remote_key =
      loadConst(loadConst(&memory_handle->rkeys) + target_pe);
  const uint64_t remote_base = pool_gin_window_address(destination);
  const uint64_t ready_remote = pool_gin_window_address(ready);

  constexpr uint32_t width = DAE_POOL_SLICE_RAW_SGL_WIDTH;
  const uint32_t sgl_count = (row_count + width - 1) / width;
  uint32_t total_wqebbs = 0;
  for (uint32_t sgl = 0; sgl < sgl_count; ++sgl) {
    const uint32_t first = sgl * width;
    const uint32_t count = row_count - first < width
        ? row_count - first
        : width;
    total_wqebbs += pool_gin_gdaki_sgl_wqebbs(count) + 1;
  }

  uint64_t base_wqe = 0;
  if (lane == 0) {
    base_wqe = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            qp, total_wqebbs);
  }
  base_wqe = __shfl_sync(0xffffffffU, base_wqe, 0);

  uint32_t slot_offset = 0;
  uint64_t remote_offset = 0;
  for (uint32_t sgl = 0; sgl < sgl_count; ++sgl) {
    const uint32_t first = sgl * width;
    const uint32_t count = row_count - first < width
        ? row_count - first
        : width;
    const uint32_t data_wqebbs = pool_gin_gdaki_sgl_wqebbs(count);
    const uint64_t wqe_base = base_wqe + slot_offset;

    if (lane == 0) {
      doca_gpunetio_ib_mlx5_wqe_ctrl_seg control{};
      control.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
          (static_cast<uint32_t>(static_cast<uint16_t>(wqe_base))
               << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
          DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE);
      control.qpn_ds = doca_gpu_dev_verbs_bswap32(
          loadConst(&qp->sq_num_shift8) | (2 + count));
      control.fm_ce_se = 0;

      doca_gpunetio_ib_mlx5_wqe_raddr_seg remote{};
      remote.raddr = doca_gpu_dev_verbs_bswap64(
          remote_base + remote_offset);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
      remote.rkey = remote_key;
#else
      remote.rkey = doca_gpu_dev_verbs_bswap32(remote_key);
#endif
      doca_gpu_dev_verbs_store_wqe_seg(
          reinterpret_cast<uint64_t*>(
              pool_gin_gdaki_wqe_segment(qp, wqe_base, 0)),
          reinterpret_cast<uint64_t*>(&control));
      doca_gpu_dev_verbs_store_wqe_seg(
          reinterpret_cast<uint64_t*>(
              pool_gin_gdaki_wqe_segment(qp, wqe_base, 1)),
          reinterpret_cast<uint64_t*>(&remote));
    }

    if (lane < count) {
      const uint32_t packed_row = first + lane;
      const uint32_t source_row = target_rows[row_begin + packed_row];
      const uint64_t local_address = pool_gin_window_address(
          token_pool + static_cast<uint64_t>(source_row) * row_bytes);
      doca_gpunetio_ib_mlx5_wqe_data_seg data{};
      data.byte_count = doca_gpu_dev_verbs_bswap32(row_bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
      data.lkey = local_key;
#else
      data.lkey = doca_gpu_dev_verbs_bswap32(local_key);
#endif
      data.addr = doca_gpu_dev_verbs_bswap64(local_address);
      doca_gpu_dev_verbs_store_wqe_seg(
          reinterpret_cast<uint64_t*>(pool_gin_gdaki_wqe_segment(
              qp, wqe_base, 2 + lane)),
          reinterpret_cast<uint64_t*>(&data));
    }
    __syncwarp();

    if (lane == 0) {
      const uint64_t progress =
          sequence * poolSliceRawSglProgressStride + sgl + 1;
      auto* progress_wqe = doca_gpu_dev_verbs_get_wqe_ptr(
          qp, wqe_base + data_wqebbs);
      const auto completion = sgl + 1 == sgl_count
          ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
          : static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(0);
      doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(
          qp,
          progress_wqe,
          static_cast<uint16_t>(wqe_base + data_wqebbs),
          completion,
          ready_remote,
          remote_key,
          sizeof(progress));
      doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data(
          qp, progress_wqe, progress);
    }
    __syncwarp();
    slot_offset += data_wqebbs + 1;
    remote_offset += static_cast<uint64_t>(count) * row_bytes;
  }

  if (lane == 0) {
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            qp, base_wqe, base_wqe + total_wqebbs - 1);
    // mark_wqes_ready issues the required GPU-scope release for descriptors;
    // submission itself therefore needs no broader ordering operation.
    doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
            qp, base_wqe + total_wqebbs);
  }
  __syncwarp();
}

#endif  // __CUDA_ARCH__

#endif  // DAE_POOL_SLICE_RAW_SGL
