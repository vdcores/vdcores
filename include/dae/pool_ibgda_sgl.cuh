#pragma once

// This header is compiled only for the explicitly selected PoolInst transport
// experiment. It uses NVSHMEM 3.4's pinned IBGDA non-ABI layer to express a
// verbs-style SGL RDMA write directly from GPU HBM. The common build never
// parses or instantiates these helpers.
#if DAE_POOL_SLICE_RAW_SGL

#include <non_abi/device/pt-to-pt/ibgda_device.cuh>

#include <cstddef>
#include <cstdint>

#ifdef __CUDA_ARCH__

static __device__ __forceinline__ void* pool_ibgda_sgl_segment(
    nvshmemi_ibgda_device_qp_t* qp,
    uint64_t base_wqe,
    uint32_t data_segment) {
  constexpr uint32_t segment_bytes = 16;
  constexpr uint32_t wqebb_bytes = 1U << MLX5_SEND_WQE_SHIFT;
  const uint32_t byte_offset = data_segment * segment_bytes;
  auto* wqebb = static_cast<uint8_t*>(
      ibgda_get_wqe_ptr(
          qp,
          static_cast<uint16_t>(base_wqe + byte_offset / wqebb_bytes)));
  return wqebb + byte_offset % wqebb_bytes;
}

static __device__ __forceinline__ uint32_t
pool_ibgda_sgl_wqebbs(uint32_t source_segments) {
  // RC WRITE has one control, one remote-address, then N local data segments.
  constexpr uint32_t segments_per_wqebb =
      (1U << MLX5_SEND_WQE_SHIFT) / 16;
  return (2 + source_segments + segments_per_wqebb - 1) /
      segments_per_wqebb;
}

// Concatenate one group of independently addressed source rows into its
// already-contiguous destination interval. One warp constructs all WQEs,
// appends an exact progress generation after every contiguous segment, and
// submits the complete chain with one doorbell. A destination reader can copy
// the first segment while later writes remain in flight without creating
// another queue instruction or reader CTA. Selecting the raw build is a static
// contract: CTA-mapped single-NIC RC QPs and wholly registered GPU buffers are
// required, so the hot path carries no public-transport fallback or validation
// matrix.
static __device__ __noinline__ void pool_ibgda_sgl_put_rows_warp(
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
  const uint32_t row_count = row_end - row_begin;

  // Acquire only the VDCores writer chunks named by this group. Each source
  // row is owned by one lane, so unrelated chunks and other groups continue
  // independently while the raw transport is being prepared.
  if (lane < row_count) {
    const uint32_t source_row = target_rows[row_begin + lane];
    const uint32_t chunk = source_row / write_chunk_rows;
    while (!pool_slice_barrier_ready(
        barriers + write_barrier + chunk))
      __nanosleep(barrierPollSleepCycles);
  }
  __syncwarp();

  bool qp_shared = true;
  nvshmemi_ibgda_device_qp_t* qp = nullptr;
  if (lane == 0)
    qp = ibgda_get_qp(target_pe, &qp_shared);
  qp = reinterpret_cast<nvshmemi_ibgda_device_qp_t*>(
      __shfl_sync(0xffffffffU, reinterpret_cast<uintptr_t>(qp), 0));

  uint64_t remote_address = 0;
  __be32 remote_key = 0;
  size_t remote_chunk = 0;
  if (lane == 0) {
    ibgda_get_raddr_rkey(
        reinterpret_cast<uint64_t>(destination),
        target_pe,
        target_pe,
        &remote_address,
        &remote_key,
        &remote_chunk,
        qp->dev_idx);
  }
  remote_address = __shfl_sync(0xffffffffU, remote_address, 0);
  remote_key = __shfl_sync(0xffffffffU, remote_key, 0);

  uint64_t ready_remote_address = 0;
  __be32 ready_remote_key = 0;
  size_t ready_remote_chunk = 0;
  if (lane == 0) {
    ibgda_get_raddr_rkey(
        reinterpret_cast<uint64_t>(ready),
        target_pe,
        target_pe,
        &ready_remote_address,
        &ready_remote_key,
        &ready_remote_chunk,
        qp->dev_idx);
  }
  ready_remote_address =
      __shfl_sync(0xffffffffU, ready_remote_address, 0);
  ready_remote_key = __shfl_sync(0xffffffffU, ready_remote_key, 0);

  constexpr uint32_t width = DAE_POOL_SLICE_RAW_SGL_WIDTH;
  const uint32_t wqe_count = (row_count + width - 1) / width;
  uint32_t total_wqebbs = 0;
  for (uint32_t wqe = 0; wqe < wqe_count; ++wqe) {
    const uint32_t first = wqe * width;
    const uint32_t count = row_count - first < width
        ? row_count - first
        : width;
    // Each data WQE is immediately followed by one inline 8-byte progress
    // WRITE on the same RC QP. Only the final progress WRITE requests a CQE.
    total_wqebbs += pool_ibgda_sgl_wqebbs(count) + 1;
  }

  const uint32_t reserved_wqebbs = total_wqebbs;
  uint64_t base_wqe = 0;
  if (lane == 0)
    base_wqe = ibgda_reserve_wqe_slots(qp, reserved_wqebbs, true);
  base_wqe = __shfl_sync(0xffffffffU, base_wqe, 0);

  uint32_t slot_offset = 0;
  uint64_t remote_offset = 0;
  for (uint32_t wqe = 0; wqe < wqe_count; ++wqe) {
    const uint32_t first = wqe * width;
    const uint32_t count = row_count - first < width
        ? row_count - first
        : width;
    const uint32_t data_wqebbs = pool_ibgda_sgl_wqebbs(count);
    const uint64_t wqe_base = base_wqe + slot_offset;
    if (lane == 0) {
      ibgda_ctrl_seg_t control_segment{};
      control_segment.qpn_ds = HTOBE32((qp->qpn << 8) | (2 + count));
      // The immediately following progress WRITE supplies the exact
      // dependency. Data WQEs themselves do not generate CQEs.
      control_segment.fm_ce_se = 0;
      control_segment.opmod_idx_opcode = HTOBE32(
          (static_cast<uint16_t>(wqe_base) << 8) |
          MLX5_OPCODE_RDMA_WRITE);

      mlx5_wqe_raddr_seg address_segment{};
      address_segment.raddr = HTOBE64(remote_address + remote_offset);
      address_segment.rkey = remote_key;

      auto* control_destination = static_cast<uint32_t*>(
          pool_ibgda_sgl_segment(qp, wqe_base, 0));
      const auto* control_source =
          reinterpret_cast<const uint32_t*>(&control_segment);
#pragma unroll
      for (uint32_t word = 0;
           word < sizeof(control_segment) / sizeof(uint32_t);
           ++word)
        ibgda_store_relaxed(
            control_destination + word, control_source[word]);

      auto* address_destination = static_cast<uint32_t*>(
          pool_ibgda_sgl_segment(qp, wqe_base, 1));
      const auto* address_source =
          reinterpret_cast<const uint32_t*>(&address_segment);
#pragma unroll
      for (uint32_t word = 0;
           word < sizeof(address_segment) / sizeof(uint32_t);
           ++word)
        ibgda_store_relaxed(
            address_destination + word, address_source[word]);
    }

    if (lane < count) {
      const uint32_t packed_row = first + lane;
      const uint32_t row = target_rows[row_begin + packed_row];
      const uint64_t address = reinterpret_cast<uint64_t>(token_pool) +
          static_cast<uint64_t>(row) * row_bytes;
      __be32 key = 0;
      size_t chunk_size = 0;
      bool system_memory = false;
      ibgda_get_lkey(
          address,
          &key,
          &chunk_size,
          &system_memory,
          qp->dev_idx);
      mlx5_wqe_data_seg data_segment{};
      data_segment.byte_count = HTOBE32(row_bytes);
      data_segment.lkey = key;
      data_segment.addr = HTOBE64(address);
      auto* data_destination = static_cast<uint32_t*>(
          pool_ibgda_sgl_segment(qp, wqe_base, 2 + lane));
      const auto* data_source =
          reinterpret_cast<const uint32_t*>(&data_segment);
#pragma unroll
      for (uint32_t word = 0;
           word < sizeof(data_segment) / sizeof(uint32_t);
           ++word)
        ibgda_store_relaxed(data_destination + word, data_source[word]);
    }
    __syncwarp();
    if (lane == 0) {
      const uint64_t progress =
          sequence * poolSliceRawSglProgressStride + wqe + 1;
      void* progress_wqes[2] = {
          ibgda_get_wqe_ptr(qp, wqe_base + data_wqebbs), nullptr};
      ibgda_write_rdma_write_inl_wqe<false>(
          qp,
          nullptr,
          &progress,
          ready_remote_address,
          ready_remote_key,
          sizeof(progress),
          static_cast<uint16_t>(wqe_base + data_wqebbs),
          wqe + 1 == wqe_count ? MLX5_WQE_CTRL_CQ_UPDATE : 0,
          progress_wqes);
    }
    __syncwarp();
    slot_offset += data_wqebbs + 1;
    remote_offset += static_cast<uint64_t>(count) * row_bytes;
  }

  if (lane == 0)
    ibgda_submit_requests<true>(qp, base_wqe, reserved_wqebbs);
  __syncwarp();
}

// Post one contiguous return payload and its exact generation as a single
// two-WQE RC chain. This avoids running the public NVSHMEM reservation and
// submission path twice for a buffer that PoolInst has already coalesced.
static __device__ __noinline__ void
pool_ibgda_put_contiguous_signal_warp(
    uint8_t* destination,
    uint64_t* ready,
    const uint8_t* source,
    uint32_t bytes,
    int target_pe,
    uint64_t sequence,
    uint32_t lane) {
  bool qp_shared = true;
  nvshmemi_ibgda_device_qp_t* qp = nullptr;
  if (lane == 0)
    qp = ibgda_get_qp(target_pe, &qp_shared);
  qp = reinterpret_cast<nvshmemi_ibgda_device_qp_t*>(
      __shfl_sync(0xffffffffU, reinterpret_cast<uintptr_t>(qp), 0));

  __be32 local_key = 0;
  size_t local_chunk = 0;
  bool system_memory = false;
  uint64_t remote_address = 0;
  __be32 remote_key = 0;
  size_t remote_chunk = 0;
  uint64_t ready_remote_address = 0;
  __be32 ready_remote_key = 0;
  size_t ready_remote_chunk = 0;
  if (lane == 0) {
    ibgda_get_lkey(
        reinterpret_cast<uint64_t>(source),
        &local_key,
        &local_chunk,
        &system_memory,
        qp->dev_idx);
    ibgda_get_raddr_rkey(
        reinterpret_cast<uint64_t>(destination),
        target_pe,
        target_pe,
        &remote_address,
        &remote_key,
        &remote_chunk,
        qp->dev_idx);
    ibgda_get_raddr_rkey(
        reinterpret_cast<uint64_t>(ready),
        target_pe,
        target_pe,
        &ready_remote_address,
        &ready_remote_key,
        &ready_remote_chunk,
        qp->dev_idx);
  }
  local_key = __shfl_sync(0xffffffffU, local_key, 0);
  remote_address = __shfl_sync(0xffffffffU, remote_address, 0);
  remote_key = __shfl_sync(0xffffffffU, remote_key, 0);
  ready_remote_address =
      __shfl_sync(0xffffffffU, ready_remote_address, 0);
  ready_remote_key = __shfl_sync(0xffffffffU, ready_remote_key, 0);

  constexpr uint32_t wqebbs = 2;
  uint64_t base_wqe = 0;
  if (lane == 0)
    base_wqe = ibgda_reserve_wqe_slots(qp, wqebbs, true);
  base_wqe = __shfl_sync(0xffffffffU, base_wqe, 0);
  if (lane == 0) {
    void* payload_wqes[2] = {
        ibgda_get_wqe_ptr(qp, base_wqe), nullptr};
    ibgda_write_rdma_write_wqe<false>(
        qp,
        nullptr,
        reinterpret_cast<uint64_t>(source),
        local_key,
        remote_address,
        remote_key,
        bytes,
        static_cast<uint16_t>(base_wqe),
        0,
        payload_wqes);
    void* signal_wqes[2] = {
        ibgda_get_wqe_ptr(qp, base_wqe + 1), nullptr};
    ibgda_write_rdma_write_inl_wqe<false>(
        qp,
        nullptr,
        &sequence,
        ready_remote_address,
        ready_remote_key,
        sizeof(sequence),
        static_cast<uint16_t>(base_wqe + 1),
        MLX5_WQE_CTRL_CQ_UPDATE,
        signal_wqes);
    ibgda_submit_requests<true>(qp, base_wqe, wqebbs);
  }
  __syncwarp();
}

#endif  // __CUDA_ARCH__

#endif  // DAE_POOL_SLICE_RAW_SGL
