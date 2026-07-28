#include <infiniband/verbs.h>

#include "host_sgl_ring_abi.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <new>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

struct HostSglPostedBatch {
  uint64_t completion_sequence = 0;
  uint32_t send_wr_count = 0;
  std::vector<uint64_t> inline_signals;
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> wrs;
};

struct HostSglQueue {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_device_attr device_attr{};
  ibv_port_attr port_attr{};
  ibv_gid gid{};
  uint8_t port_num = 1;
  uint8_t gid_index = 0;
  uint32_t psn = 0;
  uint32_t max_send_wr = 0;
  uint32_t max_sge = 0;
  uint32_t max_outstanding_batches = 0;
  bool connected = false;
  uint32_t outstanding_send_wrs = 0;
  // The NIC may still be fetching WRs and SGEs after ibv_post_send returns.
  // Retain each post's complete descriptor graph until its signaled CQE is
  // observed; sharing one scratch vector across outstanding posts races that
  // device-side fetch and defeats the intended host pipeline.
  std::deque<HostSglPostedBatch> outstanding_batches;
};

struct HostSglRing {
  void* allocation = nullptr;
  size_t allocation_bytes = 0;
  HostSglRingMemory* memory = nullptr;
  char error[256]{};
};

void set_error(char* error, size_t error_bytes, const char* format, ...) {
  if (error == nullptr || error_bytes == 0) {
    return;
  }
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(error, error_bytes, format, arguments);
  va_end(arguments);
  error[error_bytes - 1] = '\0';
}

uint32_t make_psn(const HostSglQueue* queue) {
  const auto value =
                     (static_cast<uint64_t>(::getpid()) << 12) ^
                     reinterpret_cast<uintptr_t>(queue) ^
                     static_cast<uint64_t>(queue->qp->qp_num);
  const uint32_t psn = static_cast<uint32_t>(value & 0x00ffffffu);
  return psn == 0 ? 1 : psn;
}

void destroy_queue(HostSglQueue* queue) {
  if (queue == nullptr) {
    return;
  }
  if (queue->qp != nullptr) {
    ibv_destroy_qp(queue->qp);
    queue->qp = nullptr;
  }
  if (queue->cq != nullptr) {
    ibv_destroy_cq(queue->cq);
    queue->cq = nullptr;
  }
  delete queue;
}

}  // namespace

extern "C" {

uint32_t host_sgl_abi_version() { return 4; }

void* host_sgl_create_qp(void* context_pointer, void* pd_pointer,
                         uint8_t port_num, uint8_t gid_index,
                         uint32_t requested_send_wr,
                         uint32_t requested_send_sge, char* error,
                         size_t error_bytes) {
  if (context_pointer == nullptr || pd_pointer == nullptr) {
    set_error(error, error_bytes, "context and protection domain are required");
    return nullptr;
  }
  if (requested_send_wr < 2 || requested_send_sge == 0) {
    set_error(error, error_bytes,
              "requested_send_wr must be >= 2 and requested_send_sge >= 1");
    return nullptr;
  }

  auto* queue = new (std::nothrow) HostSglQueue;
  if (queue == nullptr) {
    set_error(error, error_bytes, "failed to allocate HostSglQueue");
    return nullptr;
  }
  queue->context = static_cast<ibv_context*>(context_pointer);
  queue->pd = static_cast<ibv_pd*>(pd_pointer);
  queue->port_num = port_num;
  queue->gid_index = gid_index;

  int result = ibv_query_device(queue->context, &queue->device_attr);
  if (result != 0) {
    set_error(error, error_bytes, "ibv_query_device failed: %s",
              std::strerror(result));
    destroy_queue(queue);
    return nullptr;
  }
  result = ibv_query_port(queue->context, port_num, &queue->port_attr);
  if (result != 0) {
    set_error(error, error_bytes, "ibv_query_port(%u) failed: %s", port_num,
              std::strerror(result));
    destroy_queue(queue);
    return nullptr;
  }
  if (queue->port_attr.state != IBV_PORT_ACTIVE) {
    set_error(error, error_bytes, "verbs port %u is not active (state=%d)",
              port_num, static_cast<int>(queue->port_attr.state));
    destroy_queue(queue);
    return nullptr;
  }
  result = ibv_query_gid(queue->context, port_num, gid_index, &queue->gid);
  if (result != 0) {
    set_error(error, error_bytes, "ibv_query_gid(%u, %u) failed: %s",
              port_num, gid_index, std::strerror(result));
    destroy_queue(queue);
    return nullptr;
  }

  const uint32_t send_wr =
      std::min(requested_send_wr,
               static_cast<uint32_t>(queue->device_attr.max_qp_wr));
  const uint32_t send_sge =
      std::min(requested_send_sge,
               static_cast<uint32_t>(queue->device_attr.max_sge));
  if (send_wr < 2 || send_sge == 0) {
    set_error(error, error_bytes,
              "device limits cannot support the requested send queue");
    destroy_queue(queue);
    return nullptr;
  }
  // One completion covers each request batch. A small CQ still suffices for
  // the bounded pipeline even when the send queue is intentionally deep.
  const int cq_entries = static_cast<int>(
      std::min<uint32_t>(64, queue->device_attr.max_cqe));
  if (cq_entries <= 0) {
    set_error(error, error_bytes, "device reports no completion entries");
    destroy_queue(queue);
    return nullptr;
  }
  queue->cq =
      ibv_create_cq(queue->context, cq_entries, nullptr, nullptr, 0);
  if (queue->cq == nullptr) {
    set_error(error, error_bytes, "ibv_create_cq failed");
    destroy_queue(queue);
    return nullptr;
  }
  queue->max_outstanding_batches = static_cast<uint32_t>(cq_entries);

  ibv_qp_init_attr init{};
  uint32_t candidate_send_wr = send_wr;
  int create_errno = 0;
  for (;;) {
    init = ibv_qp_init_attr{};
    init.send_cq = queue->cq;
    init.recv_cq = queue->cq;
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    init.cap.max_send_wr = candidate_send_wr;
    init.cap.max_send_sge = send_sge;
    init.cap.max_recv_wr = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = sizeof(uint64_t);
    errno = 0;
    queue->qp = ibv_create_qp(queue->pd, &init);
    create_errno = errno;
    if (queue->qp != nullptr || candidate_send_wr <= 64) {
      break;
    }
    candidate_send_wr = std::max<uint32_t>(64, candidate_send_wr / 2);
  }
  if (queue->qp == nullptr) {
    set_error(error, error_bytes,
              "ibv_create_qp failed (requested=%u last_attempt=%u errno=%d: %s)",
              send_wr, candidate_send_wr, create_errno,
              std::strerror(create_errno));
    destroy_queue(queue);
    return nullptr;
  }
  if (init.cap.max_inline_data < sizeof(uint64_t)) {
    set_error(error, error_bytes,
              "QP provides %u inline bytes; 8 are required for readiness",
              init.cap.max_inline_data);
    destroy_queue(queue);
    return nullptr;
  }
  queue->max_send_wr = init.cap.max_send_wr;
  queue->max_sge = init.cap.max_send_sge;
  queue->psn = make_psn(queue);

  ibv_qp_attr init_attr{};
  init_attr.qp_state = IBV_QPS_INIT;
  init_attr.pkey_index = 0;
  init_attr.port_num = port_num;
  init_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  const int init_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS;
  result = ibv_modify_qp(queue->qp, &init_attr, init_mask);
  if (result != 0) {
    set_error(error, error_bytes, "QP RESET->INIT failed: %s",
              std::strerror(result));
    destroy_queue(queue);
    return nullptr;
  }
  return queue;
}

int host_sgl_get_endpoint(void* queue_pointer, HostSglEndpoint* endpoint,
                          char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  if (queue == nullptr || endpoint == nullptr) {
    set_error(error, error_bytes, "queue and endpoint are required");
    return -1;
  }
  std::memset(endpoint, 0, sizeof(*endpoint));
  endpoint->qp_num = queue->qp->qp_num;
  endpoint->psn = queue->psn;
  endpoint->lid = queue->port_attr.lid;
  endpoint->port_num = queue->port_num;
  endpoint->gid_index = queue->gid_index;
  endpoint->active_mtu = static_cast<uint8_t>(queue->port_attr.active_mtu);
  endpoint->link_layer = queue->port_attr.link_layer;
  std::memcpy(endpoint->gid, queue->gid.raw, sizeof(endpoint->gid));
  return 0;
}

int host_sgl_connect(void* queue_pointer,
                     const HostSglEndpoint* remote_endpoint, char* error,
                     size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  if (queue == nullptr || remote_endpoint == nullptr) {
    set_error(error, error_bytes, "queue and remote endpoint are required");
    return -1;
  }
  if (queue->connected) {
    set_error(error, error_bytes, "queue is already connected");
    return -1;
  }
  if (remote_endpoint->active_mtu < IBV_MTU_256 ||
      remote_endpoint->active_mtu > IBV_MTU_4096) {
    set_error(error, error_bytes, "remote endpoint has invalid MTU %u",
              remote_endpoint->active_mtu);
    return -1;
  }

  ibv_qp_attr rtr{};
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu = static_cast<ibv_mtu>(
      std::min<int>(queue->port_attr.active_mtu,
                    remote_endpoint->active_mtu));
  rtr.dest_qp_num = remote_endpoint->qp_num;
  rtr.rq_psn = remote_endpoint->psn;
  rtr.max_dest_rd_atomic = 1;
  rtr.min_rnr_timer = 12;
  rtr.ah_attr.port_num = queue->port_num;
  rtr.ah_attr.sl = 0;
  rtr.ah_attr.src_path_bits = 0;
  rtr.ah_attr.dlid = remote_endpoint->lid;
  const bool use_global_route =
      queue->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
      remote_endpoint->lid == 0;
  if (use_global_route) {
    rtr.ah_attr.is_global = 1;
    std::memcpy(rtr.ah_attr.grh.dgid.raw, remote_endpoint->gid,
                sizeof(remote_endpoint->gid));
    rtr.ah_attr.grh.sgid_index = queue->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
  }
  const int rtr_mask =
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  int result = ibv_modify_qp(queue->qp, &rtr, rtr_mask);
  if (result != 0) {
    set_error(error, error_bytes, "QP INIT->RTR failed: %s",
              std::strerror(result));
    return -1;
  }

  ibv_qp_attr rts{};
  rts.qp_state = IBV_QPS_RTS;
  // This is the RC packet-ACK retry exponent required by verbs, not a pool
  // request deadline. Logical readiness and retirement remain generation-only.
  rts.timeout = 14;
  rts.retry_cnt = 7;
  rts.rnr_retry = 7;
  rts.sq_psn = queue->psn;
  rts.max_rd_atomic = 1;
  const int rts_mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                       IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                       IBV_QP_MAX_QP_RD_ATOMIC;
  result = ibv_modify_qp(queue->qp, &rts, rts_mask);
  if (result != 0) {
    set_error(error, error_bytes, "QP RTR->RTS failed: %s",
              std::strerror(result));
    return -1;
  }
  queue->connected = true;
  return 0;
}

uint32_t host_sgl_max_send_wr(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglQueue*>(queue_pointer);
  return queue == nullptr ? 0 : queue->max_send_wr;
}

uint32_t host_sgl_max_sge(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglQueue*>(queue_pointer);
  return queue == nullptr ? 0 : queue->max_sge;
}

uint32_t host_sgl_outstanding_batches(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglQueue*>(queue_pointer);
  return queue == nullptr
      ? 0
      : static_cast<uint32_t>(queue->outstanding_batches.size());
}

uint32_t host_sgl_outstanding_wrs(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglQueue*>(queue_pointer);
  return queue == nullptr ? 0 : queue->outstanding_send_wrs;
}

int host_sgl_post_indexed_batch(void* queue_pointer,
                                const HostSglRequest* requests,
                                uint32_t request_count, int row_wr_mode,
                                uint32_t* posted_data_wrs, char* error,
                                size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  if (queue == nullptr || requests == nullptr || request_count == 0) {
    set_error(error, error_bytes, "invalid indexed-row request batch");
    return -1;
  }
  if (!queue->connected) {
    set_error(error, error_bytes, "queue is not connected");
    return -1;
  }
  if (queue->outstanding_batches.size() >=
      queue->max_outstanding_batches) {
    set_error(error, error_bytes,
              "completion credits exhausted (%u outstanding batches)",
              queue->max_outstanding_batches);
    return -1;
  }
  const uint32_t rows_per_wr = row_wr_mode != 0 ? 1 : queue->max_sge;
  uint64_t total_rows = 0;
  uint64_t data_wr_count = 0;
  for (uint32_t request_index = 0; request_index < request_count;
       ++request_index) {
    const HostSglRequest& request = requests[request_index];
    if (request.row_indices == nullptr || request.row_count == 0 ||
        request.row_bytes == 0 ||
        request.source_stride < request.row_bytes ||
        request.remote_data == 0 || request.remote_signal == 0 ||
        request.sequence == 0) {
      set_error(error, error_bytes, "invalid request at batch index %u",
                request_index);
      return -1;
    }
    total_rows += request.row_count;
    data_wr_count +=
        (request.row_count + rows_per_wr - 1) / rows_per_wr;
  }
  const uint64_t total_wr_count = data_wr_count + request_count;
  if (total_wr_count >
      static_cast<uint64_t>(queue->max_send_wr -
                            queue->outstanding_send_wrs)) {
    set_error(error, error_bytes,
              "%llu WRs exceed available SQ credits=%u (max=%u outstanding=%u)",
              static_cast<unsigned long long>(total_wr_count),
              queue->max_send_wr - queue->outstanding_send_wrs,
              queue->max_send_wr, queue->outstanding_send_wrs);
    return -1;
  }
  if (total_rows + request_count > SIZE_MAX ||
      total_wr_count > SIZE_MAX || data_wr_count > UINT32_MAX) {
    set_error(error, error_bytes, "request batch is too large");
    return -1;
  }

  queue->outstanding_batches.emplace_back();
  HostSglPostedBatch& posted = queue->outstanding_batches.back();
  posted.completion_sequence = requests[request_count - 1].sequence;
  posted.send_wr_count = static_cast<uint32_t>(total_wr_count);
  posted.inline_signals.assign(request_count, 0);
  posted.sges.assign(static_cast<size_t>(total_rows + request_count),
                     ibv_sge{});
  posted.wrs.assign(static_cast<size_t>(total_wr_count), ibv_send_wr{});

  size_t sge_cursor = 0;
  size_t wr_cursor = 0;
  for (uint32_t request_index = 0; request_index < request_count;
       ++request_index) {
    const HostSglRequest& request = requests[request_index];
    const size_t row_sge_base = sge_cursor;
    for (uint32_t row = 0; row < request.row_count; ++row) {
      ibv_sge& sge = posted.sges[sge_cursor++];
      sge.addr = request.source_base +
                 static_cast<uint64_t>(request.row_indices[row]) *
                     request.source_stride;
      sge.length = request.row_bytes;
      sge.lkey = request.local_lkey;
    }

    uint32_t first_row = 0;
    uint64_t remote_offset = 0;
    while (first_row < request.row_count) {
      const uint32_t rows =
          std::min(rows_per_wr, request.row_count - first_row);
      ibv_send_wr& wr = posted.wrs[wr_cursor++];
      wr.sg_list = &posted.sges[row_sge_base + first_row];
      wr.num_sge = static_cast<int>(rows);
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.wr.rdma.remote_addr = request.remote_data + remote_offset;
      wr.wr.rdma.rkey = request.remote_rkey;
      first_row += rows;
      remote_offset += static_cast<uint64_t>(rows) * request.row_bytes;
    }

    posted.inline_signals[request_index] = request.sequence;
    ibv_sge& signal_sge = posted.sges[sge_cursor++];
    signal_sge.addr =
        reinterpret_cast<uint64_t>(&posted.inline_signals[request_index]);
    signal_sge.length = sizeof(uint64_t);
    signal_sge.lkey = 0;
    ibv_send_wr& signal_wr = posted.wrs[wr_cursor++];
    signal_wr.wr_id = request.sequence;
    signal_wr.sg_list = &signal_sge;
    signal_wr.num_sge = 1;
    signal_wr.opcode = IBV_WR_RDMA_WRITE;
    signal_wr.send_flags = IBV_SEND_INLINE;
    if (request_index + 1 == request_count) {
      signal_wr.send_flags |= IBV_SEND_SIGNALED;
    }
    signal_wr.wr.rdma.remote_addr = request.remote_signal;
    signal_wr.wr.rdma.rkey = request.remote_rkey;
  }
  for (size_t wr_index = 0; wr_index + 1 < posted.wrs.size(); ++wr_index) {
    posted.wrs[wr_index].next = &posted.wrs[wr_index + 1];
  }
  posted.wrs.back().next = nullptr;

  ibv_send_wr* bad_wr = nullptr;
  const int result = ibv_post_send(queue->qp, posted.wrs.data(), &bad_wr);
  if (result != 0) {
    const auto bad_index =
        bad_wr == nullptr ? -1 : static_cast<int>(bad_wr - posted.wrs.data());
    set_error(error, error_bytes, "ibv_post_send failed at WR %d: %s",
              bad_index, std::strerror(result));
    // WRs preceding bad_wr may already be owned by the QP. Keep this batch's
    // descriptor storage alive until the caller destroys the failed queue.
    return -1;
  }
  queue->outstanding_send_wrs += static_cast<uint32_t>(total_wr_count);
  if (posted_data_wrs != nullptr) {
    *posted_data_wrs = static_cast<uint32_t>(data_wr_count);
  }
  return 0;
}

int host_sgl_try_poll(void* queue_pointer, uint64_t* completed_sequence,
                      char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  if (queue == nullptr) {
    set_error(error, error_bytes, "queue is required");
    return -1;
  }
  if (queue->outstanding_batches.empty()) {
    return 0;
  }
  ibv_wc completion{};
  const int count = ibv_poll_cq(queue->cq, 1, &completion);
  if (count < 0) {
    set_error(error, error_bytes, "ibv_poll_cq failed");
    return -1;
  }
  if (count == 0) {
    return 0;
  }
  if (completion.status != IBV_WC_SUCCESS) {
    set_error(error, error_bytes,
              "send completion failed: %s (vendor_err=%u)",
              ibv_wc_status_str(completion.status), completion.vendor_err);
    return -1;
  }
  const HostSglPostedBatch& expected = queue->outstanding_batches.front();
  if (completion.wr_id != expected.completion_sequence) {
    set_error(error, error_bytes,
              "completion sequence mismatch: got %llu expected %llu",
              static_cast<unsigned long long>(completion.wr_id),
              static_cast<unsigned long long>(expected.completion_sequence));
    return -1;
  }
  if (queue->outstanding_send_wrs < expected.send_wr_count) {
    set_error(error, error_bytes, "internal SQ credit underflow");
    return -1;
  }
  queue->outstanding_send_wrs -= expected.send_wr_count;
  queue->outstanding_batches.pop_front();
  if (completed_sequence != nullptr) {
    *completed_sequence = completion.wr_id;
  }
  return 1;
}

int host_sgl_poll(void* queue_pointer, uint64_t sequence,
                  char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  if (queue == nullptr || queue->outstanding_batches.empty()) {
    set_error(error, error_bytes, "queue has no outstanding request batch");
    return -1;
  }
  if (queue->outstanding_batches.front().completion_sequence != sequence) {
    set_error(error, error_bytes,
              "poll sequence %llu is not the oldest outstanding batch %llu",
              static_cast<unsigned long long>(sequence),
              static_cast<unsigned long long>(
                  queue->outstanding_batches.front().completion_sequence));
    return -1;
  }
  for (;;) {
    uint64_t completed_sequence = 0;
    const int count = host_sgl_try_poll(
        queue, &completed_sequence, error, error_bytes);
    if (count < 0) {
      return -1;
    }
    if (count == 1) {
      if (completed_sequence != sequence) {
        set_error(error, error_bytes,
                  "completion sequence mismatch: got %llu expected %llu",
                  static_cast<unsigned long long>(completed_sequence),
                  static_cast<unsigned long long>(sequence));
        return -1;
      }
      return 0;
    }
  }
}

void* host_sgl_create_ring(char* error, size_t error_bytes) {
  constexpr size_t slot_alignment = alignof(HostSglRingSlot);
  const auto align_up = [](size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
  };
  const size_t slots_offset =
      align_up(sizeof(HostSglRingMemory), slot_alignment);
  const size_t slots_bytes =
      hostSglRingCapacity * sizeof(HostSglRingSlot);
  const size_t indices_offset = align_up(slots_offset + slots_bytes, 64);
  const size_t indices_bytes =
      hostSglRingCapacity * hostSglRingMaxRows * sizeof(uint32_t);

  auto* ring = new (std::nothrow) HostSglRing;
  if (ring == nullptr) {
    set_error(error, error_bytes, "failed to allocate ring handle");
    return nullptr;
  }
  ring->allocation_bytes = indices_offset + indices_bytes;
  if (posix_memalign(
          &ring->allocation, slot_alignment, ring->allocation_bytes) != 0) {
    set_error(error, error_bytes, "aligned ring malloc failed");
    delete ring;
    return nullptr;
  }
  std::memset(ring->allocation, 0, ring->allocation_bytes);
  ring->memory = static_cast<HostSglRingMemory*>(ring->allocation);
  auto* slots = reinterpret_cast<HostSglRingSlot*>(
      static_cast<uint8_t*>(ring->allocation) + slots_offset);
  auto* indices = reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(ring->allocation) + indices_offset);
  ring->memory->abi_version = 1;
  ring->memory->capacity = hostSglRingCapacity;
  ring->memory->max_rows = hostSglRingMaxRows;
  ring->memory->slots_address = reinterpret_cast<uint64_t>(slots);
  ring->memory->indices_address = reinterpret_cast<uint64_t>(indices);
  for (uint32_t slot = 0; slot < hostSglRingCapacity; ++slot) {
    slots[slot].request.row_indices =
        indices + static_cast<size_t>(slot) * hostSglRingMaxRows;
  }
  return ring;
}

void* host_sgl_ring_memory(void* ring_pointer) {
  const auto* ring = static_cast<const HostSglRing*>(ring_pointer);
  return ring == nullptr ? nullptr : ring->memory;
}

int host_sgl_consume_ring(void* queue_pointer, void* ring_pointer,
                          uint64_t first_generation, uint32_t request_count,
                          uint32_t* posted_data_wrs,
                          char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglQueue*>(queue_pointer);
  auto* ring = static_cast<HostSglRing*>(ring_pointer);
  if (queue == nullptr || ring == nullptr || ring->memory == nullptr ||
      first_generation == 0 || request_count == 0 ||
      request_count > hostSglRingMaxMessages) {
    set_error(error, error_bytes, "invalid coherent-ring consume request");
    return -1;
  }
  if (!queue->connected || !queue->outstanding_batches.empty()) {
    set_error(error, error_bytes,
              "ring consume requires one connected idle queue");
    return -1;
  }
  HostSglRingMemory* memory = ring->memory;
  if (memory->abi_version != 1 ||
      memory->capacity != hostSglRingCapacity ||
      memory->max_rows != hostSglRingMaxRows) {
    set_error(error, error_bytes, "coherent ring header is invalid");
    return -1;
  }
  auto* slots = reinterpret_cast<HostSglRingSlot*>(memory->slots_address);
  const uint64_t last_generation =
      first_generation + request_count - 1;
  if (last_generation < first_generation) {
    set_error(error, error_bytes,
              "coherent ring generation range overflows");
    return -1;
  }
  std::array<HostSglRequest, hostSglRingBatch> requests{};
  uint64_t next_generation = first_generation;
  uint64_t last_completed = __atomic_load_n(
      &memory->completed_generation, __ATOMIC_ACQUIRE);
  uint64_t data_wrs = 0;

  const auto poll_completion = [&]() -> int {
    uint64_t completed = 0;
    const int count =
        host_sgl_try_poll(queue, &completed, error, error_bytes);
    if (count == 1) {
      last_completed = completed;
      __atomic_store_n(
          &memory->completed_generation, completed, __ATOMIC_RELEASE);
    }
    return count;
  };

  while (next_generation <= last_generation) {
    HostSglRingSlot& head =
        slots[(next_generation - 1) % memory->capacity];
    const uint64_t ready =
        __atomic_load_n(&head.ready_generation, __ATOMIC_ACQUIRE);
    if (ready != next_generation) {
      if (poll_completion() < 0)
        return -1;
      continue;
    }

    uint32_t batch_count = 0;
    uint64_t batch_end = next_generation;
    while (batch_end <= last_generation &&
           batch_count < hostSglRingBatch) {
      HostSglRingSlot& slot =
          slots[(batch_end - 1) % memory->capacity];
      if (__atomic_load_n(
              &slot.ready_generation, __ATOMIC_ACQUIRE) != batch_end)
        break;
      requests[batch_count++] = slot.request;
      ++batch_end;
    }

    uint64_t required_wrs = batch_count;
    for (uint32_t index = 0; index < batch_count; ++index) {
      const HostSglRequest& request = requests[index];
      required_wrs +=
          (request.row_count + queue->max_sge - 1) / queue->max_sge;
    }
    while (queue->outstanding_batches.size() >=
               queue->max_outstanding_batches ||
           required_wrs > queue->max_send_wr - queue->outstanding_send_wrs) {
      if (poll_completion() < 0)
        return -1;
    }

    uint32_t batch_data_wrs = 0;
    if (host_sgl_post_indexed_batch(
            queue,
            requests.data(),
            batch_count,
            0,
            &batch_data_wrs,
            error,
            error_bytes) != 0) {
      memory->error_generation = next_generation;
      return -1;
    }
    data_wrs += batch_data_wrs;
    for (uint64_t generation = next_generation;
         generation < batch_end;
         ++generation) {
      HostSglRingSlot& slot =
          slots[(generation - 1) % memory->capacity];
      __atomic_store_n(
          &slot.consumed_generation, generation, __ATOMIC_RELEASE);
    }
    __atomic_store_n(
        &memory->submitted_generation, batch_end - 1, __ATOMIC_RELEASE);
    next_generation = batch_end;
  }

  while (!queue->outstanding_batches.empty()) {
    if (poll_completion() < 0)
      return -1;
  }
  if (last_completed != last_generation) {
    memory->error_generation = last_generation;
    set_error(error, error_bytes,
              "coherent ring completed %llu instead of %llu",
              static_cast<unsigned long long>(last_completed),
              static_cast<unsigned long long>(last_generation));
    return -1;
  }
  if (posted_data_wrs != nullptr)
    *posted_data_wrs = static_cast<uint32_t>(data_wrs);
  return 0;
}

void host_sgl_destroy_ring(void* ring_pointer) {
  auto* ring = static_cast<HostSglRing*>(ring_pointer);
  if (ring == nullptr)
    return;
  std::free(ring->allocation);
  delete ring;
}

void host_sgl_destroy_qp(void* queue_pointer) {
  destroy_queue(static_cast<HostSglQueue*>(queue_pointer));
}

}  // extern "C"
