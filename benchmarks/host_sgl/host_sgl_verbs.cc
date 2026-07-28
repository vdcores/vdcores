#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

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

constexpr uint64_t hostSglDcKey = 0xffeeddccULL;
// A DCI WQE carries an address-vector segment in addition to RDMA and data
// segments.  Capping the scatter list at eight keeps the encoded WQE within a
// single provider-supported size while still coalescing eight noncontiguous
// activation rows into one network operation.
constexpr uint32_t hostSglDcMaxSge = 8;

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

struct HostSglDcPostedBatch {
  uint64_t completion_id = 0;
  uint32_t send_wr_count = 0;
  uint32_t ring_index = 0;
};

struct HostSglDcPeer {
  ibv_ah* ah = nullptr;
  uint32_t dctn = 0;
};

struct HostSglDcQueue {
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_srq* srq = nullptr;
  ibv_qp* dct = nullptr;
  ibv_qp* dci = nullptr;
  ibv_qp_ex* dci_ex = nullptr;
  mlx5dv_qp_ex* mlx5_dci_ex = nullptr;
  ibv_device_attr device_attr{};
  ibv_port_attr port_attr{};
  ibv_gid gid{};
  uint8_t port_num = 1;
  uint8_t gid_index = 0;
  uint32_t psn = 0;
  uint32_t max_send_wr = 0;
  uint32_t max_sge = 0;
  uint32_t outstanding_send_wrs = 0;
  uint64_t next_completion_id = 1;
  bool dct_ready = false;
  bool connected = false;
  std::vector<HostSglDcPeer> peers;
  std::deque<HostSglDcPostedBatch> outstanding_batches;
};

struct HostSglRing {
  void* allocation = nullptr;
  HostSglRingMemory* memory = nullptr;
  uint64_t next_generation = 1;
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

void destroy_dc_queue(HostSglDcQueue* queue) {
  if (queue == nullptr)
    return;
  for (HostSglDcPeer& peer : queue->peers) {
    if (peer.ah != nullptr)
      ibv_destroy_ah(peer.ah);
  }
  if (queue->dci != nullptr)
    ibv_destroy_qp(queue->dci);
  if (queue->dct != nullptr)
    ibv_destroy_qp(queue->dct);
  if (queue->srq != nullptr)
    ibv_destroy_srq(queue->srq);
  if (queue->cq != nullptr)
    ibv_destroy_cq(queue->cq);
  delete queue;
}

void fill_ah_attr(const HostSglDcQueue* queue,
                  const HostSglEndpoint& endpoint,
                  ibv_ah_attr* ah) {
  *ah = ibv_ah_attr{};
  ah->port_num = queue->port_num;
  ah->sl = 0;
  ah->src_path_bits = 0;
  ah->dlid = endpoint.lid;
  const bool use_global_route =
      queue->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
      endpoint.lid == 0;
  if (use_global_route) {
    ah->is_global = 1;
    std::memcpy(ah->grh.dgid.raw, endpoint.gid, sizeof(endpoint.gid));
    ah->grh.sgid_index = queue->gid_index;
    ah->grh.hop_limit = 64;
  }
}

}  // namespace

extern "C" {

uint32_t host_sgl_abi_version() { return 7; }

void* host_sgl_create_dc(void* context_pointer, void* pd_pointer,
                         uint8_t port_num, uint8_t gid_index,
                         uint32_t requested_send_wr,
                         uint32_t requested_send_sge, char* error,
                         size_t error_bytes) {
  if (context_pointer == nullptr || pd_pointer == nullptr ||
      requested_send_wr < 2 || requested_send_sge == 0) {
    set_error(error, error_bytes, "invalid DC creation arguments");
    return nullptr;
  }
  auto* queue = new (std::nothrow) HostSglDcQueue;
  if (queue == nullptr) {
    set_error(error, error_bytes, "failed to allocate HostSglDcQueue");
    return nullptr;
  }
  queue->context = static_cast<ibv_context*>(context_pointer);
  queue->pd = static_cast<ibv_pd*>(pd_pointer);
  queue->port_num = port_num;
  queue->gid_index = gid_index;

  int result = ibv_query_device(queue->context, &queue->device_attr);
  if (result == 0)
    result = ibv_query_port(queue->context, port_num, &queue->port_attr);
  if (result == 0)
    result = ibv_query_gid(queue->context, port_num, gid_index, &queue->gid);
  if (result != 0 || queue->port_attr.state != IBV_PORT_ACTIVE) {
    set_error(error, error_bytes, "DC device/port query failed: %s",
              result == 0 ? "port is not active" : std::strerror(result));
    destroy_dc_queue(queue);
    return nullptr;
  }

  const uint32_t send_wr = std::min(
      requested_send_wr, static_cast<uint32_t>(queue->device_attr.max_qp_wr));
  const uint32_t send_sge = std::min(
      {requested_send_sge, static_cast<uint32_t>(queue->device_attr.max_sge),
       hostSglDcMaxSge});
  queue->cq = ibv_create_cq(queue->context, 64, nullptr, nullptr, 0);
  if (queue->cq == nullptr) {
    set_error(error, error_bytes, "ibv_create_cq for DC failed: %s",
              std::strerror(errno));
    destroy_dc_queue(queue);
    return nullptr;
  }
  ibv_srq_init_attr srq_init{};
  srq_init.attr.max_wr = 1;
  srq_init.attr.max_sge = 1;
  queue->srq = ibv_create_srq(queue->pd, &srq_init);
  if (queue->srq == nullptr) {
    set_error(error, error_bytes, "ibv_create_srq for DCT failed: %s",
              std::strerror(errno));
    destroy_dc_queue(queue);
    return nullptr;
  }

  ibv_qp_init_attr_ex dct_init{};
  dct_init.qp_type = IBV_QPT_DRIVER;
  dct_init.pd = queue->pd;
  dct_init.comp_mask = IBV_QP_INIT_ATTR_PD;
  dct_init.send_cq = queue->cq;
  dct_init.recv_cq = queue->cq;
  dct_init.srq = queue->srq;
  mlx5dv_qp_init_attr dct_dv{};
  dct_dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  dct_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
  dct_dv.dc_init_attr.dct_access_key = hostSglDcKey;
  queue->dct = mlx5dv_create_qp(queue->context, &dct_init, &dct_dv);
  if (queue->dct == nullptr) {
    set_error(error, error_bytes, "mlx5dv_create_qp(DCT) failed: %s",
              std::strerror(errno));
    destroy_dc_queue(queue);
    return nullptr;
  }

  ibv_qp_attr dct_qp_init{};
  dct_qp_init.qp_state = IBV_QPS_INIT;
  dct_qp_init.pkey_index = 0;
  dct_qp_init.port_num = port_num;
  dct_qp_init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  result = ibv_modify_qp(
      queue->dct, &dct_qp_init,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
          IBV_QP_ACCESS_FLAGS);
  if (result != 0) {
    set_error(error, error_bytes, "DCT RESET->INIT failed: %s",
              std::strerror(result));
    destroy_dc_queue(queue);
    return nullptr;
  }
  ibv_qp_init_attr_ex dci_init{};
  dci_init.qp_type = IBV_QPT_DRIVER;
  dci_init.pd = queue->pd;
  dci_init.comp_mask =
      IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  dci_init.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
  dci_init.send_cq = queue->cq;
  dci_init.recv_cq = queue->cq;
  dci_init.cap.max_send_wr = send_wr;
  dci_init.cap.max_send_sge = send_sge;
  dci_init.cap.max_inline_data = sizeof(uint64_t);
  dci_init.sq_sig_all = 0;
  mlx5dv_qp_init_attr dci_dv{};
  dci_dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC |
      MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
  dci_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
  dci_dv.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
  queue->dci = mlx5dv_create_qp(queue->context, &dci_init, &dci_dv);
  if (queue->dci == nullptr) {
    set_error(error, error_bytes, "mlx5dv_create_qp(DCI) failed: %s",
              std::strerror(errno));
    destroy_dc_queue(queue);
    return nullptr;
  }
  queue->max_send_wr = dci_init.cap.max_send_wr;
  queue->max_sge = dci_init.cap.max_send_sge;
  if (queue->max_send_wr < 2 || queue->max_sge == 0 ||
      dci_init.cap.max_inline_data < sizeof(uint64_t)) {
    set_error(error, error_bytes, "DCI capabilities are insufficient");
    destroy_dc_queue(queue);
    return nullptr;
  }
  queue->dci_ex = ibv_qp_to_qp_ex(queue->dci);
  queue->mlx5_dci_ex = mlx5dv_qp_ex_from_ibv_qp_ex(queue->dci_ex);
  if (queue->dci_ex == nullptr || queue->mlx5_dci_ex == nullptr) {
    set_error(error, error_bytes, "DCI extended send API is unavailable");
    destroy_dc_queue(queue);
    return nullptr;
  }
  queue->psn = static_cast<uint32_t>(
      ((static_cast<uint64_t>(::getpid()) << 12) ^
       reinterpret_cast<uintptr_t>(queue) ^ queue->dci->qp_num) &
      0x00ffffffU);
  if (queue->psn == 0)
    queue->psn = 1;
  ibv_qp_attr dci_qp_init{};
  dci_qp_init.qp_state = IBV_QPS_INIT;
  dci_qp_init.pkey_index = 0;
  dci_qp_init.port_num = port_num;
  result = ibv_modify_qp(
      queue->dci, &dci_qp_init,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT);
  if (result != 0) {
    set_error(error, error_bytes, "DCI RESET->INIT failed: %s",
              std::strerror(result));
    destroy_dc_queue(queue);
    return nullptr;
  }
  return queue;
}

int host_sgl_get_dc_endpoint(void* queue_pointer, HostSglEndpoint* endpoint,
                             char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglDcQueue*>(queue_pointer);
  if (queue == nullptr || endpoint == nullptr) {
    set_error(error, error_bytes, "DC queue and endpoint are required");
    return -1;
  }
  std::memset(endpoint, 0, sizeof(*endpoint));
  endpoint->qp_num = queue->dct->qp_num;
  endpoint->psn = queue->psn;
  endpoint->lid = queue->port_attr.lid;
  endpoint->port_num = queue->port_num;
  endpoint->gid_index = queue->gid_index;
  endpoint->active_mtu = static_cast<uint8_t>(queue->port_attr.active_mtu);
  endpoint->link_layer = queue->port_attr.link_layer;
  std::memcpy(endpoint->gid, queue->gid.raw, sizeof(endpoint->gid));
  return 0;
}

int host_sgl_activate_dct(void* queue_pointer,
                          const HostSglEndpoint* remote_endpoint,
                          char* error, size_t error_bytes) {
  auto* queue = static_cast<HostSglDcQueue*>(queue_pointer);
  if (queue == nullptr || remote_endpoint == nullptr || queue->dct_ready ||
      remote_endpoint->active_mtu < IBV_MTU_256 ||
      remote_endpoint->active_mtu > IBV_MTU_4096) {
    set_error(error, error_bytes, "invalid DCT activation request");
    return -1;
  }
  ibv_qp_attr dct_rtr{};
  dct_rtr.qp_state = IBV_QPS_RTR;
  dct_rtr.path_mtu = static_cast<ibv_mtu>(std::min<int>(
      queue->port_attr.active_mtu, remote_endpoint->active_mtu));
  dct_rtr.min_rnr_timer = 12;
  fill_ah_attr(queue, *remote_endpoint, &dct_rtr.ah_attr);
  const int result = ibv_modify_qp(
      queue->dct, &dct_rtr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
          IBV_QP_MIN_RNR_TIMER);
  if (result != 0) {
    set_error(error, error_bytes, "DCT INIT->RTR failed: %s",
              std::strerror(result));
    return -1;
  }
  queue->dct_ready = true;
  return 0;
}

int host_sgl_connect_dc(void* queue_pointer,
                        const HostSglEndpoint* remote_endpoints,
                        uint32_t endpoint_count, char* error,
                        size_t error_bytes) {
  auto* queue = static_cast<HostSglDcQueue*>(queue_pointer);
  if (queue == nullptr || remote_endpoints == nullptr || endpoint_count == 0 ||
      !queue->dct_ready || queue->connected) {
    set_error(error, error_bytes, "invalid DC peer connection request");
    return -1;
  }
  queue->peers.resize(endpoint_count);
  ibv_ah_attr first_ah{};
  for (uint32_t index = 0; index < endpoint_count; ++index) {
    const HostSglEndpoint& endpoint = remote_endpoints[index];
    if (endpoint.qp_num == 0 || endpoint.active_mtu < IBV_MTU_256 ||
        endpoint.active_mtu > IBV_MTU_4096) {
      set_error(error, error_bytes, "invalid DCT endpoint %u qpn=%u mtu=%u",
                index, endpoint.qp_num, endpoint.active_mtu);
      return -1;
    }
    ibv_ah_attr ah{};
    fill_ah_attr(queue, endpoint, &ah);
    queue->peers[index].ah = ibv_create_ah(queue->pd, &ah);
    queue->peers[index].dctn = endpoint.qp_num;
    if (queue->peers[index].ah == nullptr) {
      set_error(error, error_bytes, "ibv_create_ah(%u) failed: %s", index,
                std::strerror(errno));
      return -1;
    }
    if (index == 0)
      first_ah = ah;
  }
  ibv_qp_attr rtr{};
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu = static_cast<ibv_mtu>(std::min<int>(
      queue->port_attr.active_mtu, remote_endpoints[0].active_mtu));
  rtr.ah_attr = first_ah;
  int result = ibv_modify_qp(
      queue->dci, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU);
  if (result != 0) {
    set_error(error, error_bytes, "DCI INIT->RTR failed: %s",
              std::strerror(result));
    return -1;
  }
  ibv_qp_attr rts{};
  rts.qp_state = IBV_QPS_RTS;
  rts.timeout = 14;
  rts.retry_cnt = 7;
  rts.rnr_retry = 7;
  rts.sq_psn = queue->psn;
  rts.max_rd_atomic = 1;
  result = ibv_modify_qp(
      queue->dci, &rts,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (result != 0) {
    set_error(error, error_bytes, "DCI RTR->RTS failed: %s",
              std::strerror(result));
    return -1;
  }
  queue->connected = true;
  return 0;
}

uint32_t host_sgl_dc_max_send_wr(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglDcQueue*>(queue_pointer);
  return queue == nullptr ? 0 : queue->max_send_wr;
}

uint32_t host_sgl_dc_max_sge(void* queue_pointer) {
  const auto* queue = static_cast<const HostSglDcQueue*>(queue_pointer);
  return queue == nullptr ? 0 : queue->max_sge;
}

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
  const size_t allocation_bytes = indices_offset + indices_bytes;
  if (posix_memalign(
          &ring->allocation, slot_alignment, allocation_bytes) != 0) {
    set_error(error, error_bytes, "aligned ring malloc failed");
    delete ring;
    return nullptr;
  }
  std::memset(ring->allocation, 0, allocation_bytes);
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

int host_sgl_consume_ring_group(
    void* const* queue_pointers, void* const* ring_pointers,
    uint32_t ring_count, const uint64_t* first_generations,
    const uint32_t* request_counts, uint32_t* posted_data_wrs,
    char* error, size_t error_bytes) {
  constexpr uint32_t max_ring_group = 32;
  if (queue_pointers == nullptr || ring_pointers == nullptr ||
      first_generations == nullptr || request_counts == nullptr ||
      ring_count == 0 || ring_count > max_ring_group) {
    set_error(error, error_bytes, "invalid coherent-ring group request");
    return -1;
  }

  struct RingProgress {
    HostSglQueue* queue = nullptr;
    HostSglRingMemory* memory = nullptr;
    HostSglRingSlot* slots = nullptr;
    uint64_t next_generation = 0;
    uint64_t last_generation = 0;
    uint64_t last_completed = 0;
    uint64_t data_wrs = 0;
    bool done = false;
  };
  std::array<RingProgress, max_ring_group> states{};
  for (uint32_t index = 0; index < ring_count; ++index) {
    auto* queue = static_cast<HostSglQueue*>(queue_pointers[index]);
    auto* ring = static_cast<HostSglRing*>(ring_pointers[index]);
    if (queue == nullptr || ring == nullptr || ring->memory == nullptr ||
        first_generations[index] == 0 || request_counts[index] == 0 ||
        request_counts[index] > hostSglRingMaxMessages) {
      set_error(error, error_bytes,
                "invalid coherent ring at group index %u", index);
      return -1;
    }
    if (!queue->connected || !queue->outstanding_batches.empty()) {
      set_error(error, error_bytes,
                "ring %u requires one connected idle queue", index);
      return -1;
    }
    HostSglRingMemory* memory = ring->memory;
    if (memory->abi_version != 1 ||
        memory->capacity != hostSglRingCapacity ||
        memory->max_rows != hostSglRingMaxRows) {
      set_error(error, error_bytes,
                "coherent ring %u header is invalid", index);
      return -1;
    }
    const uint64_t last_generation =
        first_generations[index] + request_counts[index] - 1;
    if (last_generation < first_generations[index]) {
      set_error(error, error_bytes,
                "coherent ring %u generation range overflows", index);
      return -1;
    }
    states[index] = RingProgress{
        queue,
        memory,
        reinterpret_cast<HostSglRingSlot*>(memory->slots_address),
        first_generations[index],
        last_generation,
        first_generations[index] - 1,
        0,
        false};
    if (posted_data_wrs != nullptr)
      posted_data_wrs[index] = 0;
  }

  std::array<HostSglRequest, hostSglRingBatch> requests{};
  uint32_t completed_rings = 0;
  while (completed_rings < ring_count) {
    bool made_progress = false;
    for (uint32_t index = 0; index < ring_count; ++index) {
      RingProgress& state = states[index];
      if (state.done)
        continue;

      if (state.next_generation <= state.last_generation) {
        HostSglRingSlot& head = state.slots[
            (state.next_generation - 1) % state.memory->capacity];
        if (__atomic_load_n(&head.ready_generation, __ATOMIC_ACQUIRE) ==
            state.next_generation) {
          uint32_t batch_count = 0;
          uint64_t batch_end = state.next_generation;
          while (batch_end <= state.last_generation &&
                 batch_count < hostSglRingBatch) {
            HostSglRingSlot& slot = state.slots[
                (batch_end - 1) % state.memory->capacity];
            if (__atomic_load_n(
                    &slot.ready_generation, __ATOMIC_ACQUIRE) != batch_end)
              break;
            requests[batch_count++] = slot.request;
            ++batch_end;
          }

          uint64_t required_wrs = batch_count;
          for (uint32_t request = 0; request < batch_count; ++request) {
            required_wrs +=
                (requests[request].row_count + state.queue->max_sge - 1) /
                state.queue->max_sge;
          }
          if (required_wrs > state.queue->max_send_wr) {
            set_error(error, error_bytes,
                      "ring %u batch needs %llu WRs but QP holds %u", index,
                      static_cast<unsigned long long>(required_wrs),
                      state.queue->max_send_wr);
            return -1;
          }
          const bool has_batch_credit =
              state.queue->outstanding_batches.size() <
              state.queue->max_outstanding_batches;
          const bool has_wr_credit =
              required_wrs <= state.queue->max_send_wr -
                                  state.queue->outstanding_send_wrs;
          if (has_batch_credit && has_wr_credit) {
            uint32_t batch_data_wrs = 0;
            if (host_sgl_post_indexed_batch(
                    state.queue, requests.data(), batch_count, 0,
                    &batch_data_wrs, error, error_bytes) != 0) {
              return -1;
            }
            state.data_wrs += batch_data_wrs;
            for (uint64_t generation = state.next_generation;
                 generation < batch_end; ++generation) {
              HostSglRingSlot& slot = state.slots[
                  (generation - 1) % state.memory->capacity];
              __atomic_store_n(
                  &slot.consumed_generation, generation, __ATOMIC_RELEASE);
            }
            state.next_generation = batch_end;
            made_progress = true;
          }
        }
      }

      if (!state.queue->outstanding_batches.empty()) {
        uint64_t completed = 0;
        const int count = host_sgl_try_poll(
            state.queue, &completed, error, error_bytes);
        if (count < 0) {
          return -1;
        }
        if (count == 1) {
          state.last_completed = completed;
          made_progress = true;
        }
      }

      if (state.next_generation > state.last_generation &&
          state.queue->outstanding_batches.empty()) {
        if (state.last_completed != state.last_generation) {
          set_error(error, error_bytes,
                    "ring %u completed %llu instead of %llu", index,
                    static_cast<unsigned long long>(state.last_completed),
                    static_cast<unsigned long long>(state.last_generation));
          return -1;
        }
        if (state.data_wrs > UINT32_MAX) {
          set_error(error, error_bytes,
                    "ring %u data WR count overflows", index);
          return -1;
        }
        if (posted_data_wrs != nullptr)
          posted_data_wrs[index] = static_cast<uint32_t>(state.data_wrs);
        state.done = true;
        ++completed_rings;
        made_progress = true;
      }
    }
    if (!made_progress)
      asm volatile("" ::: "memory");
  }
  return 0;
}

static int host_sgl_post_dc_batch(
    HostSglDcQueue* queue, const HostSglRequest* requests,
    uint32_t request_count, uint32_t peer_index, uint32_t ring_index,
    uint32_t* posted_data_wrs, char* error, size_t error_bytes) {
  if (queue == nullptr || !queue->connected || requests == nullptr ||
      request_count == 0 || peer_index >= queue->peers.size()) {
    set_error(error, error_bytes, "invalid DC indexed-row batch");
    return -1;
  }
  uint64_t total_rows = 0;
  uint64_t data_wr_count = 0;
  for (uint32_t index = 0; index < request_count; ++index) {
    const HostSglRequest& request = requests[index];
    if (request.row_indices == nullptr || request.row_count == 0 ||
        request.row_bytes == 0 || request.source_stride < request.row_bytes ||
        request.remote_data == 0 || request.remote_signal == 0 ||
        request.sequence == 0) {
      set_error(error, error_bytes, "invalid DC request at batch index %u",
                index);
      return -1;
    }
    total_rows += request.row_count;
    data_wr_count +=
        (request.row_count + queue->max_sge - 1) / queue->max_sge;
  }
  const uint64_t total_wr_count = data_wr_count + request_count;
  if (total_wr_count > queue->max_send_wr - queue->outstanding_send_wrs ||
      queue->outstanding_batches.size() >= 64 ||
      total_rows > SIZE_MAX || total_wr_count > UINT32_MAX) {
    set_error(error, error_bytes, "DC send-queue credits are exhausted");
    return -1;
  }

  std::vector<ibv_sge> sges(static_cast<size_t>(total_rows));
  size_t sge_cursor = 0;
  const uint64_t completion_id = queue->next_completion_id++;
  const HostSglDcPeer& peer = queue->peers[peer_index];
  ibv_wr_start(queue->dci_ex);
  uint64_t emitted_wrs = 0;
  for (uint32_t request_index = 0; request_index < request_count;
       ++request_index) {
    const HostSglRequest& request = requests[request_index];
    const size_t row_sge_base = sge_cursor;
    for (uint32_t row = 0; row < request.row_count; ++row) {
      ibv_sge& sge = sges[sge_cursor++];
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
          std::min(queue->max_sge, request.row_count - first_row);
      queue->dci_ex->wr_id = 0;
      queue->dci_ex->wr_flags = 0;
      ibv_wr_rdma_write(
          queue->dci_ex, request.remote_rkey,
          request.remote_data + remote_offset);
      mlx5dv_wr_set_dc_addr(
          queue->mlx5_dci_ex, peer.ah, peer.dctn, hostSglDcKey);
      ibv_wr_set_sge_list(
          queue->dci_ex, rows, &sges[row_sge_base + first_row]);
      first_row += rows;
      remote_offset += static_cast<uint64_t>(rows) * request.row_bytes;
      ++emitted_wrs;
    }
    uint64_t signal = request.sequence;
    ++emitted_wrs;
    queue->dci_ex->wr_id =
        emitted_wrs == total_wr_count ? completion_id : 0;
    queue->dci_ex->wr_flags = IBV_SEND_INLINE |
        (emitted_wrs == total_wr_count ? IBV_SEND_SIGNALED : 0);
    ibv_wr_rdma_write(
        queue->dci_ex, request.remote_rkey, request.remote_signal);
    mlx5dv_wr_set_dc_addr(
        queue->mlx5_dci_ex, peer.ah, peer.dctn, hostSglDcKey);
    ibv_wr_set_inline_data(queue->dci_ex, &signal, sizeof(signal));
  }
  const int result = ibv_wr_complete(queue->dci_ex);
  if (result != 0) {
    set_error(error, error_bytes, "ibv_wr_complete(DCI) failed: %s",
              std::strerror(result));
    return -1;
  }
  queue->outstanding_send_wrs += static_cast<uint32_t>(total_wr_count);
  queue->outstanding_batches.push_back(HostSglDcPostedBatch{
      completion_id, static_cast<uint32_t>(total_wr_count), ring_index});
  if (posted_data_wrs != nullptr)
    *posted_data_wrs = static_cast<uint32_t>(data_wr_count);
  return 0;
}

static int host_sgl_try_poll_dc(HostSglDcQueue* queue,
                                uint32_t* completed_ring,
                                char* error, size_t error_bytes) {
  if (queue == nullptr || queue->outstanding_batches.empty())
    return 0;
  ibv_wc completion{};
  const int count = ibv_poll_cq(queue->cq, 1, &completion);
  if (count < 0) {
    set_error(error, error_bytes, "ibv_poll_cq(DCI) failed");
    return -1;
  }
  if (count == 0)
    return 0;
  const HostSglDcPostedBatch& expected = queue->outstanding_batches.front();
  if (completion.status != IBV_WC_SUCCESS ||
      completion.wr_id != expected.completion_id ||
      queue->outstanding_send_wrs < expected.send_wr_count) {
    set_error(error, error_bytes,
              "DCI completion failed: %s id=%llu expected=%llu vendor=%u",
              ibv_wc_status_str(completion.status),
              static_cast<unsigned long long>(completion.wr_id),
              static_cast<unsigned long long>(expected.completion_id),
              completion.vendor_err);
    return -1;
  }
  queue->outstanding_send_wrs -= expected.send_wr_count;
  if (completed_ring != nullptr)
    *completed_ring = expected.ring_index;
  queue->outstanding_batches.pop_front();
  return 1;
}

int host_sgl_consume_ring_epoch_dc(
    void* queue_pointer, void* const* ring_pointers, uint32_t ring_count,
    uint32_t* posted_data_wrs, char* error, size_t error_bytes) {
  constexpr uint32_t max_ring_group = 32;
  auto* queue = static_cast<HostSglDcQueue*>(queue_pointer);
  if (queue == nullptr || ring_pointers == nullptr || ring_count == 0 ||
      ring_count > max_ring_group || !queue->connected ||
      queue->peers.size() != ring_count ||
      !queue->outstanding_batches.empty()) {
    set_error(error, error_bytes, "invalid coherent-ring DC epoch group");
    return -1;
  }
  struct RingProgress {
    HostSglRing* ring = nullptr;
    HostSglRingMemory* memory = nullptr;
    HostSglRingSlot* slots = nullptr;
    uint64_t next_generation = 0;
    uint64_t data_wrs = 0;
    uint32_t outstanding_batches = 0;
    bool end_seen = false;
    bool done = false;
  };
  std::array<RingProgress, max_ring_group> states{};
  for (uint32_t index = 0; index < ring_count; ++index) {
    auto* ring = static_cast<HostSglRing*>(ring_pointers[index]);
    if (ring == nullptr || ring->memory == nullptr ||
        ring->memory->abi_version != 1 ||
        ring->memory->capacity != hostSglRingCapacity ||
        ring->memory->max_rows != hostSglRingMaxRows) {
      set_error(error, error_bytes, "coherent DC ring %u is invalid", index);
      return -1;
    }
    states[index] = RingProgress{
        ring,
        ring->memory,
        reinterpret_cast<HostSglRingSlot*>(ring->memory->slots_address),
        ring->next_generation,
        0,
        0,
        false,
        false};
    if (posted_data_wrs != nullptr)
      posted_data_wrs[index] = 0;
  }

  std::array<HostSglRequest, hostSglRingBatch> requests{};
  uint32_t completed_rings = 0;
  while (completed_rings < ring_count) {
    bool made_progress = false;
    for (uint32_t index = 0; index < ring_count; ++index) {
      RingProgress& state = states[index];
      if (state.done || state.end_seen)
        continue;
      HostSglRingSlot& head = state.slots[
          (state.next_generation - 1) % state.memory->capacity];
      if (__atomic_load_n(&head.ready_generation, __ATOMIC_ACQUIRE) !=
          state.next_generation)
        continue;
      if (head.request.row_count == 0) {
        __atomic_store_n(
            &head.consumed_generation, state.next_generation,
            __ATOMIC_RELEASE);
        ++state.next_generation;
        state.end_seen = true;
        made_progress = true;
        continue;
      }
      uint32_t batch_count = 0;
      uint64_t batch_end = state.next_generation;
      while (batch_count < hostSglRingBatch) {
        HostSglRingSlot& slot = state.slots[
            (batch_end - 1) % state.memory->capacity];
        if (__atomic_load_n(&slot.ready_generation, __ATOMIC_ACQUIRE) !=
                batch_end ||
            slot.request.row_count == 0)
          break;
        requests[batch_count++] = slot.request;
        ++batch_end;
      }
      uint64_t required_wrs = batch_count;
      for (uint32_t request = 0; request < batch_count; ++request) {
        required_wrs +=
            (requests[request].row_count + queue->max_sge - 1) /
            queue->max_sge;
      }
      if (required_wrs <= queue->max_send_wr - queue->outstanding_send_wrs &&
          queue->outstanding_batches.size() < 64) {
        uint32_t batch_data_wrs = 0;
        if (host_sgl_post_dc_batch(
                queue, requests.data(), batch_count, index, index,
                &batch_data_wrs, error, error_bytes) != 0)
          return -1;
        state.data_wrs += batch_data_wrs;
        ++state.outstanding_batches;
        for (uint64_t generation = state.next_generation;
             generation < batch_end; ++generation) {
          HostSglRingSlot& slot = state.slots[
              (generation - 1) % state.memory->capacity];
          __atomic_store_n(
              &slot.consumed_generation, generation, __ATOMIC_RELEASE);
        }
        state.next_generation = batch_end;
        made_progress = true;
      }
    }

    uint32_t completed_ring = 0;
    const int completion = host_sgl_try_poll_dc(
        queue, &completed_ring, error, error_bytes);
    if (completion < 0)
      return -1;
    if (completion == 1) {
      if (completed_ring >= ring_count ||
          states[completed_ring].outstanding_batches == 0) {
        set_error(error, error_bytes, "DC ring completion accounting failed");
        return -1;
      }
      --states[completed_ring].outstanding_batches;
      made_progress = true;
    }
    for (uint32_t index = 0; index < ring_count; ++index) {
      RingProgress& state = states[index];
      if (!state.done && state.end_seen && state.outstanding_batches == 0) {
        if (state.data_wrs > UINT32_MAX) {
          set_error(error, error_bytes, "DC ring %u WR count overflows", index);
          return -1;
        }
        state.ring->next_generation = state.next_generation;
        if (posted_data_wrs != nullptr)
          posted_data_wrs[index] = static_cast<uint32_t>(state.data_wrs);
        state.done = true;
        ++completed_rings;
        made_progress = true;
      }
    }
    if (!made_progress)
      asm volatile("" ::: "memory");
  }
  return 0;
}

int host_sgl_consume_ring_epoch_group(
    void* const* queue_pointers, void* const* ring_pointers,
    uint32_t ring_count, uint32_t* posted_data_wrs,
    char* error, size_t error_bytes) {
  constexpr uint32_t max_ring_group = 32;
  if (queue_pointers == nullptr || ring_pointers == nullptr ||
      ring_count == 0 || ring_count > max_ring_group) {
    set_error(error, error_bytes, "invalid coherent-ring epoch group");
    return -1;
  }

  struct RingProgress {
    HostSglQueue* queue = nullptr;
    HostSglRing* ring = nullptr;
    HostSglRingMemory* memory = nullptr;
    HostSglRingSlot* slots = nullptr;
    uint64_t next_generation = 0;
    uint64_t data_wrs = 0;
    bool end_seen = false;
    bool done = false;
  };
  std::array<RingProgress, max_ring_group> states{};
  for (uint32_t index = 0; index < ring_count; ++index) {
    auto* queue = static_cast<HostSglQueue*>(queue_pointers[index]);
    auto* ring = static_cast<HostSglRing*>(ring_pointers[index]);
    if (queue == nullptr || ring == nullptr || ring->memory == nullptr ||
        !queue->connected || !queue->outstanding_batches.empty()) {
      set_error(error, error_bytes,
                "epoch ring %u requires one connected idle queue", index);
      return -1;
    }
    HostSglRingMemory* memory = ring->memory;
    if (memory->abi_version != 1 ||
        memory->capacity != hostSglRingCapacity ||
        memory->max_rows != hostSglRingMaxRows) {
      set_error(error, error_bytes,
                "coherent epoch ring %u header is invalid", index);
      return -1;
    }
    states[index] = RingProgress{
        queue,
        ring,
        memory,
        reinterpret_cast<HostSglRingSlot*>(memory->slots_address),
        ring->next_generation,
        0,
        false,
        false};
    if (posted_data_wrs != nullptr)
      posted_data_wrs[index] = 0;
  }

  std::array<HostSglRequest, hostSglRingBatch> requests{};
  uint32_t completed_rings = 0;
  while (completed_rings < ring_count) {
    bool made_progress = false;
    for (uint32_t index = 0; index < ring_count; ++index) {
      RingProgress& state = states[index];
      if (state.done)
        continue;

      if (!state.end_seen) {
        HostSglRingSlot& head = state.slots[
            (state.next_generation - 1) % state.memory->capacity];
        if (__atomic_load_n(&head.ready_generation, __ATOMIC_ACQUIRE) ==
            state.next_generation) {
          if (head.request.row_count == 0) {
            __atomic_store_n(
                &head.consumed_generation,
                state.next_generation,
                __ATOMIC_RELEASE);
            ++state.next_generation;
            state.end_seen = true;
            made_progress = true;
          } else {
            uint32_t batch_count = 0;
            uint64_t batch_end = state.next_generation;
            while (batch_count < hostSglRingBatch) {
              HostSglRingSlot& slot = state.slots[
                  (batch_end - 1) % state.memory->capacity];
              if (__atomic_load_n(
                      &slot.ready_generation, __ATOMIC_ACQUIRE) != batch_end ||
                  slot.request.row_count == 0)
                break;
              requests[batch_count++] = slot.request;
              ++batch_end;
            }

            uint64_t required_wrs = batch_count;
            for (uint32_t request = 0; request < batch_count; ++request) {
              required_wrs +=
                  (requests[request].row_count + state.queue->max_sge - 1) /
                  state.queue->max_sge;
            }
            if (required_wrs > state.queue->max_send_wr) {
              set_error(error, error_bytes,
                        "epoch ring %u batch needs %llu WRs but QP holds %u",
                        index,
                        static_cast<unsigned long long>(required_wrs),
                        state.queue->max_send_wr);
              return -1;
            }
            const bool has_batch_credit =
                state.queue->outstanding_batches.size() <
                state.queue->max_outstanding_batches;
            const bool has_wr_credit =
                required_wrs <= state.queue->max_send_wr -
                                    state.queue->outstanding_send_wrs;
            if (has_batch_credit && has_wr_credit) {
              uint32_t batch_data_wrs = 0;
              if (host_sgl_post_indexed_batch(
                      state.queue, requests.data(), batch_count, 0,
                      &batch_data_wrs, error, error_bytes) != 0) {
                return -1;
              }
              state.data_wrs += batch_data_wrs;
              for (uint64_t generation = state.next_generation;
                   generation < batch_end; ++generation) {
                HostSglRingSlot& slot = state.slots[
                    (generation - 1) % state.memory->capacity];
                __atomic_store_n(
                    &slot.consumed_generation, generation, __ATOMIC_RELEASE);
              }
              state.next_generation = batch_end;
              made_progress = true;
            }
          }
        }
      }

      if (!state.queue->outstanding_batches.empty()) {
        const int count = host_sgl_try_poll(
            state.queue, nullptr, error, error_bytes);
        if (count < 0)
          return -1;
        made_progress |= count == 1;
      }

      if (state.end_seen && state.queue->outstanding_batches.empty()) {
        if (state.data_wrs > UINT32_MAX) {
          set_error(error, error_bytes,
                    "epoch ring %u data WR count overflows", index);
          return -1;
        }
        state.ring->next_generation = state.next_generation;
        if (posted_data_wrs != nullptr)
          posted_data_wrs[index] = static_cast<uint32_t>(state.data_wrs);
        state.done = true;
        ++completed_rings;
        made_progress = true;
      }
    }
    if (!made_progress)
      asm volatile("" ::: "memory");
  }
  return 0;
}

int host_sgl_consume_ring(void* queue_pointer, void* ring_pointer,
                          uint64_t first_generation, uint32_t request_count,
                          uint32_t* posted_data_wrs,
                          char* error, size_t error_bytes) {
  void* queues[] = {queue_pointer};
  void* rings[] = {ring_pointer};
  return host_sgl_consume_ring_group(
      queues, rings, 1, &first_generation, &request_count,
      posted_data_wrs, error, error_bytes);
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

void host_sgl_destroy_dc(void* queue_pointer) {
  destroy_dc_queue(static_cast<HostSglDcQueue*>(queue_pointer));
}

}  // extern "C"
