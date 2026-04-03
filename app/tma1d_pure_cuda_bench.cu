#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {

constexpr int kThreadsPerBlock = 32;
constexpr int kStages = 4;
constexpr uint32_t kMinBlockBytes = 1u << 10;
constexpr uint32_t kMaxBlockBytes = 32u << 10;

using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t err__ = (expr);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err__));                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct Options {
  size_t working_set_bytes_per_sm = 8ull << 20;
  size_t target_bytes_per_sm = 256ull << 20;
  int warmup = 1;
  int repeats = 5;
};

template <typename T>
T ceil_div(T num, T den) {
  return (num + den - 1) / den;
}

size_t round_up(size_t value, size_t alignment) {
  return ceil_div(value, alignment) * alignment;
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--working-set-mb-per-sm") == 0 && i + 1 < argc) {
      opts.working_set_bytes_per_sm = std::strtoull(argv[++i], nullptr, 10) << 20;
    } else if (std::strcmp(argv[i], "--target-mb-per-sm") == 0 && i + 1 < argc) {
      opts.target_bytes_per_sm = std::strtoull(argv[++i], nullptr, 10) << 20;
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      opts.warmup = std::max(0, std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      opts.repeats = std::max(1, std::atoi(argv[++i]));
    } else {
      std::fprintf(stderr,
                   "usage: %s [--working-set-mb-per-sm MB] [--target-mb-per-sm MB] "
                   "[--warmup N] [--repeats N]\n",
                   argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  opts.working_set_bytes_per_sm =
      std::max<size_t>(opts.working_set_bytes_per_sm, kMaxBlockBytes);
  opts.working_set_bytes_per_sm =
      round_up(opts.working_set_bytes_per_sm, kMaxBlockBytes);
  opts.target_bytes_per_sm =
      std::max(opts.target_bytes_per_sm, opts.working_set_bytes_per_sm);
  return opts;
}

__device__ __forceinline__ uint8_t* align_smem_1kb(uint8_t* raw) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
  addr = (addr + 1023u) & ~uintptr_t(1023u);
  return reinterpret_cast<uint8_t*>(addr);
}

template <int Stages>
__device__ __forceinline__ const uint8_t* ptr_for_ticket(const uint8_t* base,
                                                         uint32_t ticket,
                                                         uint32_t tiles_per_pass,
                                                         uint32_t block_bytes,
                                                         uint32_t working_set_bytes) {
  return base + static_cast<uint64_t>(blockIdx.x) * working_set_bytes +
         static_cast<uint64_t>(ticket % tiles_per_pass) * block_bytes;
}

template <int Stages>
__device__ __forceinline__ uint8_t* ptr_for_ticket(uint8_t* base,
                                                   uint32_t ticket,
                                                   uint32_t tiles_per_pass,
                                                   uint32_t block_bytes,
                                                   uint32_t working_set_bytes) {
  return base + static_cast<uint64_t>(blockIdx.x) * working_set_bytes +
         static_cast<uint64_t>(ticket % tiles_per_pass) * block_bytes;
}

template <int Stages>
__device__ __forceinline__ void issue_load_ticket(const uint8_t* src,
                                                  BlockBarrier* barriers,
                                                  BlockBarrier::arrival_token* tokens,
                                                  uint8_t* smem,
                                                  uint32_t ticket,
                                                  uint32_t tiles_per_pass,
                                                  uint32_t block_bytes,
                                                  uint32_t working_set_bytes) {
  int stage = ticket % Stages;
  cuda::device::memcpy_async_tx(
      smem + stage * block_bytes,
      ptr_for_ticket<Stages>(src, ticket, tiles_per_pass, block_bytes, working_set_bytes),
      cuda::aligned_size_t<16>(block_bytes),
      barriers[stage]);
  cuda::device::barrier_expect_tx(
      barriers[stage], cuda::aligned_size_t<16>(block_bytes));
  tokens[stage] = barriers[stage].arrive();
}

template <int Pending>
__device__ __forceinline__ void wait_store_pending() {
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<Pending>{});
}

template <int Stages>
__global__ void bulk_read_1d_kernel(const uint8_t* src,
                                    uint32_t tiles_per_pass,
                                    uint32_t passes,
                                    uint32_t block_bytes,
                                    uint32_t working_set_bytes,
                                    uint32_t* sink) {
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier barriers[Stages];
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = align_smem_1kb(smem_raw);

  if (threadIdx.x < Stages) {
    init(&barriers[threadIdx.x], 1);
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  BlockBarrier::arrival_token tokens[Stages];
  uint32_t total_tiles = tiles_per_pass * passes;
  uint32_t prologue =
      total_tiles < static_cast<uint32_t>(Stages) ? total_tiles : static_cast<uint32_t>(Stages);

  for (uint32_t ticket = 0; ticket < prologue; ++ticket) {
    issue_load_ticket<Stages>(
        src, barriers, tokens, smem, ticket, tiles_per_pass, block_bytes, working_set_bytes);
  }

  uint32_t checksum = 0;
  for (uint32_t ticket = 0; ticket < total_tiles; ++ticket) {
    int stage = ticket % Stages;
    barriers[stage].wait(std::move(tokens[stage]));
    checksum ^= reinterpret_cast<uint32_t const*>(smem + stage * block_bytes)[0];

    uint32_t next_ticket = ticket + Stages;
    if (next_ticket < total_tiles) {
      issue_load_ticket<Stages>(src,
                                barriers,
                                tokens,
                                smem,
                                next_ticket,
                                tiles_per_pass,
                                block_bytes,
                                working_set_bytes);
    }
  }

  sink[blockIdx.x] = checksum;
}

template <int Stages>
__global__ void bulk_write_1d_kernel(uint8_t* dst,
                                     uint32_t tiles_per_pass,
                                     uint32_t passes,
                                     uint32_t block_bytes,
                                     uint32_t working_set_bytes) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = align_smem_1kb(smem_raw);

  for (uint32_t i = threadIdx.x; i < Stages * block_bytes; i += blockDim.x) {
    smem[i] = static_cast<uint8_t>((i + blockIdx.x) & 0xFF);
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  uint32_t total_tiles = tiles_per_pass * passes;
  for (uint32_t ticket = 0; ticket < total_tiles; ++ticket) {
    if (ticket >= Stages) {
      wait_store_pending<Stages - 1>();
    }

    int stage = ticket % Stages;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_global,
        cuda::ptx::space_shared,
        ptr_for_ticket<Stages>(dst, ticket, tiles_per_pass, block_bytes, working_set_bytes),
        smem + stage * block_bytes,
        block_bytes);
    cuda::ptx::cp_async_bulk_commit_group();
  }

  wait_store_pending<0>();
}

template <int Stages>
float run_read_case(const uint8_t* src,
                    int num_sms,
                    uint32_t tiles_per_pass,
                    uint32_t passes,
                    uint32_t block_bytes,
                    uint32_t working_set_bytes,
                    int warmup,
                    int repeats,
                    uint32_t* sink) {
  size_t dynamic_smem_bytes = static_cast<size_t>(Stages) * block_bytes + 1024;
  CHECK_CUDA(cudaFuncSetAttribute(
      bulk_read_1d_kernel<Stages>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_smem_bytes)));
  CHECK_CUDA(cudaFuncSetAttribute(
      bulk_read_1d_kernel<Stages>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  for (int i = 0; i < warmup; ++i) {
    bulk_read_1d_kernel<Stages><<<num_sms, kThreadsPerBlock, dynamic_smem_bytes>>>(
        src, tiles_per_pass, passes, block_bytes, working_set_bytes, sink);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  float total_ms = 0.0f;
  for (int i = 0; i < repeats; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    bulk_read_1d_kernel<Stages><<<num_sms, kThreadsPerBlock, dynamic_smem_bytes>>>(
        src, tiles_per_pass, passes, block_bytes, working_set_bytes, sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / repeats;
}

template <int Stages>
float run_write_case(uint8_t* dst,
                     int num_sms,
                     uint32_t tiles_per_pass,
                     uint32_t passes,
                     uint32_t block_bytes,
                     uint32_t working_set_bytes,
                     int warmup,
                     int repeats) {
  size_t dynamic_smem_bytes = static_cast<size_t>(Stages) * block_bytes + 1024;
  CHECK_CUDA(cudaFuncSetAttribute(
      bulk_write_1d_kernel<Stages>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_smem_bytes)));
  CHECK_CUDA(cudaFuncSetAttribute(
      bulk_write_1d_kernel<Stages>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  for (int i = 0; i < warmup; ++i) {
    bulk_write_1d_kernel<Stages><<<num_sms, kThreadsPerBlock, dynamic_smem_bytes>>>(
        dst, tiles_per_pass, passes, block_bytes, working_set_bytes);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  float total_ms = 0.0f;
  for (int i = 0; i < repeats; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    bulk_write_1d_kernel<Stages><<<num_sms, kThreadsPerBlock, dynamic_smem_bytes>>>(
        dst, tiles_per_pass, passes, block_bytes, working_set_bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / repeats;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts = parse_options(argc, argv);

  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
  if (props.major != 9 || props.minor != 0) {
    std::fprintf(stderr, "expected a Hopper-class GPU, got sm_%d%d\n", props.major, props.minor);
    return EXIT_FAILURE;
  }

  int num_sms = props.multiProcessorCount;
  size_t working_set_bytes_per_sm =
      round_up(opts.working_set_bytes_per_sm, kMaxBlockBytes);
  size_t total_buffer_bytes = working_set_bytes_per_sm * static_cast<size_t>(num_sms);

  uint8_t* d_src = nullptr;
  uint8_t* d_dst = nullptr;
  uint32_t* d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, total_buffer_bytes));
  CHECK_CUDA(cudaMalloc(&d_dst, total_buffer_bytes));
  CHECK_CUDA(cudaMalloc(&d_sink, static_cast<size_t>(num_sms) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_src, 0x5A, total_buffer_bytes));
  CHECK_CUDA(cudaMemset(d_dst, 0x00, total_buffer_bytes));
  CHECK_CUDA(cudaMemset(d_sink, 0x00, static_cast<size_t>(num_sms) * sizeof(uint32_t)));

  std::printf("Pure CUDA 1D async bulk sweep\n");
  std::printf("gpu=%s sms=%d stages=%d working_set_per_sm=%.1f MiB target_per_sm=%.1f MiB\n",
              props.name,
              num_sms,
              kStages,
              static_cast<double>(working_set_bytes_per_sm) / (1024.0 * 1024.0),
              static_cast<double>(opts.target_bytes_per_sm) / (1024.0 * 1024.0));
  std::printf("This matches the repo's current TmaLoad1D/TmaStore1D path: bulk global<->shared, not tensor-map descriptors.\n");
  std::printf("blocks=%d threads_per_block=%d warmup=%d repeats=%d\n",
              num_sms,
              kThreadsPerBlock,
              opts.warmup,
              opts.repeats);
  std::printf("\n%-8s %-8s %-10s %-10s %-12s\n",
              "dir",
              "blockKB",
              "tiles/sm",
              "passes",
              "GB/s");

  for (uint32_t block_bytes = kMinBlockBytes; block_bytes <= kMaxBlockBytes;
       block_bytes <<= 1) {
    uint32_t tiles_per_pass =
        static_cast<uint32_t>(working_set_bytes_per_sm / block_bytes);
    uint32_t passes = static_cast<uint32_t>(
        ceil_div(opts.target_bytes_per_sm, working_set_bytes_per_sm));
    double measured_bytes =
        static_cast<double>(num_sms) * working_set_bytes_per_sm * passes;

    float read_ms = run_read_case<kStages>(d_src,
                                           num_sms,
                                           tiles_per_pass,
                                           passes,
                                           block_bytes,
                                           static_cast<uint32_t>(working_set_bytes_per_sm),
                                           opts.warmup,
                                           opts.repeats,
                                           d_sink);
    double read_gbps = measured_bytes / (read_ms * 1.0e6);
    std::printf("%-8s %-8u %-10u %-10u %-12.2f\n",
                "read",
                block_bytes / 1024,
                tiles_per_pass,
                passes,
                read_gbps);

    float write_ms = run_write_case<kStages>(d_dst,
                                             num_sms,
                                             tiles_per_pass,
                                             passes,
                                             block_bytes,
                                             static_cast<uint32_t>(working_set_bytes_per_sm),
                                             opts.warmup,
                                             opts.repeats);
    double write_gbps = measured_bytes / (write_ms * 1.0e6);
    std::printf("%-8s %-8u %-10u %-10u %-12.2f\n",
                "write",
                block_bytes / 1024,
                tiles_per_pass,
                passes,
                write_gbps);
  }

  CHECK_CUDA(cudaFree(d_sink));
  CHECK_CUDA(cudaFree(d_dst));
  CHECK_CUDA(cudaFree(d_src));
  return 0;
}
