#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>

#include "nvshmem.h"
#include "nvshmemx.h"

#define CUDA_CHECK(stmt) do {                                             \
    cudaError_t e = (stmt);                                               \
    if (e != cudaSuccess) {                                               \
        fprintf(stderr, "[%s:%d] CUDA error: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e));               \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

#define MPI_CHECK(stmt) do {                                              \
    int e = (stmt);                                                       \
    if (e != MPI_SUCCESS) {                                               \
        fprintf(stderr, "[%s:%d] MPI error\n", __FILE__, __LINE__);       \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

static size_t mib_to_bytes(size_t mib) {
    return mib << 20;
}

__global__ void init_buf(char *src, char *dst, size_t nbytes) {
    int pe = nvshmem_my_pe();
    size_t tid = threadIdx.x + (size_t) blockIdx.x * blockDim.x;
    size_t stride = (size_t) blockDim.x * gridDim.x;

    for (size_t i = tid; i < nbytes; i += stride) {
        src[i] = (char)(pe & 0x7f);
        dst[i] = 0;
    }
}

__global__ void ring_put_kernel(char *dst,
                                const char *src,
                                size_t nbytes,
                                int iters) {
    int pe = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (pe + 1) % npes;

    // Timed path: normal NVSHMEM device-side API only.
    // IBGDA is selected by runtime env vars, not a different public API name.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < iters; i++) {
            nvshmem_char_put_nbi(dst, src, nbytes, peer);
            nvshmem_quiet();
        }
    }
}

int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0, nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (nranks < 2) {
        if (rank == 0) {
            fprintf(stderr, "Need at least 2 MPI ranks/PEs. On TACC Vista GH: idev -p gh-dev -N 2 -n 2 -t 01:00:00\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Args: ./main [chunk_mib] [iters]
    size_t chunk_mib = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 128;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 50;

    if (chunk_mib == 0 || iters <= 0) {
        if (rank == 0) {
            fprintf(stderr, "usage: %s [chunk_mib>0] [iters>0]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nbytes = mib_to_bytes(chunk_mib);

    // Set NVSHMEM runtime config before nvshmemx_init_attr().
    size_t heap_mib = std::max<size_t>(512, chunk_mib * 3);
    char heap_env[64];
    std::snprintf(heap_env, sizeof(heap_env), "%zuM", heap_mib);
    setenv("NVSHMEM_SYMMETRIC_SIZE", heap_env, 0);

    setenv("NVSHMEM_BOOTSTRAP", "MPI", 0);
    setenv("NVSHMEM_REMOTE_TRANSPORT", "ibrc", 0);
    setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 0);
    setenv("NVSHMEM_IBGDA_NIC_HANDLER", "gpu", 0);

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD,
                                  MPI_COMM_TYPE_SHARED,
                                  rank,
                                  MPI_INFO_NULL,
                                  &local_comm));

    int local_rank = 0, local_size = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));

    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    if (local_size > ngpus) {
        fprintf(stderr,
                "rank %d: local MPI ranks (%d) > local GPUs (%d). "
                "Use one MPI rank per GPU.\n",
                rank, local_size, ngpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    CUDA_CHECK(cudaSetDevice(local_rank % ngpus));

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;

    int init_ret = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (init_ret != 0) {
        fprintf(stderr, "rank %d: nvshmemx_init_attr failed: %d\n", rank, init_ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int pe = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    char *src = (char *)nvshmem_malloc(nbytes);
    char *dst = (char *)nvshmem_malloc(nbytes);
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed for two %zu MiB buffers\n",
                pe, chunk_mib);
        nvshmem_global_exit(1);
    }

    init_buf<<<256, 256>>>(src, dst, nbytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    ring_put_kernel<<<1, 1>>>(dst, src, nbytes, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float local_ms_float = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&local_ms_float, start, stop));

    nvshmem_barrier_all();

    char first = 0, last = 0;
    CUDA_CHECK(cudaMemcpy(&first, dst, 1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last, dst + nbytes - 1, 1, cudaMemcpyDeviceToHost));

    int sender = (pe - 1 + npes) % npes;
    int local_ok = ((((int)first & 0x7f) == (sender & 0x7f)) &&
                    (((int)last  & 0x7f) == (sender & 0x7f))) ? 1 : 0;

    int all_ok = 0;
    double local_ms = (double)local_ms_float;
    double max_ms = 0.0;

    MPI_CHECK(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    double sec = max_ms / 1.0e3;
    double one_way_gib_per_pe = ((double)nbytes * iters) / (1024.0 * 1024.0 * 1024.0);
    double per_pe_gib_s = one_way_gib_per_pe / sec;
    double aggregate_gib_s = per_pe_gib_s * npes;

    if (pe == 0) {
        printf("NVSHMEM normal-API IBGDA ring-put: pes=%d chunk=%zu MiB iters=%d "
               "max_time=%.3f ms per_pe_data=%.2f GiB per_pe_bw=%.2f GiB/s "
               "aggregate_bw=%.2f GiB/s %s\n",
               npes, chunk_mib, iters, max_ms,
               one_way_gib_per_pe, per_pe_gib_s, aggregate_gib_s,
               all_ok ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    nvshmem_free(src);
    nvshmem_free(dst);

    nvshmem_finalize();

    MPI_CHECK(MPI_Comm_free(&local_comm));
    MPI_CHECK(MPI_Finalize());

    return all_ok ? 0 : 1;
}

