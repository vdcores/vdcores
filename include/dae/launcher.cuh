#pragma once

#include "runtime.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <array>

static inline uint16_t nslot(const size_t bytes) {
  auto slots = (bytes + slotSizeKb * 1024 - 1) / (slotSizeKb * 1024);
  assert(slots < numSlots && "Request exceeds maximum number of slots");
  return static_cast<uint16_t>(slots);
}

static inline int16_t to_int16(int x) {
  if (x < std::numeric_limits<int16_t>::min() ||
    x > std::numeric_limits<int16_t>::max()) {
    throw std::overflow_error("int → int16_t overflow");
  }
  return static_cast<int16_t>(x);
}

static inline uint16_t to_uint16(int x) {
  if (x < 0 || x > std::numeric_limits<uint16_t>::max()) {
    throw std::overflow_error("Coordinate out of range for uint16_t (must be 0-65535)");
  }
  return static_cast<uint16_t>(x);
}


template<typename T>
static inline uint64_t make_cord(T x) {
  return static_cast<uint64_t>(to_uint16(x));
}

template<typename T1, typename T2>
static inline uint64_t make_cord(T1 x, T2 y) {
  return static_cast<uint64_t>(to_uint16(x)) |
         (static_cast<uint64_t>(to_uint16(y)) << 16);
}

template<typename T1, typename T2, typename T3>
static inline uint64_t make_cord(T1 x, T2 y, T3 z) {
  return static_cast<uint64_t>(to_uint16(x)) |
         (static_cast<uint64_t>(to_uint16(y)) << 16) |
         (static_cast<uint64_t>(to_uint16(z)) << 32);
}

template<typename T1, typename T2, typename T3, typename T4>
static inline uint64_t make_cord(T1 x, T2 y, T3 z, T4 w) {
  return static_cast<uint64_t>(to_uint16(x)) |
         (static_cast<uint64_t>(to_uint16(y)) << 16) |
         (static_cast<uint64_t>(to_uint16(z)) << 32) |
         (static_cast<uint64_t>(to_uint16(w)) << 48);
}

struct CUDARAIIBuilder {
  void * d_data_ = nullptr;
  ~CUDARAIIBuilder() {
      cudaFree(d_data_);
  }
 protected:
  template<typename T>
  T* allocate_and_copy(void *h_data, size_t num_elements) {
    cudaMalloc(&d_data_, sizeof(T) * num_elements);
    cudaMemcpy(d_data_, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice);
    return static_cast<T*>(d_data_);
  }
};

struct PerSMRAIIBuilder : CUDARAIIBuilder {
  int num_sms_;
  size_t count_per_sm_;
  PerSMRAIIBuilder(int num_sms, size_t count_per_sm)
    : num_sms_(num_sms), count_per_sm_(count_per_sm) {}
};

struct SMDataBuilder : PerSMRAIIBuilder {
  void * h_data_;
  SMDataBuilder(int num_sms, size_t sm_size)
  : PerSMRAIIBuilder(num_sms, sm_size) {
    h_data_ = malloc(num_sms * sm_size);
    auto err = cudaMalloc(&d_data_, sm_size * num_sms);
    assert(err == cudaSuccess && "Failed to allocate SM data on device");
  }
  ~SMDataBuilder() { free(h_data_); }

  template<typename T = char>
  T* d_get(int sm_id = 0, size_t offset = 0) {
    assert(offset + sizeof(T) <= count_per_sm_ && "Offset out of bounds for per-SM access");
    return (T*)((char*)d_data_ + sm_id * count_per_sm_ + offset);
  }

  template<typename T = char>
  T* h_get(int sm_id = 0, size_t offset = 0) {
    assert(offset + sizeof(T) <= count_per_sm_ && "Offset out of bounds for per-SM access");
    return (T*)((char*)h_data_ + sm_id * count_per_sm_ + offset);
  }
  
  void* copy_to_device() {
    cudaMemcpy(d_data_, h_data_, num_sms_ * count_per_sm_, cudaMemcpyHostToDevice);
    return d_data_;
  }
  void *copy_to_host() {
    cudaMemcpy(h_data_, d_data_, num_sms_ * count_per_sm_, cudaMemcpyDeviceToHost);
    return h_data_;
  }
};

struct TMABuilder : CUDARAIIBuilder {
  std::vector<CUtensorMap> tma_descs;

  // by default this constructs a tiled tensor map, no swizzle
  // For N-D tensor with dims [d0, d1, d2, ...] from fastest to slowest moving:
  uint16_t add (
    CUtensorMapDataType data_type,
    int dims,
    void * base,
    std::array<uint64_t, 5> global_dims,
    std::array<uint32_t, 5> box_dims,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
    std::array<uint64_t, 5> global_strides_opt = {0, 0, 0, 0, 0}
  ) {

    auto desc = create_tma_descriptor(
      data_type,
      dims,
      base,
      std::move(global_dims),
      std::move(box_dims),
      swizzle,
      std::move(global_strides_opt)
    );

    tma_descs.push_back(desc);
    return (uint16_t)(tma_descs.size() - 1);
  }

  CUtensorMap* copy_to_device() {
    return allocate_and_copy<CUtensorMap>(tma_descs.data(), tma_descs.size());
  }
};

template<typename T>
struct ProfileBuilder : PerSMRAIIBuilder {
    using PerSMRAIIBuilder::PerSMRAIIBuilder;

    T* copy_to_device() {
        cudaMalloc(&d_data_, num_sms_ * count_per_sm_ * sizeof(T));
        return (T*)d_data_;
    }
    
    T* d_get(int sm) {
        return (T*)d_data_ + sm * count_per_sm_;
    }
};

static CInst cinst(uint16_t opcode, std::array<uint16_t, 3> args = {0,0,0}) {
  CInst inst;
  inst.opcode = opcode;
  for (int i = 0; i < 3; i++)
    inst.args[i] = args[i];
  return inst;
}

static MInst minst(uint16_t opcode, uint16_t num_slots = 0, uint64_t address = 0, uint16_t size = 0, uint16_t arg = 0) {
  MInst inst;
  inst.opcode = opcode;
  inst.num_slots = num_slots;
  inst.address = address;
  inst.size = size;
  inst.arg = arg;
  return inst;
}

template <typename T, int MAX_INSTS = numInsts>
struct InstructionBuilder : PerSMRAIIBuilder {
  std::vector<T> instructions[256]; // max 256 SMs

  InstructionBuilder(int num_sms)
    : PerSMRAIIBuilder(num_sms, MAX_INSTS) {}

  template<typename F>
  void add(F && func) {
    for (int sm = 0; sm < num_sms_; sm++) {
      add(sm, func(sm));
    }
  }
  void add(int sm, T inst) {
    assert(sm >= 0 && sm < num_sms_ && "SM index out of range");
    instructions[sm].push_back(std::move(inst));
    assert(instructions[sm].size() <= MAX_INSTS && "Too many instructions added");
  }
  void add(T inst) {
    for (int sm = 0; sm < num_sms_; sm++) {
      add(sm, inst);
    }
  }

  T * copy_to_device() {
    cudaMalloc(&d_data_, sizeof(T) * MAX_INSTS * num_sms_);
    for (int sm = 0; sm < num_sms_; sm++) {
      cudaMemcpy(
        (T *)d_data_ + sm * MAX_INSTS,
        instructions[sm].data(),
        sizeof(T) * instructions[sm].size(),
        cudaMemcpyHostToDevice
      );
    }
    return static_cast<T*>(d_data_);
  }
};

struct DAELauncher {
  int numSMs;
  
  InstructionBuilder<MInst> mem;
  InstructionBuilder<CInst> comp;
  ProfileBuilder<uint64_t> profile;
  TMABuilder tma;

  std::vector<std::unique_ptr<SMDataBuilder>> data_blocks;

  DAELauncher(int num_sms)
    : numSMs(num_sms),
      mem(num_sms),
      comp(num_sms),
      profile(num_sms, numProfileEvents) {}

  SMDataBuilder& sm_data(size_t sm_bytes) {
    data_blocks.emplace_back(std::make_unique<SMDataBuilder>(numSMs, sm_bytes));
    return *data_blocks.back();
  }
  SMDataBuilder& data(size_t total_bytes) {
    assert(total_bytes % numSMs == 0 && "Total bytes must be divisible by number of SMs");
    return sm_data(total_bytes / numSMs);
  }

  size_t slot_bytes() const {
    return slotSizeKb * 1024;
  }

  int launch() {
    copy_to_device();
    size_t smem_size = set_smem_size();

    int *bars;
    cudaMalloc(&bars, sizeof(int) * 128);

    auto comp_d = comp.copy_to_device();
    auto mem_d = mem.copy_to_device();
    auto tma_d = tma.copy_to_device();
    auto profile_d = profile.copy_to_device();

    cudaError_t err = launch_dae(
      numSMs, smem_size,
      comp_d, mem_d, tma_d, bars,
      profile_d
    );

    if (err != cudaSuccess) {
      std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }

    cudaFree(bars);

    return 0;
  }

  int launch_bench(float totalBytes, int numRuns = 1000) {
    copy_to_device();
    size_t smem_size = set_smem_size();
    int *bars;
    cudaMalloc(&bars, sizeof(int) * 128);

    std::cout << "Launching DAE2 kernel with " << numSMs << " SMs, " << (32 * (numComputeWarps + 1)) << " threads per block, " << smem_size << " bytes dynamic shared memory" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<uint64_t> profileCycles;
    
    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
      auto comp_d = comp.copy_to_device();
      auto mem_d = mem.copy_to_device();
      auto tma_d = tma.copy_to_device();
      auto profile_d = profile.copy_to_device();

      launch_dae(
        numSMs, smem_size,
        comp_d, mem_d, tma_d, bars,
        profile_d
      );

      for (int sm = 0; sm < numSMs; sm++) {
        uint64_t h_events[numProfileEvents];
        cudaMemcpy(&h_events[0], profile.d_get(sm), sizeof(uint64_t) * numProfileEvents, cudaMemcpyDeviceToHost);
        uint64_t start_cycle = h_events[0];
        uint64_t end_cycle = h_events[1];
        if (end_cycle <= start_cycle) {
          continue;
        }
        profileCycles.push_back(end_cycle - start_cycle);
      }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print benchmark results
    float avgTime = milliseconds / numRuns;
    std::cout << "Benchmarking Results:" << std::endl;
    std::cout << "  Total runs: " << numRuns << std::endl;
    std::cout << "  Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "  Average time per run: " << avgTime << " ms" << std::endl;
    float bandwidth = totalBytes / (avgTime / 1000.0f) / (1024.0f * 1024.0f); // MB/s
    std::cout << "  Effective Bandwidth: " << bandwidth << " MB/s" << std::endl;
    double totalNs = 0;
    for (auto cycles : profileCycles) {
      totalNs += cycles;
    }
    double avgNs = totalNs / profileCycles.size();
    std::cout << "  Average time per run: " << avgNs << " ns" << std::endl;
    std::cout << "  Average Bandwidth from profile: " << (totalBytes / (avgNs / 1e9)) / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }
    cudaFree(bars);

    return 0;
  }

  // Kernel launch helper
  void copy_to_device() {
    for (auto &db : data_blocks) {
      db->copy_to_device();
    }
  }
};
