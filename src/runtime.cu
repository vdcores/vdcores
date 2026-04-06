#include "dae2.cuh"
#include "runtime.cuh"

#include <cuda.h>
#include <vector>

static std::vector<cudaStream_t> &compiled_split_streams() {
    static std::vector<cudaStream_t> streams;
    if (streams.size() < static_cast<size_t>(daeCompiledProgramCount)) {
        size_t old_size = streams.size();
        streams.resize(static_cast<size_t>(daeCompiledProgramCount), nullptr);
        for (size_t index = old_size; index < streams.size(); ++index) {
            cudaError_t err = cudaStreamCreateWithFlags(&streams[index], cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                streams.resize(old_size);
                break;
            }
        }
    }
    return streams;
}

size_t set_smem_size(size_t smem_size) {
    cudaError_t err = cudaFuncSetAttribute(
        dae2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    if (err != cudaSuccess) {
        std::cerr << "Kernel set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFuncSetAttribute(
        dae2_compiled,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    if (err != cudaSuccess) {
        std::cerr << "Compiled kernel set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = dae_set_compiled_program_smem_size(smem_size);
    if (err != cudaSuccess) {
        std::cerr << "Compiled split kernel set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
    return smem_size;
}

cudaError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
) {
  // wait for all pre-launch meta-data copying
  cudaDeviceSynchronize();
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dae2<<<numSMs, numThreads, smem_size, cuda_stream>>>(
    compute_instructions,
    memory_instructions,
    tma_descs,
    bars,
    profile
  );
  // TODO(zhiyuang): check launch error here?

  cudaDeviceSynchronize();

  return cudaGetLastError();
}

cudaError_t launch_dae_compiled(
  int numSMs,
  size_t smem_size,
  uint64_t* compiled_live_values,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream,
  int launch_mode
) {
  cudaDeviceSynchronize();
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (launch_mode == daeCompiledLaunchModeSplit && daeCompiledProgramCount > 1) {
    cudaError_t err = cudaMemsetAsync(
        bars + daeCompiledSplitLaunchArrivalBar,
        0,
        static_cast<size_t>(daeCompiledSplitLaunchReservedBars) * sizeof(int),
        cuda_stream
    );
    if (err != cudaSuccess) {
      return err;
    }

    cudaEvent_t ready_event = nullptr;
    err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaEventRecord(ready_event, cuda_stream);
    if (err != cudaSuccess) {
      cudaEventDestroy(ready_event);
      return err;
    }

    auto &streams = compiled_split_streams();
    if (streams.size() < static_cast<size_t>(daeCompiledProgramCount)) {
      cudaEventDestroy(ready_event);
      return cudaErrorMemoryAllocation;
    }

    for (int program_id = 0; program_id < daeCompiledProgramCount; ++program_id) {
      cudaStream_t program_stream = streams[static_cast<size_t>(program_id)];
      err = cudaStreamWaitEvent(program_stream, ready_event, 0);
      if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        return err;
      }
      err = dae_launch_compiled_split_program(
        program_id,
        dae_compiled_program_block_count(program_id),
        smem_size,
        program_stream,
        compiled_live_values,
        tma_descs,
        bars,
        profile
      );
      if (err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        return err;
      }
    }
    cudaEventDestroy(ready_event);
  } else {
    dae2_compiled<<<numSMs, numThreads, smem_size, cuda_stream>>>(
      compiled_live_values,
      tma_descs,
      bars,
      profile
    );
  }
  cudaDeviceSynchronize();
  return cudaGetLastError();
}

cudaError_t set_compiled_live_values_constant(
  const uint64_t* compiled_live_values,
  size_t count
) {
  if (count == 0) {
    return cudaSuccess;
  }
  if (count != static_cast<size_t>(daeCompiledProgramLiveValueCount)) {
    return cudaErrorInvalidValue;
  }
  return cudaMemcpyToSymbol(
    daeCompiledLiveValuesConst,
    compiled_live_values,
    count * sizeof(uint64_t),
    0,
    cudaMemcpyDeviceToDevice
  );
}

CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle,
  std::array<uint64_t, 5> global_strides_opt
) {
  assert(dims <= 5 && "Maximum 5 dimensions supported");

  CUtensorMap desc;

  int element_size = -1; // default to BF16

  if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT8) {
    element_size = 1;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT16 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) {
    element_size = 2;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT32 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT32) {
    element_size = 4;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT64 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT64) {
    element_size = 8;
  }
  assert(element_size > 0 && "Unsupported data type");

  uint64_t global_strides[5];
  uint32_t box_strides[5];

  // Calculate global strides using cumulative products
  global_strides[0] = global_dims[0] * element_size;
  for (int i = 1; i < dims - 1; i++) {
    global_strides[i] = global_strides[i-1] * global_dims[i];
  }

  // Box strides are always 1 (contiguous within each tile)
  for (int i = 0; i < dims; i++) {
    box_strides[i] = 1;
  }

  auto result = cuTensorMapEncodeTiled(
    &desc,
    data_type,
    dims,
    base,
    global_dims.data(),
    // we go with a compact layout if no strides are provided
    global_strides_opt[0] == 0 ? global_strides : global_strides_opt.data(),
    box_dims.data(),
    box_strides,

    CU_TENSOR_MAP_INTERLEAVE_NONE,    // No interleaving
    swizzle,       // Swizzle mode
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,  // No L2 promotion
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // No special OOB handling
  );
  assert(result == CUDA_SUCCESS && "Failed to create tensor map");
  
  return desc;
}
