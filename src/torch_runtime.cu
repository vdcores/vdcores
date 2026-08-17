#include "dae/runtime.cuh"
#include "dae/context.cuh"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <ATen/cuda/CUDAContext.h>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cuda.h>            // Driver API
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kNvfp4TileM = 128;
constexpr int kNvfp4TileK = 256;
constexpr int kNvfp4PackedK = kNvfp4TileK / 2;
constexpr int kNvfp4WeightDataBytes = kNvfp4TileM * kNvfp4PackedK;
constexpr int kNvfp4WeightTileBytes = 18432;
constexpr int kNvfp4K512TileK = 512;
constexpr int kNvfp4K512PackedK = kNvfp4K512TileK / 2;
constexpr int kNvfp4K512WeightDataBytes =
    kNvfp4TileM * kNvfp4K512PackedK;
constexpr int kNvfp4K512WeightScaleBytes = 4096;
constexpr int kNvfp4K512ActivationDataBytes = 8 * kNvfp4K512PackedK;
constexpr int kNvfp4K512ActivationScaleBytes = 32;
constexpr int kFp8TileM = 128;
constexpr int kFp8TileK = 128;
constexpr int kFp8WeightDataBytes = kFp8TileM * kFp8TileK;
constexpr int kFp8WeightTileBytes = 16896;

__global__ void prepack_nvfp4_checkpoint_kernel(
    const uint8_t *__restrict__ weight,
    const cutlass::float_e4m3_t *__restrict__ checkpoint_scale,
    uint8_t *__restrict__ output,
    int packed_k,
    int k_tiles) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, float, Scale,
      kNvfp4TileM, 8, 16,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
  using ScaleProblemShape = Shape<Int<kNvfp4TileM>, Int<128>, Int<kNvfp4TileK>>;

  const int m_tile = blockIdx.x / k_tiles;
  const int k_tile = blockIdx.x - m_tile * k_tiles;
  auto *tile_output = output + blockIdx.x * kNvfp4WeightTileBytes;

  // This is the inverse of the 128-byte TMA XOR swizzle used by the UMMA A
  // operand. Each physical 16-byte chunk receives logical chunk d xor row.
  for (int index = threadIdx.x; index < kNvfp4WeightDataBytes;
       index += blockDim.x) {
    const int row = index / kNvfp4PackedK;
    const int destination_column = index - row * kNvfp4PackedK;
    const int destination_chunk = destination_column / 16;
    const int byte_in_chunk = destination_column - destination_chunk * 16;
    const int source_chunk = destination_chunk ^ (row & 7);
    const int source_column =
        k_tile * kNvfp4PackedK + source_chunk * 16 + byte_in_chunk;
    tile_output[index] =
        weight[(m_tile * kNvfp4TileM + row) * packed_k + source_column];
  }

  TiledMma tiled_mma;
  const auto logical_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      ScaleProblemShape{});
  cutlass::NumericConverter<Scale, cutlass::float_e4m3_t> convert_scale;
  constexpr int kScaleColumns = kNvfp4TileK / 16;
  auto *packed_scale = reinterpret_cast<Scale *>(
      tile_output + kNvfp4WeightDataBytes);
  const int source_scale_columns = packed_k / 8;
  for (int index = threadIdx.x;
       index < kNvfp4TileM * kScaleColumns;
       index += blockDim.x) {
    const int row = index / kScaleColumns;
    const int sf = index - row * kScaleColumns;
    const int destination = int(logical_sfa(row, sf * 16));
    packed_scale[destination] = convert_scale(
        checkpoint_scale[
            (m_tile * kNvfp4TileM + row) * source_scale_columns +
            k_tile * kScaleColumns + sf]);
  }
}

void py_prepack_nvfp4_checkpoint(
    torch::Tensor weight,
    torch::Tensor checkpoint_scale,
    torch::Tensor output) {
  TORCH_CHECK(weight.is_cuda() && checkpoint_scale.is_cuda() && output.is_cuda(),
              "NVFP4 prepack tensors must be CUDA tensors");
  TORCH_CHECK(weight.device() == checkpoint_scale.device() &&
                  weight.device() == output.device(),
              "NVFP4 prepack tensors must share one CUDA device");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Byte &&
                  weight.dim() == 2 && weight.is_contiguous(),
              "NVFP4 weight must be contiguous rank-2 uint8");
  TORCH_CHECK(checkpoint_scale.scalar_type() ==
                  at::ScalarType::Float8_e4m3fn &&
                  checkpoint_scale.dim() == 2 &&
                  checkpoint_scale.is_contiguous(),
              "NVFP4 scale must be contiguous rank-2 E4M3");
  const int64_t rows = weight.size(0);
  const int64_t packed_k = weight.size(1);
  TORCH_CHECK(rows % kNvfp4TileM == 0 && packed_k % kNvfp4PackedK == 0,
              "NVFP4 weight must be M128/K256 aligned");
  TORCH_CHECK(checkpoint_scale.size(0) == rows &&
                  checkpoint_scale.size(1) == packed_k / 8,
              "NVFP4 checkpoint scale shape does not match the weight");
  const int64_t m_tiles = rows / kNvfp4TileM;
  const int64_t k_tiles = packed_k / kNvfp4PackedK;
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Byte &&
                  output.is_contiguous() && output.dim() == 3 &&
                  output.size(0) == m_tiles && output.size(1) == k_tiles &&
                  output.size(2) == kNvfp4WeightTileBytes,
              "NVFP4 native output must be contiguous [M/128,K/256,18432] uint8");

  const auto stream = at::cuda::getCurrentCUDAStream(weight.device().index());
  prepack_nvfp4_checkpoint_kernel<<<
      static_cast<unsigned>(m_tiles * k_tiles), 256, 0, stream>>>(
      weight.data_ptr<uint8_t>(),
      reinterpret_cast<const cutlass::float_e4m3_t *>(
          checkpoint_scale.data_ptr()),
      output.data_ptr<uint8_t>(),
      static_cast<int>(packed_k),
      static_cast<int>(k_tiles));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "NVFP4 checkpoint prepack failed: ", cudaGetErrorString(error));
}

__global__ void prepack_nvfp4_checkpoint_k512_split_kernel(
    const uint8_t *__restrict__ weight,
    const cutlass::float_e4m3_t *__restrict__ checkpoint_scale,
    uint8_t *__restrict__ output_data,
    uint8_t *__restrict__ output_scale,
    int packed_k,
    int k_tiles) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, float, Scale,
      kNvfp4TileM, 8, 16,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
  using ScaleProblemShape =
      Shape<Int<kNvfp4TileM>, Int<128>, Int<kNvfp4TileK>>;

  const int m_tile = blockIdx.x / k_tiles;
  const int k_tile = blockIdx.x - m_tile * k_tiles;
  auto *tile_data =
      output_data + blockIdx.x * kNvfp4K512WeightDataBytes;
  auto *tile_scale = reinterpret_cast<Scale *>(
      output_scale + blockIdx.x * kNvfp4K512WeightScaleBytes);

  for (int index = threadIdx.x; index < kNvfp4K512WeightDataBytes;
       index += blockDim.x) {
    const int subtile = index / kNvfp4WeightDataBytes;
    const int local_index = index - subtile * kNvfp4WeightDataBytes;
    const int row = local_index / kNvfp4PackedK;
    const int destination_column = local_index - row * kNvfp4PackedK;
    const int destination_chunk = destination_column / 16;
    const int byte_in_chunk = destination_column - destination_chunk * 16;
    const int source_chunk = destination_chunk ^ (row & 7);
    const int source_column =
        (k_tile * 2 + subtile) * kNvfp4PackedK +
        source_chunk * 16 + byte_in_chunk;
    tile_data[index] =
        weight[(m_tile * kNvfp4TileM + row) * packed_k + source_column];
  }

  TiledMma tiled_mma;
  const auto logical_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      ScaleProblemShape{});
  cutlass::NumericConverter<Scale, cutlass::float_e4m3_t> convert_scale;
  constexpr int kScaleColumns = kNvfp4TileK / 16;
  const int source_scale_columns = packed_k / 8;
  for (int index = threadIdx.x;
       index < 2 * kNvfp4TileM * kScaleColumns;
       index += blockDim.x) {
    const int subtile = index / (kNvfp4TileM * kScaleColumns);
    const int local_index =
        index - subtile * kNvfp4TileM * kScaleColumns;
    const int row = local_index / kScaleColumns;
    const int sf = local_index - row * kScaleColumns;
    const int destination = subtile * (kNvfp4TileM * kScaleColumns) +
        int(logical_sfa(row, sf * 16));
    tile_scale[destination] = convert_scale(
        checkpoint_scale[
            (m_tile * kNvfp4TileM + row) * source_scale_columns +
            (k_tile * 2 + subtile) * kScaleColumns + sf]);
  }
}

void py_prepack_nvfp4_checkpoint_k512_split(
    torch::Tensor weight,
    torch::Tensor checkpoint_scale,
    torch::Tensor output_data,
    torch::Tensor output_scale) {
  TORCH_CHECK(
      weight.is_cuda() && checkpoint_scale.is_cuda() &&
          output_data.is_cuda() && output_scale.is_cuda(),
      "K512 NVFP4 prepack tensors must be CUDA tensors");
  TORCH_CHECK(
      weight.device() == checkpoint_scale.device() &&
          weight.device() == output_data.device() &&
          weight.device() == output_scale.device(),
      "K512 NVFP4 prepack tensors must share one CUDA device");
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Byte && weight.dim() == 2 &&
          weight.is_contiguous(),
      "K512 NVFP4 weight must be contiguous rank-2 uint8");
  TORCH_CHECK(
      checkpoint_scale.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          checkpoint_scale.dim() == 2 && checkpoint_scale.is_contiguous(),
      "K512 NVFP4 scale must be contiguous rank-2 E4M3");
  const int64_t rows = weight.size(0);
  const int64_t packed_k = weight.size(1);
  TORCH_CHECK(
      rows % kNvfp4TileM == 0 && packed_k % kNvfp4K512PackedK == 0,
      "K512 NVFP4 weight must be M128/K512 aligned");
  TORCH_CHECK(
      checkpoint_scale.size(0) == rows &&
          checkpoint_scale.size(1) == packed_k / 8,
      "K512 NVFP4 checkpoint scale shape does not match the weight");
  const int64_t m_tiles = rows / kNvfp4TileM;
  const int64_t k_tiles = packed_k / kNvfp4K512PackedK;
  TORCH_CHECK(
      output_data.scalar_type() == at::ScalarType::Byte &&
          output_data.is_contiguous() && output_data.dim() == 3 &&
          output_data.size(0) == m_tiles &&
          output_data.size(1) == k_tiles &&
          output_data.size(2) == kNvfp4K512WeightDataBytes,
      "K512 NVFP4 data output must be [M/128,K/512,32768] uint8");
  TORCH_CHECK(
      output_scale.scalar_type() == at::ScalarType::Byte &&
          output_scale.is_contiguous() && output_scale.dim() == 3 &&
          output_scale.size(0) == m_tiles &&
          output_scale.size(1) == k_tiles &&
          output_scale.size(2) == kNvfp4K512WeightScaleBytes,
      "K512 NVFP4 scale output must be [M/128,K/512,4096] uint8");

  const auto stream = at::cuda::getCurrentCUDAStream(weight.device().index());
  prepack_nvfp4_checkpoint_k512_split_kernel<<<
      static_cast<unsigned>(m_tiles * k_tiles), 256, 0, stream>>>(
      weight.data_ptr<uint8_t>(),
      reinterpret_cast<const cutlass::float_e4m3_t *>(
          checkpoint_scale.data_ptr()),
      output_data.data_ptr<uint8_t>(),
      output_scale.data_ptr<uint8_t>(),
      static_cast<int>(packed_k),
      static_cast<int>(k_tiles));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "K512 NVFP4 checkpoint prepack failed: ", cudaGetErrorString(error));
}

__global__ void prepack_nvfp4_activation_k512_split_kernel(
    const uint8_t *__restrict__ activation,
    const cutlass::float_e4m3_t *__restrict__ checkpoint_scale,
    uint8_t *__restrict__ output_data,
    uint8_t *__restrict__ output_scale) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, float, Scale,
      kNvfp4TileM, 8, 16,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
  const int k_tile = blockIdx.x;
  auto *tile_data =
      output_data + k_tile * kNvfp4K512ActivationDataBytes;
  auto *tile_scale_bytes =
      output_scale + k_tile * kNvfp4K512ActivationScaleBytes;
  for (int index = threadIdx.x;
       index < kNvfp4K512ActivationDataBytes;
       index += blockDim.x) {
    constexpr int kActivationK256DataBytes = 8 * kNvfp4PackedK;
    const int subtile = index / kActivationK256DataBytes;
    const int local_index = index - subtile * kActivationK256DataBytes;
    const int row = local_index / kNvfp4PackedK;
    const int destination_column = local_index - row * kNvfp4PackedK;
    const int destination_chunk = destination_column / 16;
    const int byte_in_chunk = destination_column - destination_chunk * 16;
    const int source_chunk = destination_chunk ^ row;
    const int source_column =
        (k_tile * 2 + subtile) * kNvfp4PackedK +
        source_chunk * 16 + byte_in_chunk;
    tile_data[index] = activation[source_column];
  }
  cutlass::NumericConverter<Scale, cutlass::float_e4m3_t> convert_scale;
  auto *tile_scale = reinterpret_cast<Scale *>(tile_scale_bytes);
  constexpr int kScaleColumns = kNvfp4K512TileK / 16;
  for (int index = threadIdx.x; index < kScaleColumns;
       index += blockDim.x) {
    tile_scale[index] = convert_scale(
        checkpoint_scale[k_tile * kScaleColumns + index]);
  }
}

void py_prepack_nvfp4_activation_k512_split(
    torch::Tensor activation,
    torch::Tensor checkpoint_scale,
    torch::Tensor output_data,
    torch::Tensor output_scale) {
  TORCH_CHECK(
      activation.is_cuda() && checkpoint_scale.is_cuda() &&
          output_data.is_cuda() && output_scale.is_cuda(),
      "K512 NVFP4 activation prepack tensors must be CUDA tensors");
  TORCH_CHECK(
      activation.device() == checkpoint_scale.device() &&
          activation.device() == output_data.device() &&
          activation.device() == output_scale.device(),
      "K512 NVFP4 activation prepack tensors must share one CUDA device");
  TORCH_CHECK(
      activation.scalar_type() == at::ScalarType::Byte &&
          activation.dim() == 1 && activation.is_contiguous(),
      "K512 NVFP4 activation must be contiguous rank-1 uint8");
  TORCH_CHECK(
      checkpoint_scale.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          checkpoint_scale.dim() == 1 && checkpoint_scale.is_contiguous(),
      "K512 NVFP4 activation scale must be contiguous rank-1 E4M3");
  TORCH_CHECK(
      activation.numel() % kNvfp4K512PackedK == 0 &&
          checkpoint_scale.numel() == activation.numel() / 8,
      "K512 NVFP4 activation and scale shapes do not match");
  const int64_t k_tiles = activation.numel() / kNvfp4K512PackedK;
  TORCH_CHECK(
      output_data.scalar_type() == at::ScalarType::Byte &&
          output_data.is_contiguous() && output_data.dim() == 2 &&
          output_data.size(0) == k_tiles &&
          output_data.size(1) == kNvfp4K512ActivationDataBytes,
      "K512 activation data output must be [K/512,2048] uint8");
  TORCH_CHECK(
      output_scale.scalar_type() == at::ScalarType::Byte &&
          output_scale.is_contiguous() && output_scale.dim() == 2 &&
          output_scale.size(0) == k_tiles &&
          output_scale.size(1) == kNvfp4K512ActivationScaleBytes,
      "K512 activation scale output must be [K/512,32] uint8");

  const auto stream =
      at::cuda::getCurrentCUDAStream(activation.device().index());
  prepack_nvfp4_activation_k512_split_kernel<<<
      static_cast<unsigned>(k_tiles), 256, 0, stream>>>(
      activation.data_ptr<uint8_t>(),
      reinterpret_cast<const cutlass::float_e4m3_t *>(
          checkpoint_scale.data_ptr()),
      output_data.data_ptr<uint8_t>(),
      output_scale.data_ptr<uint8_t>());
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "K512 activation prepack failed: ", cudaGetErrorString(error));
}

__global__ void prepack_fp8_checkpoint_kernel(
    const uint8_t *__restrict__ weight,
    const cutlass::float_ue8m0_t *__restrict__ checkpoint_scale,
    uint8_t *__restrict__ output,
    int k,
    int k_tiles,
    int scale_pack) {
  using namespace cute;
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Atom = SM100_MMA_MXF8F6F4_SS<
      Fp8, Fp8, float, Scale, kFp8TileM, 8,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  using ScaleProblemShape = Shape<Int<kFp8TileM>, Int<128>, Int<kFp8TileK>>;

  const int scale_groups = k_tiles / scale_pack;
  const int m_tile = blockIdx.x / scale_groups;
  const int scale_group = blockIdx.x - m_tile * scale_groups;
  const int group_start = scale_group * scale_pack;

  for (int pack_tile = 0; pack_tile < scale_pack; ++pack_tile) {
    const int k_tile = group_start + pack_tile;
    auto *tile_output = output +
        (m_tile * k_tiles + k_tile) * kFp8WeightTileBytes;
    for (int index = threadIdx.x; index < kFp8WeightDataBytes;
         index += blockDim.x) {
      const int row = index / kFp8TileK;
      const int destination_column = index - row * kFp8TileK;
      const int destination_chunk = destination_column / 16;
      const int byte_in_chunk = destination_column - destination_chunk * 16;
      const int source_chunk = destination_chunk ^ (row & 7);
      const int source_column =
          k_tile * kFp8TileK + source_chunk * 16 + byte_in_chunk;
      tile_output[index] =
          weight[(m_tile * kFp8TileM + row) * k + source_column];
    }
    for (int index = threadIdx.x;
         index < kFp8WeightTileBytes - kFp8WeightDataBytes;
         index += blockDim.x) {
      tile_output[kFp8WeightDataBytes + index] = 0;
    }
  }
  __syncthreads();

  TiledMma tiled_mma;
  const auto logical_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      ScaleProblemShape{});
  constexpr int kScaleColumns = kFp8TileK / 32;
  auto *packed_scale = reinterpret_cast<Scale *>(
      output + (m_tile * k_tiles + group_start) * kFp8WeightTileBytes +
      kFp8WeightDataBytes);
  if (scale_pack == 1) {
    const Scale tile_scale =
        checkpoint_scale[m_tile * k_tiles + group_start];
    for (int index = threadIdx.x;
         index < kFp8TileM * kScaleColumns;
         index += blockDim.x) {
      const int row = index / kScaleColumns;
      const int sf = index - row * kScaleColumns;
      const int destination = int(logical_sfa(row, sf * 32));
      packed_scale[destination] = tile_scale;
    }
  } else {
    for (int index = threadIdx.x;
         index < kFp8TileM * scale_pack;
         index += blockDim.x) {
      const int row = index / scale_pack;
      const int sf = index - row * scale_pack;
      const int destination = int(logical_sfa(row, sf * 32));
      packed_scale[destination] =
          checkpoint_scale[m_tile * k_tiles + group_start + sf];
    }
  }
}

void py_prepack_fp8_checkpoint(
    torch::Tensor weight,
    torch::Tensor checkpoint_scale,
    torch::Tensor output,
    int scale_pack) {
  TORCH_CHECK(weight.is_cuda() && checkpoint_scale.is_cuda() && output.is_cuda(),
              "FP8 prepack tensors must be CUDA tensors");
  TORCH_CHECK(weight.device() == checkpoint_scale.device() &&
                  weight.device() == output.device(),
              "FP8 prepack tensors must share one CUDA device");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  weight.dim() == 2 && weight.is_contiguous(),
              "FP8 weight must be contiguous rank-2 E4M3");
  TORCH_CHECK(checkpoint_scale.scalar_type() ==
                  at::ScalarType::Float8_e8m0fnu &&
                  checkpoint_scale.dim() == 2 &&
                  checkpoint_scale.is_contiguous(),
              "FP8 scale must be contiguous rank-2 UE8M0");
  const int64_t rows = weight.size(0);
  const int64_t k = weight.size(1);
  TORCH_CHECK(rows % kFp8TileM == 0 && k % kFp8TileK == 0,
              "FP8 weight must be M128/K128 aligned");
  const int64_t m_tiles = rows / kFp8TileM;
  const int64_t k_tiles = k / kFp8TileK;
  TORCH_CHECK(
      (scale_pack == 1 || scale_pack == 2 || scale_pack == 4) &&
          k_tiles % scale_pack == 0,
      "FP8 scale pack must be 1, 2, or 4 and divide K/128");
  TORCH_CHECK(checkpoint_scale.size(0) == m_tiles &&
                  checkpoint_scale.size(1) == k_tiles,
              "FP8 checkpoint scale shape does not match the weight");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Byte &&
                  output.is_contiguous() && output.dim() == 3 &&
                  output.size(0) == m_tiles && output.size(1) == k_tiles &&
                  output.size(2) == kFp8WeightTileBytes,
              "FP8 native output must be contiguous [M/128,K/128,16896] uint8");

  const auto stream = at::cuda::getCurrentCUDAStream(weight.device().index());
  prepack_fp8_checkpoint_kernel<<<
      static_cast<unsigned>(m_tiles * k_tiles / scale_pack), 256, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(weight.data_ptr()),
      reinterpret_cast<const cutlass::float_ue8m0_t *>(
          checkpoint_scale.data_ptr()),
      output.data_ptr<uint8_t>(),
      static_cast<int>(k),
      static_cast<int>(k_tiles),
      scale_pack);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "FP8 checkpoint prepack failed: ", cudaGetErrorString(error));
}

}  // namespace

static cudaDeviceProp current_device_prop();
static void set_persistent_cache();

static LoopCounters checked_loop_counts(const std::vector<int64_t>& values) {
  TORCH_CHECK(
      values.size() == numComputeLoopCounters,
      "initial_loop_counts must have ", numComputeLoopCounters, " elements");
  LoopCounters counters{};
  for (int i = 0; i < numComputeLoopCounters; ++i) {
    TORCH_CHECK(
        values[i] >= 0 && values[i] <= UINT32_MAX,
        "initial_loop_counts[", i, "] must fit in uint32");
    counters.values[i] = static_cast<uint32_t>(values[i]);
  }
  return counters;
}

// function 1: set smem size
size_t py_set_smem_size(size_t requested_size) {
  const cudaDeviceProp prop = current_device_prop();
  const size_t max_optin = static_cast<size_t>(prop.sharedMemPerBlockOptin);
  TORCH_CHECK(
      requested_size <= max_optin,
      "requested dynamic shared memory (", requested_size,
      " bytes) exceeds this device's per-block opt-in limit (", max_optin, " bytes)");
  const size_t configured_size = set_smem_size(requested_size);
  TORCH_CHECK(
      configured_size == requested_size,
      "failed to configure dae2 dynamic shared memory to ", requested_size, " bytes");
  set_persistent_cache();
  return configured_size;
}

template <typename T>
static inline T* check_tensor_ptr(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");

  const int64_t rows = t.size(0);
  const int64_t cols = t.size(1);

  TORCH_CHECK(cols == (int64_t)sizeof(T),
              name, " second dimension must equal sizeof(T) = ",
              sizeof(T), " but got ", cols);

  // Now memory layout is guaranteed to be:
  // rows contiguous records of sizeof(T) bytes each.
  auto* p = reinterpret_cast<T*>(t.data_ptr<uint8_t>());

  // Alignment safety (important for 16-byte aligned structs)
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  TORCH_CHECK(addr % alignof(T) == 0,
              name, " misaligned pointer: address mod alignof(T) = ",
              (addr % alignof(T)));

  return p;
}

static cudaDeviceProp current_device_prop() {
  cudaDeviceProp prop{};
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  return prop;
}

static std::optional<size_t> env_size_t(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0')) {
    return std::nullopt;
  }
  return static_cast<size_t>(parsed);
}

static std::optional<double> env_double(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  double parsed = std::strtod(raw, &end);
  if (end == raw || (end != nullptr && *end != '\0')) {
    return std::nullopt;
  }
  return parsed;
}

static size_t select_persisting_l2_size(const cudaDeviceProp& prop) {
  const size_t max_size = static_cast<size_t>(prop.persistingL2CacheMaxSize);
  if (max_size == 0) {
    return 0;
  }
  if (auto requested_bytes = env_size_t("DAE_PERSISTING_L2_BYTES")) {
    return std::min(*requested_bytes, max_size);
  }

  const double requested_fraction = env_double("DAE_PERSISTING_L2_FRACTION").value_or(0.0625);
  const double clamped_fraction = std::clamp(requested_fraction, 0.0, 1.0);
  return std::min(static_cast<size_t>(clamped_fraction * max_size), max_size);
}

static CUtensorMapL2promotion select_tma_l2_promotion() {
  const char* raw = std::getenv("DAE_TMA_L2_PROMOTION");
  if (raw == nullptr || raw[0] == '\0') {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }

  const std::string value(raw);
  if (value == "0" || value == "none") {
    return CU_TENSOR_MAP_L2_PROMOTION_NONE;
  }
  if (value == "64" || value == "64b" || value == "l2_64b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
  }
  if (value == "128" || value == "128b" || value == "l2_128b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  }
  if (value == "256" || value == "256b" || value == "l2_256b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
  TORCH_CHECK(false, "Unsupported DAE_TMA_L2_PROMOTION=", value, " (expected none/64/128/256)");
}

static void set_persistent_cache() {
  const cudaDeviceProp prop = current_device_prop();

  // printf("L2 size: %d bytes\n", prop.l2CacheSize);
  // printf("persistingL2CacheMaxSize: %zu bytes\n", prop.persistingL2CacheMaxSize);
  // printf("accessPolicyMaxWindowSize: %zu bytes\n", prop.accessPolicyMaxWindowSize);

  const size_t set_aside = select_persisting_l2_size(prop);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
  // printf("persistentCacheSize: %zu bytes\n", set_aside);
}

// function 2: launch_dae
int py_launch_dae(
    int64_t num_sms,
    size_t smem_size,
    torch::Tensor compute_insts_bytes,   // uint8 buffer
    torch::Tensor memory_insts_bytes,    // uint8 buffer
    torch::Tensor tma_descs_bytes,       // uint8 buffer
    torch::Tensor bars_int32,            // int32
    torch::Tensor profile_u64,           // uint64
    const std::vector<int64_t>& initial_loop_counts,
    int64_t stream,
    bool synchronize
) {
  const cudaDeviceProp prop = current_device_prop();
  TORCH_CHECK(
      num_sms > 0 && num_sms <= prop.multiProcessorCount,
      "num_sms out of range for ", prop.name, ": requested ", num_sms,
      ", device has ", prop.multiProcessorCount, " SMs");
  TORCH_CHECK(
      smem_size <= static_cast<size_t>(prop.sharedMemPerBlockOptin),
      "smem_size exceeds this device's per-block opt-in shared-memory limit");

  // Make sure we run on the right device/stream
  auto cinst = check_tensor_ptr<CInst>(compute_insts_bytes, "compute_insts_bytes");
  auto minst = check_tensor_ptr<MInst>(memory_insts_bytes, "memory_insts_bytes");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  auto prof = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");
  LoopCounters counters = checked_loop_counts(initial_loop_counts);

  cudaError_t st = launch_dae(
      static_cast<int>(num_sms), smem_size,
      cinst, minst, tma,
      bars, prof, counters, stream, synchronize
  );

  TORCH_CHECK(st == cudaSuccess, "launch_dae failed: ", cudaGetErrorString(st));

  // Return something meaningful; often you return profile or nothing.
  return 0;
}

int py_launch_dae_sequence(
    int64_t num_sms,
    size_t smem_size,
    torch::Tensor compute_insts_bytes,
    torch::Tensor memory_insts_bytes,
    torch::Tensor tma_descs_bytes,
    torch::Tensor bars_int32,
    torch::Tensor profile_u64,
    const std::vector<std::vector<int64_t>>& sequence_loop_counts,
    int64_t stream,
    bool synchronize
) {
  const cudaDeviceProp prop = current_device_prop();
  TORCH_CHECK(
      num_sms > 0 && num_sms <= prop.multiProcessorCount,
      "num_sms out of range for ", prop.name, ": requested ", num_sms,
      ", device has ", prop.multiProcessorCount, " SMs");
  TORCH_CHECK(
      smem_size <= static_cast<size_t>(prop.sharedMemPerBlockOptin),
      "smem_size exceeds this device's per-block opt-in shared-memory limit");
  TORCH_CHECK(!sequence_loop_counts.empty(), "sequence_loop_counts must not be empty");

  auto cinst = check_tensor_ptr<CInst>(compute_insts_bytes, "compute_insts_bytes");
  auto minst = check_tensor_ptr<MInst>(memory_insts_bytes, "memory_insts_bytes");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  auto prof = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");

  std::vector<LoopCounters> counters;
  counters.reserve(sequence_loop_counts.size());
  for (const auto& values : sequence_loop_counts) {
    counters.push_back(checked_loop_counts(values));
  }

  cudaError_t st = launch_dae_sequence(
      static_cast<int>(num_sms), smem_size,
      cinst, minst, tma,
      bars, prof, counters, stream, synchronize
  );
  TORCH_CHECK(st == cudaSuccess, "launch_dae_sequence failed: ", cudaGetErrorString(st));
  return 0;
}

int py_launch_dae_legacy(
    int64_t num_sms,
    size_t smem_size,
    torch::Tensor compute_insts_bytes,
    torch::Tensor memory_insts_bytes,
    torch::Tensor tma_descs_bytes,
    torch::Tensor bars_int32,
    torch::Tensor profile_u64,
    int64_t stream
) {
  return py_launch_dae(
      num_sms,
      smem_size,
      compute_insts_bytes,
      memory_insts_bytes,
      tma_descs_bytes,
      bars_int32,
      profile_u64,
      std::vector<int64_t>(numComputeLoopCounters, 0),
      stream,
      true
  );
}

#if defined(DAE_FFN_SPECIALIZED_KERNELS)
int py_launch_dae_ffn_linear1_direct(
    int64_t num_blocks,
    size_t smem_size,
    torch::Tensor metadata_bytes,
    torch::Tensor tma_descs_bytes,
    torch::Tensor bars_int32,
    int64_t reduction_bar_base,
    int64_t reduction_tiles,
    torch::Tensor profile_u64,
    int64_t stream) {
  const cudaDeviceProp prop = current_device_prop();
  TORCH_CHECK(
      num_blocks > 0 && num_blocks <= prop.multiProcessorCount,
      "direct Linear-1 block count must fit the physical SM count");
  TORCH_CHECK(
      smem_size <= static_cast<size_t>(prop.sharedMemPerBlockOptin),
      "smem_size exceeds this device's per-block opt-in shared-memory limit");
  auto metadata = check_tensor_ptr<uint8_t>(metadata_bytes, "metadata_bytes");
  TORCH_CHECK(
      metadata_bytes.numel() >= num_blocks * 128,
      "direct Linear-1 requires one 128-byte metadata record per block");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  TORCH_CHECK(
      reduction_bar_base >= 0 && reduction_tiles >= 0 &&
          reduction_bar_base + reduction_tiles <= bars_int32.size(0),
      "reduction barrier range is outside bars_int32");
  auto profile = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");
  TORCH_CHECK(
      profile_u64.size(0) >= num_blocks * numProfileEvents,
      "direct Linear-1 profile buffer is too small");
  const cudaError_t status = launch_dae_ffn_linear1_direct(
      static_cast<int>(num_blocks), smem_size, metadata, tma, bars,
      static_cast<int>(reduction_bar_base), static_cast<int>(reduction_tiles),
      profile, stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "direct Linear-1 launch failed: ", cudaGetErrorString(status));
  return 0;
}

int py_launch_dae_ffn_down_direct(
    int64_t num_blocks,
    size_t smem_size,
    torch::Tensor metadata_bytes,
    torch::Tensor tma_descs_bytes,
    torch::Tensor bars_int32,
    torch::Tensor profile_u64,
    int64_t stream) {
  const cudaDeviceProp prop = current_device_prop();
  TORCH_CHECK(
      num_blocks > 0 && num_blocks <= prop.multiProcessorCount,
      "paired direct down block count must fit one block per physical SM");
  TORCH_CHECK(
      smem_size <= static_cast<size_t>(prop.sharedMemPerBlockOptin),
      "smem_size exceeds this device's per-block opt-in shared-memory limit");
  auto metadata = check_tensor_ptr<uint8_t>(metadata_bytes, "metadata_bytes");
  TORCH_CHECK(
      metadata_bytes.numel() >= num_blocks * 2 * 128,
      "paired direct down requires two 128-byte metadata records per block");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  auto profile = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");
  TORCH_CHECK(
      profile_u64.size(0) >= num_blocks * numProfileEvents,
      "direct down profile buffer is too small");
  const cudaError_t status = launch_dae_ffn_down_direct(
      static_cast<int>(num_blocks), smem_size, metadata, tma, bars,
      profile, stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "direct down launch failed: ", cudaGetErrorString(status));
  return 0;
}

#endif

// function 3: build TMA descriptors
static inline CUtensorMapInterleave to_interleave(int64_t interleave) {
  switch (interleave) {
    case 0: return CU_TENSOR_MAP_INTERLEAVE_NONE;
    case 16: return CU_TENSOR_MAP_INTERLEAVE_16B;
    case 32: return CU_TENSOR_MAP_INTERLEAVE_32B;
    default: TORCH_CHECK(false, "Unsupported interleave=", interleave, " (expected 0/16/32)");
  }
}

static inline CUtensorMapSwizzle to_swizzle(int64_t swizzle_bytes) {
  switch (swizzle_bytes) {
    case 0:   return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default: TORCH_CHECK(false, "Unsupported swizzle_bytes=", swizzle_bytes, " (expected 0/32/64/128)");
  }
}

static inline CUtensorMapDataType to_dtype(torch::ScalarType st) {
  // Extend as you need
  switch (st) {
    case torch::kFloat16:  return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case torch::kBFloat16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case torch::kFloat32:  return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case torch::kUInt8:    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case torch::kInt32:    return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case torch::kUInt32:   return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype for TMA: ", c10::toString(st));
  }
}

// Build a CUtensorMap descriptor for a tensor.
// Arguments that must be consistent with your kernel's expected layout.
//
// shape:          sizes in elements, rank R
// strides_bytes:  strides in BYTES, rank R  (yes, bytes; not elements)
// box_dim:        tile dimensions in elements, rank R
// elem_strides:   element strides inside the tile, rank R (often all-ones)
// swizzle_bytes:  0/32/64/128
// interleave:     0 for NONE, 1 for 16B, 2 for 32B (optional; use NONE if unsure)
// l2_promo:       0 NONE, 1 64B, 2 128B, 3 256B (varies; use 256B commonly)
// oob_fill:       0 NONE, 1 NAN (float) etc (usually NONE)
torch::Tensor py_build_tma_desc(
    torch::Tensor base,                    // CUDA tensor providing base_ptr + device
    std::vector<int64_t> shape,            // length R
    std::vector<int64_t> strides_bytes,    // length R
    std::vector<int64_t> box_dim,          // length R
    std::vector<int64_t> elem_strides,     // length R
    int64_t swizzle_bytes,
    int64_t interleave_bytes,
    bool unpack_fp4 = false
) {
  TORCH_CHECK(base.defined(), "base must be defined");
  TORCH_CHECK(base.is_cuda(), "base must be a CUDA tensor");
  TORCH_CHECK(base.numel() > 0, "base must have storage");
  TORCH_CHECK(shape.size() == strides_bytes.size() + 1, "shape and strides_bytes must have same length");
  TORCH_CHECK(shape.size() == box_dim.size(), "shape and box_dim must have same length");
  TORCH_CHECK(shape.size() == elem_strides.size(), "shape and elem_strides must have same length");

  const int R = (int)shape.size();
  TORCH_CHECK(R >= 1 && R <= 5, "tensorRank=", R, " not supported here (adjust if needed)");

  // Allocate descriptor storage on device as opaque bytes
  auto desc = torch::empty({(int64_t)sizeof(CUtensorMap)},
                           torch::TensorOptions().dtype(torch::kUInt8));

  // Prepare arrays
  std::vector<cuuint64_t> gdim(5, 0);
  std::vector<cuuint64_t> gstride(5, 0);
  std::vector<cuuint32_t> bdim(5, 0);
  std::vector<cuuint32_t> estride(5, 0);

  for (int i = 0; i < R; i++) {
    TORCH_CHECK(shape[i] > 0, "shape[", i, "] must be > 0");
    TORCH_CHECK(box_dim[i] > 0, "box_dim[", i, "] must be > 0");
    TORCH_CHECK(elem_strides[i] > 0, "elem_strides[", i, "] must be > 0");
    gdim[i]    = (cuuint64_t)shape[i];
    bdim[i]    = (cuuint32_t)box_dim[i];
    estride[i] = (cuuint32_t)elem_strides[i];

    if (i < R - 1) {
      // TORCH_CHECK(strides_bytes[i] > 0, "strides_bytes[", i, "] must be > 0");
      gstride[i] = (cuuint64_t)strides_bytes[i];
    } else
      gstride[i] = (cuuint64_t)0; // last stride is not used by hardware, can be 0
  }

  CUtensorMapDataType dtype = unpack_fp4
      ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
      : to_dtype(base.scalar_type());
  CUtensorMapSwizzle swz = to_swizzle(swizzle_bytes);
  CUtensorMapInterleave interleave = to_interleave(interleave_bytes);

  CUtensorMapL2promotion l2p = select_tma_l2_promotion();
  CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // Fill descriptor in device memory
  CUtensorMap* tma = reinterpret_cast<CUtensorMap*>(desc.data_ptr<uint8_t>());

  CUresult r = cuTensorMapEncodeTiled(
      tma,
      dtype,
      (cuuint32_t)R,
      (void*)base.data_ptr(),
      gdim.data(),
      gstride.data(),
      bdim.data(),
      estride.data(),
      interleave,
      swz,
      l2p,
      oob
  );

  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", r);

  return desc;
}

enum CachePolicy : int {
  DAE_CACHE_NORMAL = cudaAccessPropertyNormal,
  DAE_CACHE_STREAMING = cudaAccessPropertyStreaming,
  DAE_CACHE_PERSISTING = cudaAccessPropertyPersisting
};

// Set cache policy for a CUDA tensor on the specified stream.
void py_reset_cache_policy(int64_t stream_id) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  auto err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute reset failed: ", cudaGetErrorString(err));
}

void py_tensor_set_cache_policy(
    torch::Tensor t,
    int64_t stream_id,
    float hit_ratio,
    int hit_policy,
    int miss_policy,
    int64_t num_bytes) {
  TORCH_CHECK(t.defined(), "Tensor must be defined");
  TORCH_CHECK(t.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(t.numel() > 0, "Tensor must have storage");

  // Get the current CUDA stream
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);

  cudaAccessPolicyWindow apw{};
  apw.base_ptr  = (void*)t.data_ptr();          // some device pointer
  const cudaDeviceProp prop = current_device_prop();

  const size_t tensor_bytes = (size_t)t.numel() * (size_t)t.element_size();
  size_t requested_bytes = tensor_bytes;
  if (num_bytes > 0) {
    requested_bytes = std::min(requested_bytes, static_cast<size_t>(num_bytes));
  }
  if (prop.accessPolicyMaxWindowSize > 0) {
    requested_bytes = std::min(requested_bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  }
  TORCH_CHECK(requested_bytes > 0, "cache window must be non-zero");
  apw.num_bytes = requested_bytes;
  apw.hitRatio  = hit_ratio;                    // 0..1

  apw.hitProp = static_cast<cudaAccessProperty>(hit_policy);
  apw.missProp = static_cast<cudaAccessProperty>(miss_policy);

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = apw;
  auto err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto op = m.def_submodule("opcode", "DAE2 OpCodes");
  #define DAE_OP(name, value) op.attr(#name) = (int)name;
  #include "dae/opcode.cuh.inc"
  #undef DAE_OP

  py::list compute_family_specs;
  #define DAE_OP(name, value)
  #define DAE_DEFINE_COMP_FAMILY(name, ...) { \
    py::dict spec; \
    spec["family"] = py::str(#name); \
    spec["definition"] = py::str(#__VA_ARGS__); \
    compute_family_specs.append(spec); \
  }
  #include "dae/opcode.cuh.inc"
  #undef DAE_OP
  #undef DAE_DEFINE_COMP_FAMILY
  m.attr("compute_family_specs") = compute_family_specs;

  py::list supported_compute_ops;
  #define DAE_COMPUTE_OP(name) supported_compute_ops.append(py::str(#name));
  #include "dae/selected_compute_ops.inc"
  #undef DAE_COMPUTE_OP
  m.attr("supported_compute_ops") = supported_compute_ops;

  auto config = m.def_submodule("config", "DAE2 Configuration Constants");
  config.attr("slot_size") = slotSizeKb * 1024;
  config.attr("num_slots") = numSlots;
  config.attr("max_insts") = numInsts;
  config.attr("load_instructions") = dae2LoadInstructions;
  config.attr("dynamic_smem_size") = dynamicSmemBytes;
  config.attr("num_profile_events") = numProfileEvents;
  config.attr("layer_profile_event_base") = layerProfileEventBase;
  config.attr("reload_profile_event_base") = reloadProfileEventBase;
  config.attr("track_profile_event_base") = trackProfileEventBase;
  config.attr("num_loop_counters") = numComputeLoopCounters;
  config.attr("max_tmas") = numTmas;
  config.attr("max_bars") = numBars;
  config.attr("num_special_slots") = numSpecialSlots;
  config.attr("instructions_in_shared") = dae2LoadInstructions;
  config.attr("nvfp4_umma_pipeline_stages") = nvfp4UmmaPipelineStages;
  config.attr("nvfp4_scale_copy_stages") = nvfp4ScaleCopyBarrierCount;
  config.attr("mxfp4_mxfp8_tma_scale_stages") =
      mxfp4Mxfp8TmaScaleStages;
  config.attr("mxfp4_mxfp8_direct_tma_enabled") =
      mxfp4Mxfp8DirectTmaEnabled;
  config.attr("mxfp_gate_up_direct_output") =
      mxfpGateUpDirectOutputEnabled;
  config.attr("mxfp_gate_up_direct_activation") =
      mxfpGateUpDirectActivationEnabled;
  config.attr("mxfp_gate_up_ldu_weight_ring") =
      mxfpGateUpLduWeightRingEnabled;
  config.attr("mxfp_gate_up_direct_activation_tiles") =
      mxfpGateUpDirectActivationTiles;
  config.attr("mxfp_gate_up_fixed_output_rows") =
      mxfpGateUpFixedOutputRows;
  config.attr("mxfp_gate_up_fixed_bf16_epilogue") =
      mxfpGateUpFixedBf16Epilogue;

  // auto flag = m.def_submodule("flag", "DAE2 Instruction Flags");
  // flag.attr("jump") = MEM_OP_FLAGS_JUMP;
  // flag.attr("writeback") = MEM_OP_FLAGS_WRITEBACK;
  // flag.attr("group") = MEM_OP_FLAGS_GROUP;
  // flag.attr("barrier") = MEM_OP_FLAGS_BARRIER;
  // flag.attr("port") = MEM_OP_FLAGS_PORT;

  // auto cache = m.def_submodule("cache_policy", "DAE2 Cache Policy Constants");
  // cache.attr("normal") = DAE_CACHE_NORMAL;
  // cache.attr("streaming") = DAE_CACHE_STREAMING;
  // cache.attr("persisting") = DAE_CACHE_PERSISTING;

  m.def("set_smem_size", &py_set_smem_size,
            "Set dynamic shared memory size for DAE2 kernel");
  m.def("launch_dae", &py_launch_dae,
            py::arg("num_sms"),
            py::arg("smem_size"),
            py::arg("compute_insts_bytes"),
            py::arg("memory_insts_bytes"),
            py::arg("tma_descs_bytes"),
            py::arg("bars_int32"),
            py::arg("profile_u64"),
            py::arg("initial_loop_counts"),
            py::arg("stream"),
            py::arg("synchronize") = true,
            "Launch DAE2 kernel with given parameters");
#if defined(DAE_FFN_SPECIALIZED_KERNELS)
  m.def("launch_dae_ffn_linear1_direct",
            &py_launch_dae_ffn_linear1_direct,
            py::arg("num_blocks"), py::arg("smem_size"),
            py::arg("metadata_bytes"), py::arg("tma_descs_bytes"),
            py::arg("bars_int32"), py::arg("reduction_bar_base"),
            py::arg("reduction_tiles"), py::arg("profile_u64"),
            py::arg("stream"),
            "Launch the focused native-record FFN Linear-1 entrypoint");
  m.def("launch_dae_ffn_down_direct",
            &py_launch_dae_ffn_down_direct,
            py::arg("num_blocks"), py::arg("smem_size"),
            py::arg("metadata_bytes"), py::arg("tma_descs_bytes"),
            py::arg("bars_int32"), py::arg("profile_u64"),
            py::arg("stream"),
            "Launch the focused native-record FFN down entrypoint");
#endif
  m.def("launch_dae", &py_launch_dae_legacy,
            py::arg("num_sms"),
            py::arg("smem_size"),
            py::arg("compute_insts_bytes"),
            py::arg("memory_insts_bytes"),
            py::arg("tma_descs_bytes"),
            py::arg("bars_int32"),
            py::arg("profile_u64"),
            py::arg("stream"),
            "Launch DAE2 with zero loop counters and synchronous completion");
  m.def("launch_dae_sequence", &py_launch_dae_sequence,
            py::arg("num_sms"),
            py::arg("smem_size"),
            py::arg("compute_insts_bytes"),
            py::arg("memory_insts_bytes"),
            py::arg("tma_descs_bytes"),
            py::arg("bars_int32"),
            py::arg("profile_u64"),
            py::arg("sequence_loop_counts"),
            py::arg("stream"),
            py::arg("synchronize") = true,
            "Launch a sequence of independent DAE2 kernels with one host dispatch");
  m.def("build_tma_desc", &py_build_tma_desc,
            py::arg("base"),
            py::arg("shape"),
            py::arg("strides_bytes"),
            py::arg("box_dim"),
            py::arg("elem_strides"),
            py::arg("swizzle_bytes"),
            py::arg("interleave_bytes"),
            py::arg("unpack_fp4") = false,
            "Build CUtensorMap descriptor for given tensor and layout");
  m.def("reset_cache_policy", &py_reset_cache_policy,
            py::arg("stream"),
            "Clear the access-policy window on the specified CUDA stream");
  m.def("set_cache_policy", &py_tensor_set_cache_policy,
            py::arg("tensor"),
            py::arg("stream"),
            py::arg("hit_ratio"),
            py::arg("hit_policy"),
            py::arg("miss_policy"),
            py::arg("num_bytes") = -1,
            "Set cache policy for a CUDA tensor on the specified stream");
  m.def("prepack_nvfp4_checkpoint", &py_prepack_nvfp4_checkpoint,
            py::arg("weight"),
            py::arg("checkpoint_scale"),
            py::arg("output"),
            "Convert one raw checkpoint NVFP4 linear to native SM100 tiles");
  m.def("prepack_nvfp4_checkpoint_k512_split",
            &py_prepack_nvfp4_checkpoint_k512_split,
            py::arg("weight"),
            py::arg("checkpoint_scale"),
            py::arg("output_data"),
            py::arg("output_scale"),
            "Pack one NVFP4 linear into split K512 data and scale records");
  m.def("prepack_nvfp4_activation_k512_split",
            &py_prepack_nvfp4_activation_k512_split,
            py::arg("activation"),
            py::arg("checkpoint_scale"),
            py::arg("output_data"),
            py::arg("output_scale"),
            "Pack one NVFP4 activation into split K512 data and scale records");
  m.def("prepack_fp8_checkpoint", &py_prepack_fp8_checkpoint,
            py::arg("weight"),
            py::arg("checkpoint_scale"),
            py::arg("output"),
            py::arg("scale_pack") = 1,
            "Convert one raw checkpoint FP8 linear to native SM100 tiles");
}
