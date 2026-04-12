#include "opt_attention_params.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <optional>

namespace opt_attention {
void launch_decode(const OptAttentionParams& params, at::ScalarType dtype);
}

namespace {

void check_qkv(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "query must be float16 or bfloat16");
  TORCH_CHECK(key.scalar_type() == query.scalar_type(), "key dtype must match query dtype");
  TORCH_CHECK(value.scalar_type() == query.scalar_type(), "value dtype must match query dtype");
  TORCH_CHECK(query.dim() == 4, "query must have shape [B, H, 1, D]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [B, H, S, D]");
  TORCH_CHECK(value.dim() == 4, "value must have shape [B, H, S, D]");
  TORCH_CHECK(query.size(2) == 1, "decode kernel requires q_len == 1");
  TORCH_CHECK(query.size(3) == opt_attention::kHeadDim, "decode kernel requires head_dim == 128");
  TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
  TORCH_CHECK(key.size(0) == query.size(0), "key batch must match query batch");
  TORCH_CHECK(key.size(1) == query.size(1), "key heads must match query heads");
  TORCH_CHECK(key.size(3) == query.size(3), "key head_dim must match query head_dim");
  TORCH_CHECK(query.stride(3) == 1, "query head_dim stride must be 1");
  TORCH_CHECK(key.stride(3) == 1, "key head_dim stride must be 1");
  TORCH_CHECK(value.stride(3) == 1, "value head_dim stride must be 1");
}

void check_mask(const torch::Tensor& mask, const torch::Tensor& query, const torch::Tensor& key) {
  TORCH_CHECK(mask.is_cuda(), "attention_mask must be CUDA");
  TORCH_CHECK(mask.scalar_type() == torch::kFloat32, "attention_mask must be float32");
  TORCH_CHECK(mask.dim() == 4, "attention_mask must have shape [B, 1, 1, S]");
  TORCH_CHECK(mask.size(0) == query.size(0), "attention_mask batch must match query batch");
  TORCH_CHECK(mask.size(1) == 1 && mask.size(2) == 1, "attention_mask must have singleton head and query dims");
  TORCH_CHECK(mask.size(3) == key.size(2), "attention_mask sequence length must match key sequence length");
  TORCH_CHECK(mask.stride(3) == 1 || mask.size(3) == 1, "attention_mask last dimension must be contiguous");
}

void check_static_cache_fast_path(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t split_size) {
  constexpr uintptr_t kVectorBytes = 16;
  const int64_t elems_per_vec = static_cast<int64_t>(kVectorBytes / query.element_size());

  TORCH_CHECK(key.size(2) % opt_attention::kKvTile == 0,
              "fast OPT StaticCache path requires key sequence length to be a multiple of 64");
  TORCH_CHECK(split_size % opt_attention::kKvTile == 0,
              "fast OPT StaticCache path requires split_size to be a multiple of 64");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(key.data_ptr()) % kVectorBytes == 0,
              "key data pointer must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % kVectorBytes == 0,
              "value data pointer must be 16-byte aligned");
  TORCH_CHECK(key.stride(0) % elems_per_vec == 0 && key.stride(1) % elems_per_vec == 0 &&
                  key.stride(2) % elems_per_vec == 0,
              "key batch/head/sequence strides must preserve 16-byte row alignment");
  TORCH_CHECK(value.stride(0) % elems_per_vec == 0 && value.stride(1) % elems_per_vec == 0 &&
                  value.stride(2) % elems_per_vec == 0,
              "value batch/head/sequence strides must preserve 16-byte row alignment");
}

void check_static_cache_tma_path(const torch::Tensor& key, const torch::Tensor& value) {
  TORCH_CHECK(key.stride(3) == 1 && value.stride(3) == 1,
              "TMA path requires StaticCache [B,H,S,D] with contiguous D");
  TORCH_CHECK(key.stride(2) == opt_attention::kHeadDim &&
                  value.stride(2) == opt_attention::kHeadDim,
              "TMA path requires StaticCache [B,H,S,D] with S stride == D");
  TORCH_CHECK(key.stride(1) == key.size(2) * opt_attention::kHeadDim &&
                  value.stride(1) == value.size(2) * opt_attention::kHeadDim,
              "TMA path requires StaticCache [B,H,S,D] with H stride == S*D");
  TORCH_CHECK(key.stride(0) == key.size(1) * key.size(2) * opt_attention::kHeadDim &&
                  value.stride(0) == value.size(1) * value.size(2) * opt_attention::kHeadDim,
              "TMA path requires contiguous StaticCache [B,H,S,D] layout");
}

void check_driver(CUresult result, const char* call) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = "unknown";
  const char* text = "unknown";
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &text));
  TORCH_CHECK(false, call, " failed: ", name, ": ", text);
}

CUtensorMapDataType tensor_map_dtype(at::ScalarType dtype) {
  if (dtype == torch::kFloat16) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  }
  if (dtype == torch::kBFloat16) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  }
  TORCH_CHECK(false, "unsupported TMA dtype");
}

void encode_static_cache_tma_desc(CUtensorMap* desc, const torch::Tensor& tensor) {
  const auto elem_size = static_cast<uint64_t>(tensor.element_size());
  const uint64_t global_dim[2] = {
      static_cast<uint64_t>(opt_attention::kHeadDim),
      static_cast<uint64_t>(tensor.size(0) * tensor.size(1) * tensor.size(2)),
  };
  const uint64_t global_strides[1] = {
      static_cast<uint64_t>(opt_attention::kHeadDim) * elem_size,
  };
  const uint32_t box_dim[2] = {
      static_cast<uint32_t>(opt_attention::kHeadDim),
      static_cast<uint32_t>(opt_attention::kKvTile),
  };
  const uint32_t element_strides[2] = {1, 1};

  check_driver(cuTensorMapEncodeTiled(
                   desc,
                   tensor_map_dtype(tensor.scalar_type()),
                   2,
                   const_cast<void*>(tensor.data_ptr()),
                   global_dim,
                   global_strides,
                   box_dim,
                   element_strides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_NONE,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled");
}

}  // namespace

torch::Tensor decode(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::optional<torch::Tensor> attention_mask,
    double scaling,
    int64_t split_size) {
  check_qkv(query, key, value);
  TORCH_CHECK(split_size > 0, "split_size must be positive");
  check_static_cache_fast_path(query, key, value, split_size);
  if constexpr (opt_attention::kUseTensorTma) {
    check_static_cache_tma_path(key, value);
  }

  const bool has_mask = attention_mask.has_value() && attention_mask->defined();
  if (has_mask) {
    check_mask(*attention_mask, query, key);
  }

  auto output = torch::empty(
      {query.size(0), 1, query.size(1), query.size(3)},
      query.options());
  const int64_t num_splits = (key.size(2) + split_size - 1) / split_size;
  auto partial_options = query.options().dtype(torch::kFloat32);
  torch::Tensor partial_out;
  torch::Tensor partial_m;
  torch::Tensor partial_l;
  if (num_splits > 1) {
    partial_out = torch::empty({query.size(0), query.size(1), num_splits, opt_attention::kHeadDim}, partial_options);
    partial_m = torch::empty({query.size(0), query.size(1), num_splits}, partial_options);
    partial_l = torch::empty({query.size(0), query.size(1), num_splits}, partial_options);
  }

  opt_attention::OptAttentionParams params{};
  params.query = query.data_ptr();
  params.key = key.data_ptr();
  params.value = value.data_ptr();
  params.mask = has_mask ? attention_mask->data_ptr<float>() : nullptr;
  params.output = output.data_ptr();
  params.partial_out = num_splits > 1 ? partial_out.data_ptr<float>() : nullptr;
  params.partial_m = num_splits > 1 ? partial_m.data_ptr<float>() : nullptr;
  params.partial_l = num_splits > 1 ? partial_l.data_ptr<float>() : nullptr;
  params.tma_descs = nullptr;
  params.batch_size = static_cast<int>(query.size(0));
  params.num_heads = static_cast<int>(query.size(1));
  params.key_seq_len = static_cast<int>(key.size(2));
  params.num_splits = static_cast<int>(num_splits);
  params.split_size = static_cast<int>(split_size);
  params.scaling = static_cast<float>(scaling);

  params.q_stride_b = query.stride(0);
  params.q_stride_h = query.stride(1);
  params.q_stride_q = query.stride(2);
  params.q_stride_d = query.stride(3);

  params.k_stride_b = key.stride(0);
  params.k_stride_h = key.stride(1);
  params.k_stride_s = key.stride(2);
  params.k_stride_d = key.stride(3);

  params.v_stride_b = value.stride(0);
  params.v_stride_h = value.stride(1);
  params.v_stride_s = value.stride(2);
  params.v_stride_d = value.stride(3);

  params.o_stride_b = output.stride(0);
  params.o_stride_q = output.stride(1);
  params.o_stride_h = output.stride(2);
  params.o_stride_d = output.stride(3);

  if (has_mask) {
    params.m_stride_b = attention_mask->stride(0);
    params.m_stride_s = attention_mask->stride(3);
  }

  torch::Tensor tma_storage;
  if constexpr (opt_attention::kUseTensorTma) {
    check_driver(cuInit(0), "cuInit");
    std::array<CUtensorMap, 2> host_descs{};
    encode_static_cache_tma_desc(&host_descs[0], key);
    encode_static_cache_tma_desc(&host_descs[1], value);

    const auto desc_align = static_cast<uintptr_t>(alignof(CUtensorMap));
    tma_storage = torch::empty(
        {static_cast<int64_t>(host_descs.size() * sizeof(CUtensorMap) + desc_align)},
        query.options().dtype(torch::kUInt8));
    const uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(tma_storage.data_ptr());
    const uintptr_t aligned_ptr = (raw_ptr + desc_align - 1) & ~(desc_align - 1);
    params.tma_descs = reinterpret_cast<const CUtensorMap*>(aligned_ptr);
    C10_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(aligned_ptr),
        host_descs.data(),
        host_descs.size() * sizeof(CUtensorMap),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
  }

  opt_attention::launch_decode(params, query.scalar_type());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "OPT StaticCache decode attention");
  m.def("uses_tma", []() { return opt_attention::kUseTensorTma; }, "Whether the extension was built with Tensor TMA loads");
}
