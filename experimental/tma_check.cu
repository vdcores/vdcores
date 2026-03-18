#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  printf("=== Confirming Stride Pattern ===\n\n");
  
  // Initialize
  cudaFree(0);
  cuInit(0);
  
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, 0);
  
  CUcontext cuContext;
  cuCtxCreate(&cuContext, 0, cuDevice);
  
  // Allocate memory
  float* d_ptr;
  cudaMalloc(&d_ptr, 4096 * sizeof(float));
  
  CUtensorMap desc;
  CUresult cuResult;
  
  // Confirm the pattern with different sizes
  printf("Test 1: 2D [8, 8], strides=[32, 4] (row_stride, elem_stride)\n");
  {
    cuuint64_t dims[2] = {8, 8};
    cuuint64_t strides[2] = {32, 4};  // {8*sizeof(float), sizeof(float)}
    cuuint32_t box[2] = {8, 8};
    cuuint32_t elem[2] = {1, 1};
    
    cuResult = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)d_ptr,
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  Result: %s\n\n", cuResult == 0 ? "✓ SUCCESS" : "✗ FAILED");
  }
  
  printf("Test 2: 2D [32, 32], strides=[128, 4], box=[16, 16]\n");
  {
    cuuint64_t dims[2] = {32, 32};
    cuuint64_t strides[2] = {128, 4};  // {32*sizeof(float), sizeof(float)}
    cuuint32_t box[2] = {16, 16};
    cuuint32_t elem[2] = {1, 1};
    
    cuResult = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)d_ptr,
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  Result: %s\n\n", cuResult == 0 ? "✓ SUCCESS" : "✗ FAILED");
  }
  
  printf("Test 3: 2D [64, 32], strides=[256, 4], box=[8, 8]\n");
  {
    cuuint64_t dims[2] = {64, 32};
    cuuint64_t strides[2] = {256, 4};  // {64*sizeof(float), sizeof(float)}
    cuuint32_t box[2] = {8, 8};
    cuuint32_t elem[2] = {1, 1};
    
    cuResult = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)d_ptr,
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  Result: %s\n\n", cuResult == 0 ? "✓ SUCCESS" : "✗ FAILED");
  }
  
  printf("Test 4: Non-square 2D [16, 32], strides=[64, 4], box=[8, 16]\n");
  {
    cuuint64_t dims[2] = {16, 32};
    cuuint64_t strides[2] = {64, 4};  // {16*sizeof(float), sizeof(float)}
    cuuint32_t box[2] = {8, 16};
    cuuint32_t elem[2] = {1, 1};
    
    cuResult = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)d_ptr,
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  Result: %s\n\n", cuResult == 0 ? "✓ SUCCESS" : "✗ FAILED");
  }
  
  printf("Test 5: With 32B swizzle - 2D [16, 16], strides=[64, 4]\n");
  {
    cuuint64_t dims[2] = {16, 16};
    cuuint64_t strides[2] = {64, 4};
    cuuint32_t box[2] = {16, 16};
    cuuint32_t elem[2] = {1, 1};
    
    cuResult = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)d_ptr,
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,  // With swizzle
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  Result: %s\n\n", cuResult == 0 ? "✓ SUCCESS" : "✗ FAILED");
  }
  
  cudaFree(d_ptr);
  cuCtxDestroy(cuContext);
  
  return 0;
}