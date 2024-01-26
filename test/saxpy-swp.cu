// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuda/pipeline>

// a monolithic kernel, because it assumes a single large grid of threads to
// process the entire array in one pass
__global__ void saxpy_kernel(int n, float a, float* x, float* y, float* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#if 0
  if (i < n)
    printf(
      "threadIdx.x=%d, blockIdx=%d, blockDim.x=%d, offset=%d\n",
      threadIdx.x, blockIdx.x,
      blockDim.x,
      i);
#endif
  if (i < n)
    z[i] = a * x[i] + y[i];
}

// The stride of the loop is blockDim.x * gridDim.x is the total number of
// threads in the grid When you limit the number of blocks in your grid, threads
// are reused for multiple computations.
__global__ void
saxpy_stride_kernel(int n, float a, float* x, float* y, float* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#if 0
  if (i < n)
    printf(
      "threadIdx.x=%d, blockIdx=%d, blockDim.x=%d, gridDim.x=%d, offset=%d\n",
      threadIdx.x, blockIdx.x,
      blockDim.x,
      gridDim.x,
      i);
#endif
  for (; i < n; i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

// Assume each invocation handles BLOCK_SIZE elements for all threads
// cooperatively.
// template <int BLOCK_SIZE>
__global__ void saxpy_pipeline_kernel(
    int n,
    float a,
    float* X,
    float* Y,
    float* Z,
    int BLOCK_SIZE) {
  // BLOCK_SIZE is number of elements handled for one invocation of a thread
  // block. Multi-stage pipeline version
  constexpr size_t maxPipelineStages = 2;
  constexpr size_t xStep = 1024; // each iteration handles 1024 elements 256*4
  __shared__ alignas(alignof(float4)) float xs[maxPipelineStages][xStep];
  __shared__ alignas(alignof(float4)) float ys[maxPipelineStages][xStep];

  int xBegin = BLOCK_SIZE * blockIdx.x;
  int xEnd = xBegin + BLOCK_SIZE;
  if (xEnd > n)
    xEnd = n;

#if 0
  if (xBegin < n)
    printf(
      "threadIdx.x=%d, blockIdx=%d, blockDim.x=%d, gridDim.x=%d, offset=%d\n",
      threadIdx.x, blockIdx.x,
      blockDim.x,
      gridDim.x,
      xBegin);
#endif

  // try to use vectorized load where each thread loads 4 elements, and some
  // threads will not load anything.
  const int t4x = threadIdx.x * 4;
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Each thread loads 4 elements
  for (int x = xBegin, xStage = xBegin, i = 0, iStage = 0; x < xEnd;
       x += xStep, ++i) {
    // first iteration of outer loop, inner loop runs maxPipelineStages
    // iterations afterwards, inner loop only run one iteration
    for (; xStage <= x + xStep * maxPipelineStages; xStage += xStep, ++iStage) {
      pipe.producer_acquire();
      // All threads populate the smem together.
      if (xStage < xEnd && t4x < xStep) {
        // Rotating buffer
        const int j = iStage % maxPipelineStages;
        cuda::memcpy_async(&xs[j][t4x], &X[xStage + t4x], shape4, pipe);
        cuda::memcpy_async(&ys[j][t4x], &Y[xStage + t4x], shape4, pipe);
      }
      pipe.producer_commit();
    }
    pipe.consumer_wait(); // Wait for ‘subset’ stage to be available
    // Synchronize to make sure values are loaded?
    __syncthreads();

    // Rotating buffer
    const int j = i % maxPipelineStages;
    float z0 = a * xs[j][t4x] + ys[j][t4x];
    float z1 = a * xs[j][t4x + 1] + ys[j][t4x + 1];
    float z2 = a * xs[j][t4x + 2] + ys[j][t4x + 2];
    float z3 = a * xs[j][t4x + 3] + ys[j][t4x + 3];
    pipe.consumer_release();
    Z[x + t4x] = z0;
    Z[x + t4x + 1] = z1;
    Z[x + t4x + 2] = z2;
    Z[x + t4x + 3] = z3;
  }
}

void saxpy_stub(int n, float a, float* x, float* y, float* z, int8_t config) {
  // number of blocks, number of threads
  if (config == 0)
    saxpy_kernel<<<(n + 511) / 512, 512>>>(n, a, x, y, z);
  else if (config == 1) {
    int numSMs = 8;
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    saxpy_stride_kernel<<<32 * numSMs, 256>>>(n, a, x, y, z);
    // 32x8, 256 --> blockDim.x is 256, gridDim.x is 256, threadIdx from 0 to
    // 255
  }
  if (config == 2) {
    int gridD = 256;
    int BLOCK_SIZE = n / gridD; // 2^16 for 16M elements vs 4096 for 1M elements
                                // vs for 2^7 for 2^15 elements
    saxpy_pipeline_kernel<<<gridD, 256>>>(n, a, x, y, z, BLOCK_SIZE);
    // gridDim should be 32, blockDim is 256, threadIdx from 0 to 255
    // each invocation across one thread block handles BLOCK_SIZE elements
    // each iteration of the kernel invocation across one thread block handles
    // 1024 elements
  }
}

at::Tensor saxpy(at::Tensor x, at::Tensor y, double a) {
  auto z = at::empty_like(x);
  auto n = x.numel();
  saxpy_stub(
      n, a, x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), 0);
  return z;
}
at::Tensor saxpy1(at::Tensor x, at::Tensor y, double a) {
  auto z = at::empty_like(x);
  auto n = x.numel();
  saxpy_stub(
      n, a, x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), 1);
  return z;
}
at::Tensor saxpy2(at::Tensor x, at::Tensor y, double a) {
  auto z = at::empty_like(x);
  auto n = x.numel();
  saxpy_stub(
      n, a, x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), 2);
  return z;
}

TORCH_LIBRARY(saxpy, m) {
  m.def("saxpy(Tensor x, Tensor y, float a) -> Tensor");
  m.impl("saxpy", saxpy);
  m.def("saxpy1(Tensor x, Tensor y, float a) -> Tensor");
  m.impl("saxpy1", saxpy1);
  m.def("saxpy2(Tensor x, Tensor y, float a) -> Tensor");
  m.impl("saxpy2", saxpy2);
}
