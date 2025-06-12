__device__ 
inline float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);

  return val;
}

__device__
inline float matmul_row(const float* row, const float* x, int offset, int dim) {
  float sum = 0.0;
  for (int j = offset; j < dim; j += WARP_SIZE) {
    float v = row[j] * x[j];
    sum += v;
  }
  return warp_reduce_sum(sum);
}

__global__
void matmul(const float* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Blocks are 1-D and same size as warp.
  int i = blockIdx.x;
  if (i >= d) return;
  int offset = threadIdx.x;
  float rowSum = matmul_row(&A[n * i], x, offset, n);
  if (threadIdx.x == 0) {
    out[i] = rowSum;
  }
}

/* usage */
int BLOCK_SIZE = WARP_SIZE;
matmul<<<d, BLOCK_SIZE>>>(A, x, n, d, out);
        