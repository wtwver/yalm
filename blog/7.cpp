
__global__
void matmul(const float* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float sum = 0.0;
  for (int j = 0; j < n; j++) {
    sum += A[n * i + j] * x[j];
  }
  out[i] = sum;
}

/* usage */
int MAX_THREADS_PER_BLOCK = 1024;
matmul<<<
  (d + MAX_THREADS_PER_BLOCK - 1)/MAX_THREADS_PER_BLOCK, 
  MAX_THREADS_PER_BLOCK
>>>(A, x, n, d, out);