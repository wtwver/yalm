__device__ inline float blocktranspose(float v, float def) {
  // Performs block-and-warp transpose operation:
  //   For a block containing K warps where lane 0 contains val_k,
  //   this function returns:
  //   - For warp 0, lane K: val_k
  //   - For all other warps and lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  
  // Will hold results of all warps.
  // Each lane of the warp accumulates across 1 head element at a time.
  // NOTE: Assumes warpSize is 32
  __shared__ float sm[32];
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  
  return lane < blockDim.x / warpSize ? sm[lane] : def;
}

template <typename T>
__global__
void matmul_wide(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] = rowSum;
  }
}