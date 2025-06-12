__global__
void att_mix(
  const float* vb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 2-D (warp_size, t_stride)
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const float* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int warp_id = threadIdx.y;
  int t_stride = blockDim.y;
  
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  
  for (int i = threadIdx.x; i < head_dim; i += warpSize) {
    if (warp_id == 0) {
      shared[threadIdx.x] = 0;
    }
    __syncthreads();
    float sum = 0.0;
    for (int t = warp_id; t < seq_len; t += t_stride) {
      sum += vh[kv_stride * t + i] * atth[t];	
    }
    atomicAdd(&shared[threadIdx.x], sum);
    __syncthreads();
    if (warp_id == 0) {
      outh[i] = shared[threadIdx.x];
      shared[threadIdx.x] = 0;
    }
  }
}

dim3 tpb;
tpb.x = warp_size;
tpb.y = min(kv_len, max_threads_per_block / warp_size);
dim3 blocks;
blocks.x = c.n_heads;
att_mix<<<blocks, tpb>>>(
  vb, s.att(),
  c.head_dim, c.n_heads, c.n_kv_heads, 
  kv_len, c.max_seq_len, s.xb2()
);