
__device__ 
inline float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);

  return val;
}

__global__
void att_mix(
  const float* vb, // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 1-D and blockDim.x == WARP_SIZE
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int i = blockIdx.y;
  int offset = threadIdx.x;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const float* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  float sum = 0.0;
  for (int t = offset; t < seq_len; t += WARP_SIZE) {
    sum += vh[kv_stride * t + i] * atth[t];
  }
  sum = warp_reduce_sum(sum);
  if (offset == 0) outh[i] = sum;
}

/* usage */
dim3 tpb;
tpb.x = WARP_SIZE;
dim3 blocks;
blocks.x = n_heads;
blocks.y = head_dim;
att_mix<<<blocks, tpb>>>(
  vb, att, 
  head_dim, n_heads, n_kv_heads, 
  seq_len, max_seq_len, out
);
      