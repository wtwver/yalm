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
  // PRECOND: blocks are 1-D and `out` has been zeroed
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const float* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int t_per_thread = seq_len / gridDim.y;
  int t_start = blockIdx.y * t_per_thread;
  
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0;
    for (int t = t_start; t < t_start + t_per_thread; t++) {
      sum += vh[kv_stride * t + i] * atth[t];	
    }
    atomicAdd(&outh[i], sum);
  }
}

int max_t_per_thread = 256;

dim3 tpb;
tpb.x = warp_size;
dim3 blocks;
blocks.x = c.n_heads;
blocks.y = (kv_len + max_t_per_thread - 1) / max_t_per_thread;
cudaMemset(s.xb2(), 0, c.n_heads * c.head_dim * sizeof(float));
att_mix<<<blocks, tpb>>>(
  vb, s.att(),
  c.head_dim, c.n_heads, c.n_kv_heads, 
  kv_len, c.max_seq_len, s.xb2()
);