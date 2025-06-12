
// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
// Note this is a feedforward with a GLU, not a simple MLP.
matmul<<<c.hidden_dim, WARP_SIZE>>>(
  w1(), s.xb(), c.dim, c.hidden_dim, s.hb()
);
matmul<<<c.hidden_dim, WARP_SIZE>>>(
  w3(), s.xb(), c.dim, c.hidden_dim, s.hb2()
);
glu_gelu<<<
  (c.hidden_dim + MAX_THREADS_PER_BLOCK - 1)/MAX_THREADS_PER_BLOCK, 
  MAX_THREADS_PER_BLOCK,
>>>(
  s.hb(), s.hb2(), s.hb()
);
matmul<<<c.dim, WARP_SIZE>>>(
  w2(), s.hb(), c.hidden_dim, c.dim, s.xb2()
);
// ffn residual back into x
add_residuals<<<
  (c.dim + MAX_THREADS_PER_BLOCK - 1)/MAX_THREADS_PER_BLOCK,
  MAX_THREADS_PER_BLOCK
>>>(
  s.x(), s.xb2(), c.dim, s.x()
);