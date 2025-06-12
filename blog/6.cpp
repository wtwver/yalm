// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
// Note this is a feedforward with a GLU, not a simple MLP.
matmul(s.hb(), s.xb(), w1<T>(), c.dim, c.hidden_dim);
matmul(s.hb2(), s.xb(), w3<T>(), c.dim, c.hidden_dim);
for (int i = 0; i < c.hidden_dim; ++i) {
  s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
}
matmul(s.xb2(), s.hb(), w2<T>(), c.hidden_dim, c.dim);
// residual connection back into x
for (int i = 0; i < c.dim; ++i) {
  s.x()[i] += s.xb2()[i];
}