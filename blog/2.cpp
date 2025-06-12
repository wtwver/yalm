/* PSUEDOCODE */

// InferenceState is the minimum set of buffers needed to
// hold state during the forward pass and exists to avoid
// extra allocations
void Model::forward(InferenceState& s, int token) {
  // The embedding table maps token IDs to embedding vectors,
  // which are copied into a buffer of the inference state
  s.x = copy_embedding(token, this->token_embedding_table);
  
  // Models consist of a sequence of transformer blocks which
  // mutate the inference state in order
  for (Block& b : this->blocks) {
    b->block(s);
  }
  
  // Usually there is a layer norm right before the final classifier
  s.x = layernorm(s.x, this->lm_head_prenorm_weights);
  // Typically we end with a linear transform from (dim) -> (vocab_size)
  s.logits = linear(s.x, this->lm_head_classifier_weights); 
}

void Block::block(InferenceState& s) {
  s.x_resid = layernorm(s.x, this->att_prenorm_weights);
  // Multi-head attention typically includes: 
  // 1. RoPE on input (element-wise mutation w/ sines/cosines)
  // 2. QKV matmuls and updating the KV cache
  // 3. Causal self-attention, softmax, and value mixing
  // 4. Projection back into the residual stream
  s.x_resid = multi_head_attn(
    s.x_resid,
    this->wq, 
    this->wk, 
    this->wv, 
    this->key_cache,
    this->value_cache
  );
  s.x += s.x_resid;
  s.x_resid = layernorm(s.x, this->ffn_prenorm_weights);
  // On modern architectures like Llama, this is a GLU feedforward 
  // with 3 linear transforms, not a simple MLP:
  // -> w2(F.silu(w1(x)) * w3(x))
  // Some architectures also split the FFN into a mixture of experts.
  s.x_resid = ffn(s.x_resid, this->w1, this->w2, this->w3);
  s.x += s.x_resid;
}
        