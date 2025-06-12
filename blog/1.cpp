/* PSUEDOCODE */

void generate(Model& model, std::string prompt, int steps) {
  std::vector<int> encoded = tokenizer.encode(prompt);
  InferenceState s(model);
  
  // 1. Prefill step: Forward the model on each prompt token, discarding 
  // the output. This lets the model read the prompt and hydrates the KV 
  // cache.
  for (int token : encoded) {
    model.forward(s, token);
  }
  // 2. Decode step: Forward the model repeatedly, generating 1 token at a time.
  for (int i = 0; i < steps; i++) {
    model.forward(s, encoded.back());
    int next_token = sampler.sample(s.logits);
    encoded.push_back(next_token);
    std::cout << tokenizer.decode_one(next_token) << std::flush;
    if (next_token == tokenizer.EOS) {
      break;
    }
  }
}