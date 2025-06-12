import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mistral_models_path = "../Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(mistral_models_path)

model = AutoModelForCausalLM.from_pretrained(mistral_models_path, torch_dtype=torch.float16)
model.to("cuda")

# short prompt
model_inputs = tokenizer(["Q: What is the meaning of life?"], return_tensors="pt").to("cuda")

# do a warmup inference
generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=True)[0].tolist()

# benchmark
start_time = time.time() 
generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)[0].tolist()
# decode with mistral tokenizer
result = tokenizer.decode(generated_ids)
end_time = time.time()

elapsed_s = end_time - start_time
print(result)

num_prompt_tokens = model_inputs["input_ids"].shape[-1]
num_generated_tokens = len(generated_ids) - num_prompt_tokens

print(f"Generation stats:\n" +
      f"  prompt: {num_prompt_tokens} tokens\n" +
      f"  generated: {num_generated_tokens} tokens\n" +
      f"  throughput: {num_generated_tokens/elapsed_s}tok/s\n" +
      f"  latency: {elapsed_s/num_generated_tokens}s/tok\n" +
      f"  total: {elapsed_s}s\n")
        