from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

# Replace with your actual token
#login(token="")

# Use the instruction-tuned Mistral model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Load model in 8-bit quantization (for RTX 4080 Super)
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model fully into GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Run inference with a chatbot-style prompt
prompt = "Explain quantum mechanics in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(output[0], skip_special_tokens=True))
