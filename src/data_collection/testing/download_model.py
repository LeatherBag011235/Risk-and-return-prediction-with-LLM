from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

login(token="")

model_name = "google/gemma-7b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Downloaded:
# mistralai/Mixtral-8x7B-v0.1
# mistralai/Mistral-7B-v0.1
# 01-ai/Yi-34B
# google/gemma-7b-it