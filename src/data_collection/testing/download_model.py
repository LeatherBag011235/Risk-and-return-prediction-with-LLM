from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

login(token="")

model_name = "mistralai/Mixtral-8x7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
