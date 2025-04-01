import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 


from src.data_collection.testing.model_testing_class.model_tester import ModelTester

start = time.perf_counter()

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
)

tester = ModelTester(model_name=model_name, model=model, device='cpu')
output = tester.generate_text("based on financial report this company")
print(output)

end = time.perf_counter()
print(f"Execution time: {end - start:.2f} seconds")