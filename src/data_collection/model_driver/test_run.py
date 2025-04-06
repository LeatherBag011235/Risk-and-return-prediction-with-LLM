import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 


from src.data_collection.model_driver.model_driver_class import ModelDriver

start = time.perf_counter()

model_name = "mistralai/Mixtral-8x7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype=torch.bfloat16,
)

tester = ModelDriver(model_name=model_name, model=model)

tester.generate_text("bla bla")


end = time.perf_counter()
print(f"Execution time: {end - start:.2f} seconds")