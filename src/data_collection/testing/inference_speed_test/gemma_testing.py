import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 

from src.data_collection.model_driver.model_driver_class import ModelDriver

verolizer = {
    'positive': ['buy'],
    'negative': ['sell'],
}

model_name = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "8GiB", "cpu": "85GiB"},  # adjust based on actual memory
)

tester = ModelDriver(model_name=model_name, model=model, verbolizer=verolizer)

reports: list[str] = tester.get_random_reports(n=5)

avg_durations = []

for report in reports:
    scores, avg_duration = tester.compute_sample_scores(text=report)
    avg_durations.append(avg_duration)
    print(scores)
    print('\n', f"avg_inference: {avg_duration}, '\n")

avg_time_to_infer = sum(avg_durations) / len(avg_durations)
print(f"average time per segment: {avg_time_to_infer}")

# avg time per 8000 tokens - 2.70 sec