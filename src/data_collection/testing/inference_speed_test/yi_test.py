import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import time 


from src.data_collection.model_driver.model_driver_class import ModelDriver

verolizer = {
    'positive': ['buy'],
    'negative': ['sell'],
}

model_name = "01-ai/Yi-34B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "12GiB", "cpu": "85GiB"},  # adjust based on actual memory
)

# Apply RoPE scaling patch
model.config.rope_scaling = {"type": "linear", "factor": 2.0}  # for 8192 ctx (factor=8192/4096)

# Ensure model knows it's allowed
model.config.max_position_embeddings = 8192

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

# # avg time per 8000 tokens - 12.07 sec