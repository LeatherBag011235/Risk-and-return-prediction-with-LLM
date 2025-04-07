import torch
import psycopg2
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.data_collection.consts import DB_PARAMS
from src.data_collection.model_driver.model_driver_class import ModelDriver
from src.data_collection.logging_config import logger

def min_max_avg(input_list: list[float]) -> tuple[float, float, float]:
    minimun = min(input_list)
    maximum = max(input_list)
    avg = sum(input_list) / len(input_list)

    return minimun, maximum, avg

verolizer = {
    'positive': ['buy'],
    'negative': ['sell'],
}

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "13GiB", "cpu": "64GiB"},  
)

model_driver = ModelDriver(model_name=model_name, model=model, verbolizer=verolizer)

# --- DB setup
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

# --- Fetch all rows that need processing
cur.execute("""
    SELECT id, raw_text
    FROM reports
    WHERE raw_text IS NOT NULL
      AND NOT (
        full_list_default_verbolizer IS NOT NULL AND
        min_default_verbolizer IS NOT NULL AND
        max_default_verbolizer IS NOT NULL AND
        avg_default_verbolizer IS NOT NULL AND
        full_list_shrink_verbolizer IS NOT NULL AND
        min_shrink_verbolizer IS NOT NULL AND
        max_shrink_verbolizer IS NOT NULL AND
        avg_shrink_verbolizer IS NOT NULL
      )
""")
rows = cur.fetchall()


for i, (report_id, raw_text) in enumerate(tqdm(
    rows, 
    total=len(rows), 
    desc="Processing Reports", 
    unit="report", 
    position=1,
    )):
    try:  
        # sample_scores: list[list[float]], avg_inf_duration: float
        sample_scores, avg_inf_duration = model_driver.compute_sample_scores(raw_text)

        default_verbolizer_scores = [row[0] for row in sample_scores]
        shrink_verbolizer_scores = [row[1] for row in sample_scores]

        d_min, d_max, d_avg = min_max_avg(default_verbolizer_scores)
        s_min, s_max, s_avg = min_max_avg(shrink_verbolizer_scores)

        # Update DB
        cur.execute("""
            UPDATE reports
            SET 
                full_list_default_verbolizer = %s,
                min_default_verbolizer = %s,
                max_default_verbolizer = %s,
                avg_default_verbolizer = %s,
                full_list_shrink_verbolizer = %s,
                min_shrink_verbolizer = %s,
                max_shrink_verbolizer = %s,
                avg_shrink_verbolizer = %s
            WHERE id = %s
        """, (
            default_verbolizer_scores,
            d_min, d_max, d_avg,
            shrink_verbolizer_scores,
            s_min, s_max, s_avg,
            report_id,
        ))
        
        if i % 5 == 0:
            conn.commit()

    except Exception as e:
        logger.error(f"Failed to process report_id={report_id}: {e}")
        conn.rollback()

conn.commit()
cur.close()
conn.close()