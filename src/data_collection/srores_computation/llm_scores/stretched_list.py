import psycopg2
from tqdm import tqdm
from src.data_collection.consts import DB_PARAMS
from src.data_collection.logging_config import logger


def stretch_transform(input_list: list[float], max_len: int) -> list[float]:
    cur_len = len(input_list)
    stretched_list = []

    for i, value in enumerate(input_list):
        left_ind = round((max_len / cur_len) * (i))
        right_ind = round((max_len / cur_len) * (i+1)) - 1

        chunk_len = right_ind - left_ind + 1        
        chunk = [value] * chunk_len

        stretched_list.extend(chunk)

    return stretched_list

# --- Connect
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

cur.execute("""
    SELECT
        MAX(GREATEST(
            COALESCE(array_length(full_list_default_verbolizer, 1), 0),
            COALESCE(array_length(full_list_shrink_verbolizer, 1), 0)
        ))
    FROM reports
    WHERE full_list_default_verbolizer IS NOT NULL
      AND full_list_shrink_verbolizer IS NOT NULL;
""")

(max_len,) = cur.fetchone()
print(f"Maximum list length: {max_len}")

cur.execute("""
    SELECT id, full_list_default_verbolizer, full_list_shrink_verbolizer
    FROM reports
    WHERE full_list_default_verbolizer IS NOT NULL
      AND full_list_shrink_verbolizer IS NOT NULL
""")
rows = cur.fetchall()

for i, (report_id, default_list, shrink_list) in enumerate(tqdm(rows, desc="Transforming & Storing Stretched Lists", unit="report")):
    try:

        stretch_default = stretch_transform(default_list, max_len)
        stretch_shrink = stretch_transform(shrink_list, max_len)

        cur.execute("""
            UPDATE reports
            SET
                stretch_default = %s,
                stretch_shrink = %s
            WHERE id = %s
        """, (stretch_default, stretch_shrink, report_id))

        if i % 100 == 0:
            conn.commit()

    except Exception as e:
        logger.error(f"Failed to process report_id={report_id}: {e}")
        conn.rollback()

conn.commit()
cur.close()
conn.close()
