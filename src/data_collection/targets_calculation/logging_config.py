# src/data_collection/targets_calculation/logging_config.py

import logging

logger = logging.getLogger("rnr")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False  
