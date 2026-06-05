import argparse
import sys
from pathlib import Path


project_root = next(
    (
        path
        for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        if (path / "src" / "data_collection" / "consts.py").is_file()
    ),
    None,
)
if project_root is None:
    raise RuntimeError("Could not locate project root containing 'src' directory.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import pandas as pd
import psycopg2
import torch
from psycopg2.extras import Json, execute_batch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from src.data_analysis.data_fetcher.data_fetcher_class import DataFetcher
from src.data_collection.consts import DB_PARAMS
from src.data_collection.logging_config import logger
from src.data_collection.model_driver.model_driver_class import ModelDriver


MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DEFAULT_CONTEXT_MAX_TOKENS = 8000
DEFAULT_BATCH_SIZE = 5
VALID_REPORT_TYPES = ("10-K", "10-Q")


TASK_CONFIGS = {
    "report_type": {
        "prompt": "From this report, the filing cadence is: annual or quarterly? It is",
        "verbolizer": {
            "10-K": ["annual"],
            "10-Q": ["quarter"],
        },
        "true_col": "true_report_type",
        "pred_col": "pred_report_type",
        "confidence_col": "report_type_confidence",
        "probabilities_col": "report_type_probabilities",
    },
    "company_quarter": {
        "prompt": "Read the report excerpt and identify this company's quarter within its reporting year. Selected company quarter:",
        "verbolizer": {
            1: ["first", "january", "february", "march", "spring", "earlyyear"],
            2: ["second", "april", "may", "june", "midyear"],
            3: ["third", "july", "august", "september", "lateyear"],
            4: ["fourth", "october", "november", "december", "yearend", "annual", "yearly"],
        },
        "true_col": "true_company_quarter",
        "pred_col": "pred_company_quarter",
        "confidence_col": "company_quarter_confidence",
        "probabilities_col": "company_quarter_probabilities",
    },
    "sector": {
        "prompt": "I think this company belongs to a sector of:",
        "verbolizer": {
            "Technology": ["technology", "tech", "software", "semiconductor"],
            "Healthcare": ["healthcare", "medical", "pharma", "biotech"],
            "Financial Services": ["financial", "banking", "finance", "fin"],
            "Consumer Cyclical": ["cyclical", "consumer", "retail", "auto"],
            "Consumer Defensive": ["defensive", "staples", "food", "beverage"],
            "Industrials": ["industrials", "industrial", "indust", "manufacturing"],
            "Energy": ["energy", "oil", "gas", "natural"],
            "Utilities": ["utilities", "utility", "util", "electric"],
            "Real Estate": ["real", "estate", "property", "properties", "reit"],
            "Basic Materials": ["materials", "material", "mining", "basic"],
            "Communication Services": ["entertainment", "media", "communications", "commun", "internet"],
        },
        "true_col": "true_sector",
        "pred_col": "pred_sector",
        "confidence_col": "sector_confidence",
        "probabilities_col": "sector_probabilities",
    },
    "market_cap": {
        "prompt": "I think the market size of this company is:",
        "verbolizer": {
            "small": ["small", "micro", "lower"],
            "mid": ["mid", "med", "middle"],
            "large": ["large", "big", "high"],
        },
        "true_col": "true_market_cap",
        "pred_col": "pred_market_cap",
        "confidence_col": "market_cap_confidence",
        "probabilities_col": "market_cap_probabilities",
    },
}


UPSERT_SQL = """
    INSERT INTO metadata_predictions (
        report_id,
        true_report_type,
        pred_report_type,
        report_type_confidence,
        report_type_probabilities,
        true_company_quarter,
        pred_company_quarter,
        company_quarter_confidence,
        company_quarter_probabilities,
        true_sector,
        pred_sector,
        sector_confidence,
        sector_probabilities,
        true_market_cap,
        pred_market_cap,
        market_cap_confidence,
        market_cap_probabilities
    )
    VALUES (
        %(report_id)s,
        %(true_report_type)s,
        %(pred_report_type)s,
        %(report_type_confidence)s,
        %(report_type_probabilities)s,
        %(true_company_quarter)s,
        %(pred_company_quarter)s,
        %(company_quarter_confidence)s,
        %(company_quarter_probabilities)s,
        %(true_sector)s,
        %(pred_sector)s,
        %(sector_confidence)s,
        %(sector_probabilities)s,
        %(true_market_cap)s,
        %(pred_market_cap)s,
        %(market_cap_confidence)s,
        %(market_cap_probabilities)s
    )
    ON CONFLICT (report_id) DO UPDATE SET
        true_report_type = EXCLUDED.true_report_type,
        pred_report_type = EXCLUDED.pred_report_type,
        report_type_confidence = EXCLUDED.report_type_confidence,
        report_type_probabilities = EXCLUDED.report_type_probabilities,
        true_company_quarter = EXCLUDED.true_company_quarter,
        pred_company_quarter = EXCLUDED.pred_company_quarter,
        company_quarter_confidence = EXCLUDED.company_quarter_confidence,
        company_quarter_probabilities = EXCLUDED.company_quarter_probabilities,
        true_sector = EXCLUDED.true_sector,
        pred_sector = EXCLUDED.pred_sector,
        sector_confidence = EXCLUDED.sector_confidence,
        sector_probabilities = EXCLUDED.sector_probabilities,
        true_market_cap = EXCLUDED.true_market_cap,
        pred_market_cap = EXCLUDED.pred_market_cap,
        market_cap_confidence = EXCLUDED.market_cap_confidence,
        market_cap_probabilities = EXCLUDED.market_cap_probabilities
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata prediction pipeline for reports missing metadata predictions."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of reports to save per DB commit.",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=DEFAULT_CONTEXT_MAX_TOKENS,
        help="Maximum total context tokens per prompt call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of pending reports to process.",
    )
    return parser.parse_args()


def build_limit_clause(limit: int | None) -> str:
    if limit is None:
        return ""
    if limit <= 0:
        raise ValueError("limit must be positive when provided.")
    return f"LIMIT {limit}"


def fetch_pending_reports(fetcher: DataFetcher, limit: int | None = None) -> pd.DataFrame:
    limit_clause = build_limit_clause(limit)
    query = f"""
        SELECT
            r.id AS report_id,
            r.f_size
        FROM reports r
        LEFT JOIN metadata_predictions mp
            ON mp.report_id = r.id
        WHERE r.raw_text IS NOT NULL
          AND r.report_type IN %s
          AND mp.report_id IS NULL
        ORDER BY r.filed_date, r.id
        {limit_clause}
    """
    with fetcher.get_db_conn() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params=(VALID_REPORT_TYPES,),
        )


def prepare_pending_reports(
    fetcher: DataFetcher,
    limit: int | None = None,
) -> pd.DataFrame:
    pending_keys_df = fetch_pending_reports(fetcher=fetcher, limit=limit)
    if pending_keys_df.empty:
        return pending_keys_df

    reports_df = fetcher.fetch_reports_with_company_metadata(
        regressors=["raw_text"],
        report_filters={"report_type": list(VALID_REPORT_TYPES)},
    )
    quarter_labels_df = fetcher.derive_company_quarter_labels(
        reports_df,
        report_id_col="id",
        output_col="true_company_quarter",
    ).rename(columns={"id": "report_id"})

    reports_df = reports_df.rename(
        columns={
            "id": "report_id",
            "report_type": "true_report_type",
            "sector": "true_sector",
        }
    )
    pending_df = pending_keys_df.merge(reports_df, on="report_id", how="left")
    pending_df = pending_df.merge(quarter_labels_df, on="report_id", how="left")
    pending_df = fetcher.add_market_cap_bucket(
        pending_df,
        source_col="f_size",
        output_col="true_market_cap",
    )
    pending_df["filed_date"] = pd.to_datetime(pending_df["filed_date"])
    return pending_df


def load_model_driver(model_name: str = MODEL_NAME) -> ModelDriver:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_memory={0: "12GiB", "cpu": "64GiB"},
    )
    return ModelDriver(model_name=model_name, model=model)


def predict_single_task(
    model_driver: ModelDriver,
    raw_text: str,
    prompt: str,
    verbolizer: dict,
    context_max_tokens: int,
) -> tuple[object, float, dict]:
    result = model_driver.predict_metadata(
        verbolizer=verbolizer,
        prompt=prompt,
        text=raw_text,
        context_max_tokens=context_max_tokens,
    )
    probabilities = result["probabilities"]
    predicted_label = max(probabilities, key=probabilities.get)
    return predicted_label, float(result["confidence"]), probabilities


def build_prediction_row(
    row,
    model_driver: ModelDriver,
    context_max_tokens: int,
) -> dict:
    payload = {
        "report_id": int(row.report_id),
        "true_report_type": row.true_report_type,
        "true_company_quarter": (
            None if pd.isna(row.true_company_quarter) else int(row.true_company_quarter)
        ),
        "true_sector": None if pd.isna(row.true_sector) else row.true_sector,
        "true_market_cap": None if pd.isna(row.true_market_cap) else row.true_market_cap,
    }

    for config in TASK_CONFIGS.values():
        predicted_label, confidence, probabilities = predict_single_task(
            model_driver=model_driver,
            raw_text=row.raw_text,
            prompt=config["prompt"],
            verbolizer=config["verbolizer"],
            context_max_tokens=context_max_tokens,
        )
        payload[config["pred_col"]] = predicted_label
        payload[config["confidence_col"]] = confidence
        payload[config["probabilities_col"]] = Json(probabilities)

    return payload


def save_batch(conn: psycopg2.extensions.connection, batch_rows: list[dict]) -> None:
    if not batch_rows:
        return

    with conn.cursor() as cur:
        execute_batch(cur, UPSERT_SQL, batch_rows, page_size=len(batch_rows))
    conn.commit()


def run_pipeline(
    batch_size: int,
    context_max_tokens: int,
    limit: int | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    fetcher = DataFetcher(DB_PARAMS)
    pending_df = prepare_pending_reports(fetcher=fetcher, limit=limit)
    if pending_df.empty:
        logger.info("No pending reports found for metadata prediction.")
        return

    logger.info("Pending reports to process: %s", len(pending_df))

    model_driver = load_model_driver()
    saved_rows = 0
    failed_rows = 0
    batch_rows: list[dict] = []

    with psycopg2.connect(**DB_PARAMS) as conn:
        for row in tqdm(
            pending_df.itertuples(index=False),
            total=len(pending_df),
            desc="Metadata prediction",
            unit="report",
        ):
            try:
                batch_rows.append(
                    build_prediction_row(
                        row=row,
                        model_driver=model_driver,
                        context_max_tokens=context_max_tokens,
                    )
                )
            except Exception:
                failed_rows += 1
                logger.exception("Failed to process report_id=%s", row.report_id)
                continue

            if len(batch_rows) >= batch_size:
                save_batch(conn, batch_rows)
                saved_rows += len(batch_rows)
                logger.info("Saved %s prediction rows.", saved_rows)
                batch_rows.clear()

        if batch_rows:
            save_batch(conn, batch_rows)
            saved_rows += len(batch_rows)
            logger.info("Saved %s prediction rows.", saved_rows)

    logger.info(
        "Metadata pipeline finished. Saved=%s, Failed=%s, Total=%s",
        saved_rows,
        failed_rows,
        len(pending_df),
    )


def main() -> None:
    args = parse_args()
    run_pipeline(
        batch_size=args.batch_size,
        context_max_tokens=args.context_max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
