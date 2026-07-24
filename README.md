# Return Prediction with LLMs and Financial Texts

Replication code and supplementary materials for:

**Maxim Shibanov, "Return Prediction with LLMs and Financial Texts"**

This project studies whether sentiment extracted from firms' 10-K and 10-Q filings with large language models helps explain subsequent stock returns. The empirical analysis compares LLM-based sentiment scores with dictionary-based sentiment measures and evaluates their predictive content for raw returns, excess returns, CAPM abnormal returns, and Fama-French five-factor abnormal returns.

## Repository Contents

- `src/data_collection/`: code for filing collection, parsing, target construction, and sentiment-score computation.
- `src/data_analysis/`: descriptive analysis, validation checks, regressions, plots, and portfolio-performance calculations.
- `src/data_analysis/linear_modeling/`: main regression analyses, bootstrap checks, and generated coefficient plots.
- `src/data_analysis/sharpe_ratio/`: portfolio-performance and Sharpe-ratio calculations, including selected generated outputs.
- `rnr_env.yml`: main conda environment for data analysis and plotting.
- `vllm_env.yml`: optional GPU-oriented environment for LLM inference workloads.
- `CITATION.cff`: citation metadata for GitHub and Zenodo.
- `.zenodo.json`: metadata used by Zenodo when archiving a GitHub release.

## Data Availability

This repository contains code and selected generated outputs. Raw SEC filings, local database dumps, model weights, and some external market-data inputs are not included because they are large, generated locally, or obtained from third-party sources.

The main external inputs are:

- SEC EDGAR 10-K and 10-Q filings.
- Equity price data obtained through Yahoo Finance / `yfinance` and related data-fetching code.
- Fama-French factor data.
- Dictionary resources used for textual sentiment baselines.
- Local PostgreSQL tables used by the data collection and analysis scripts.

## Installation

Create the main analysis environment:

```bash
conda env create -f rnr_env.yml
conda activate rnr_env
pip install -e .
```

For GPU-based LLM inference workloads, create the optional inference environment:

```bash
conda env create -f vllm_env.yml
conda activate vllm_env
```

## Configuration

Database and API credentials should be supplied through environment variables. The main code reads configuration from `src/data_collection/consts.py`.

Expected variables include:

```bash
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_HOST=
API_KEY=
SECRET_KEY=
THESAURUS_API_KEY=
```

Do not commit private credentials, API keys, raw database dumps, or model-access tokens to the repository.

## Reproducing the Results

The analysis was originally run through a combination of Python scripts and Jupyter notebooks. The main result-producing materials are located in:

- `src/data_collection/text_collection/`: filing download, parsing, and conversion utilities.
- `src/data_collection/srores_computation/dictionary_sentiments/`: dictionary-based sentiment scoring.
- `src/data_collection/srores_computation/llm_scores/`: LLM-based sentiment scoring.
- `src/data_collection/targets_calculation/`: return-target construction.
- `src/data_analysis/dict_scores_aggregation/`: descriptive analysis for dictionary scores.
- `src/data_analysis/llm_scores_aggregation/`: descriptive analysis for LLM sentiment scores.
- `src/data_analysis/linear_modeling/`: fixed-effects regressions, bootstrap checks, adaptive-lasso checks, and result plots.
- `src/data_analysis/sharpe_ratio/`: portfolio construction and performance summaries.
- `src/data_analysis/reprod_check/`: reproducibility and data-consistency checks.

Selected generated figures and CSV outputs are included under the corresponding analysis directories. The exact release archived on Zenodo should be cited together with the GitHub release tag and commit hash.

## Citation

If you use this repository, please cite the archived software release:

```bibtex
@software{shibanov_return_llm_2026,
  author = {Shibanov, Maxim},
  title = {Return Prediction with LLMs and Financial Texts: Replication Code and Supplementary Materials},
  year = {2026},
  publisher = {Zenodo},
  url = {https://github.com/LeatherBag011235/Risk-and-return-prediction-with-LLM}
}
```

The current release-preparation version is `1.0.0-ssrn`.

## License

This repository is released under the MIT License. See `LICENSE`.
