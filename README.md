# Oil Sales Analysis

A compact analytics project for retail packaged-oil sales. The repository contains data cleaning and EDA notebooks, baseline predictive modeling, a Streamlit dashboard for interactive exploration, and a small utility to generate an executive PDF report.

## Short description

Oil sales analysis: data cleaning, exploratory analysis, and baseline predictive modeling for a retail chain. Includes a Streamlit dashboard and an automated PDF report generator.

## Contents

- `app.py` — Streamlit dashboard for interactive exploration (launch with `streamlit run app.py`).
- `data/cleaned_data.csv` — cleaned dataset snapshot used by notebooks and scripts.
- `notebooks/` — Jupyter notebooks for reproducibility:
	- `01_data_cleaning.ipynb` — data cleaning and preprocessing.
	- `02_eda.ipynb` — exploratory data analysis and visualizations.
	- `03_modeling.ipynb` — model experiments and evaluation.
- `scripts/` — helper scripts:
	- `data_cleaning.py` — programmatic cleaning utilities.
	- `eda.py` — scripts that reproduce EDA charts.
	- `modeling.py` — model training and evaluation entrypoints.
	- `generate_report.py` — generates `outputs/sales_report.pdf` (concise executive summary).
- `outputs/` — generated artifacts (PDF reports, charts, model artifacts).
- `requirements.txt` — pinned Python dependencies used for the project.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

3. Run the Streamlit dashboard:

```powershell
python -m streamlit run app.py
# or
streamlit run app.py
```

4. Regenerate the executive PDF report:

```powershell
python scripts\generate_report.py
# output: outputs\sales_report.pdf
```

## Methods & modelling notes

- Cleaning: numeric coercion for sales/price/size, drop rows missing `value_sales`, construct a `date` column from `year` and `month` where available.
- Analysis: aggregate metrics (total value/volume, top cities/brands), monthly seasonality analysis.
- Baseline model: a RandomForestRegressor baseline is provided as a robust starting point. Use a time-based train/validation split for production forecasting and add lag features, promotions, and external signals to improve performance.

## Recommendations for production use

1. Move from random holdout to time-based validation (walk-forward) to avoid lookahead bias.
2. Add lag features (previous-month sales), rolling statistics, and store-level history.
3. Include binary flags for promotions, holidays, and price changes; incorporate external signals where available.
4. Automate model retraining (e.g., monthly) and expose forecasts through the dashboard or an API.

## Extending the report

The PDF generator (`scripts/generate_report.py`) uses ReportLab and currently produces a concise executive summary. It can be extended to embed PNG charts (monthly seasonality, top-city bar charts) or to output multiple report formats (HTML, PPTX).

## License & contact

Add your preferred license file (e.g., `LICENSE`) and a contact email or maintainer details here.

---

If you want, I can:
- Commit this `README.md` (already written to disk) and push further edits.
- Shorten or expand any section, or add sample screenshots of the Streamlit app.
- Add a `CONTRIBUTING.md` or `LICENSE` file.
pandas
numpy
matplotlib
seaborn
scikit-learn