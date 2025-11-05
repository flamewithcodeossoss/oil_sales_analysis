import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'cleaned_data.csv')
    out_dir = os.path.join(repo_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, 'sales_report.pdf')

    df = pd.read_csv(data_path)

    # Basic cleaning and defensive conversions
    for col in ['value_sales', 'volume_sales', 'average_price', 'size']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without target
    if 'value_sales' not in df.columns:
        raise SystemExit('cleaned_data.csv must contain a value_sales column')
    df = df.dropna(subset=['value_sales'])

    # Date column from year/month if available
    if 'year' in df.columns and 'month' in df.columns:
        try:
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
        except Exception:
            df['date'] = None
    else:
        df['date'] = None

    # Summaries
    total_value = df['value_sales'].sum()
    total_volume = df['volume_sales'].sum() if 'volume_sales' in df.columns else np.nan
    n_records = len(df)

    top_cities = df.groupby('city')['value_sales'].sum().sort_values(ascending=False).head(5)
    top_brands = df.groupby('brand')['value_sales'].sum().sort_values(ascending=False).head(5)

    # Seasonality (by month)
    if 'month' in df.columns:
        monthly = df.groupby('month')['value_sales'].sum().reindex(range(1,13), fill_value=0)
    else:
        monthly = pd.Series()

    # Simple predictive model: predict value_sales using month, size, average_price and top brands
    features = []
    if 'month' in df.columns:
        features.append('month')
    if 'size' in df.columns:
        features.append('size')
    if 'average_price' in df.columns:
        features.append('average_price')

    X = pd.DataFrame()
    if features:
        X = df[features].copy()
    else:
        # fallback: use average_price if available
        X = pd.DataFrame({'average_price': df['average_price']}) if 'average_price' in df.columns else pd.DataFrame({'const': 1}, index=df.index)

    # One-hot top 8 brands to add some categorical signal
    if 'brand' in df.columns:
        top_8_brands = df['brand'].value_counts().head(8).index.tolist()
        for b in top_8_brands:
            X[f'brand_{b}'] = (df['brand'] == b).astype(int)

    # Fill NaNs
    X = X.fillna(0)
    y = df['value_sales']

    # Train/test split and model
    r2 = None
    rmse = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
    except Exception as e:
        r2 = None
        rmse = None

    # Build PDF
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['Title']
    normal = styles['Normal']
    heading = Paragraph('Oil Sales Analysis — Executive Summary', title_style)
    story.append(heading)
    story.append(Spacer(1, 12))

    # Introduction
    intro = (
        "<b>Introduction</b><br/>" 
        "Approach: I prepared a concise summary and built a simple predictive model to estimate monthly value sales. "
        "Data: `data/cleaned_data.csv` (aggregated sales by store/sku/month). "
        "Assumptions: data rows represent monthly store-SKU observations; modelling uses month, size, price and top brands as predictors."
    )
    story.append(Paragraph(intro, normal))
    story.append(Spacer(1, 12))

    # Methods
    methods_text = (
        "<b>Methods</b><br/>"
        "Cleaning steps: numeric coercion for sales/price/size, dropped rows missing `value_sales`, constructed a date from year/month where present. "
        "Analysis: computed aggregates (total sales, top cities/brands, monthly seasonality). "
        "Model selection: a RandomForestRegressor was used as a robust baseline (handles nonlinearity and mixed features). "
        "Evaluation: random 80/20 train-test split; reported R² and RMSE on the holdout set."
    )
    story.append(Paragraph(methods_text, normal))
    story.append(Spacer(1, 12))

    # Findings
    findings_intro = "<b>Findings</b><br/>"
    findings_lines = []
    findings_lines.append(f"Records analysed: {n_records:,}")
    findings_lines.append(f"Total value sales: {total_value:,.2f}")
    if not np.isnan(total_volume):
        findings_lines.append(f"Total volume sold: {total_volume:,.1f}")
    if len(top_cities):
        findings_lines.append("Top cities by value sales:")
    for city, val in top_cities.items():
        findings_lines.append(f" - {city}: {val:,.2f}")
    if len(top_brands):
        findings_lines.append("Top brands by value sales:")
    for brand, val in top_brands.items():
        findings_lines.append(f" - {brand}: {val:,.2f}")

    # Seasonality note
    if not monthly.empty:
        peak_month = int(monthly.idxmax())
        findings_lines.append(f"Observed seasonality with peak aggregate sales in month: {peak_month}")

    # Model metrics
    if r2 is not None:
        findings_lines.append(f"Predictive baseline (RandomForest): R² = {r2:.3f}, RMSE ≈ {rmse:,.2f}")
    else:
        findings_lines.append("Model training/evaluation was not completed due to insufficient/invalid features.")

    findings_text = findings_intro + '<br/>'.join(findings_lines)
    story.append(Paragraph(findings_text, normal))
    story.append(Spacer(1, 12))

    # Simple tables (top cities)
    if len(top_cities):
        story.append(Paragraph('<b>Top 5 Cities — value sales</b>', normal))
        data = [['City', 'Value Sales']]
        for city, val in top_cities.items():
            data.append([city, f"{val:,.2f}"])
        t = Table(data, hAlign='LEFT', colWidths=[200, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Recommendations
    recs = (
        "<b>Recommendations</b><br/>"
        "1) Use the baseline model for short-term forecasting and re-train monthly with new data; include promotional flags and price changes to improve accuracy. "
        "2) Prioritise inventory and promotions in the top cities/brands identified — these drive a large share of value sales. "
        "3) Investigate observed seasonality (peak month) and align stock and marketing spend accordingly. "
        "4) For production: add time-series features (lagged sales), store-level historical trends, and external signals (holidays, oil price) to move from a baseline model to a higher-performing model."
    )
    story.append(Paragraph(recs, normal))
    story.append(Spacer(1, 18))

    # Footer / notes removed per user request

    doc.build(story)
    print(f"Report written to: {out_pdf}")


if __name__ == '__main__':
    main()
