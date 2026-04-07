from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# LOAD FILES
# -----------------------------
patients = pd.read_csv(DATA_DIR / "patients.csv")
encounters = pd.read_csv(DATA_DIR / "encounters.csv")
monthly_claim_summary = pd.read_csv(DATA_DIR / "monthly_claim_summary.csv")
forecast_df = pd.read_csv(DATA_DIR / "claim_cost_forecast.csv")

# -----------------------------
# CLEAN ENCOUNTERS
# -----------------------------
if "ENCOUNTERCLASS" not in encounters.columns:
    encounters["ENCOUNTERCLASS"] = "Unknown"
else:
    encounters["ENCOUNTERCLASS"] = encounters["ENCOUNTERCLASS"].fillna("Unknown")

if "PAYER" not in encounters.columns:
    encounters["PAYER"] = "Unknown"
else:
    encounters["PAYER"] = encounters["PAYER"].fillna("Unknown")

# -----------------------------
# HELPER
# -----------------------------
def find_column(df, possible_names):
    lower_map = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

# -----------------------------
# DETECT REAL COLUMNS FOR INSURANCE
# -----------------------------
actual_month_col = find_column(
    monthly_claim_summary,
    ["Month", "MonthStart", "month", "monthstart"]
)

actual_value_col = find_column(
    monthly_claim_summary,
    [
        "Sum of TotalClaimCost",
        "TotalClaimCost",
        "ClaimCost",
        "InsuranceClaimAmount",
        "claim_cost"
    ]
)

forecast_month_col = find_column(
    forecast_df,
    ["Month", "MonthStart", "month", "monthstart"]
)

forecast_value_col = find_column(
    forecast_df,
    [
        "ForecastClaimCost",
        "Forecast",
        "Predicted",
        "ForecastInsuranceClaimAmount"
    ]
)

# -----------------------------
# PREPARE INSURANCE DATA
# -----------------------------
if actual_month_col and actual_value_col:
    actual_df = monthly_claim_summary[[actual_month_col, actual_value_col]].copy()
    actual_df.columns = ["Month", "Actual"]
else:
    actual_df = pd.DataFrame(columns=["Month", "Actual"])

if forecast_month_col and forecast_value_col:
    future_df = forecast_df[[forecast_month_col, forecast_value_col]].copy()
    future_df.columns = ["Month", "Forecast"]
else:
    future_df = pd.DataFrame(columns=["Month", "Forecast"])

actual_df["Month"] = pd.to_datetime(actual_df["Month"], errors="coerce")
future_df["Month"] = pd.to_datetime(future_df["Month"], errors="coerce")

actual_df["Actual"] = pd.to_numeric(actual_df["Actual"], errors="coerce")
future_df["Forecast"] = pd.to_numeric(future_df["Forecast"], errors="coerce")

actual_df = actual_df.sort_values("Month").dropna(subset=["Month"])
future_df = future_df.sort_values("Month").dropna(subset=["Month"])

total_actual_value = actual_df["Actual"].fillna(0).sum()
total_forecast_value = future_df["Forecast"].fillna(0).sum()

# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -----------------------------
# INSURANCE DASHBOARD
# -----------------------------
@app.route("/insurance-dashboard")
def insurance_dashboard():
    total_claims = f"{len(encounters):,}"
    total_actual = f"{total_actual_value:,.2f}"
    total_forecast = f"{total_forecast_value:,.2f}"

    actual_plot_df = actual_df.copy()
    forecast_plot_df = future_df.copy()
    actual_plot_df = actual_plot_df[actual_plot_df["Month"] >= pd.Timestamp("2023-01-01")]

    trend_fig = go.Figure()

    trend_fig.add_trace(go.Scatter(
        x=actual_plot_df["Month"],
        y=actual_plot_df["Actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(width=3, color="#4F6BED"),
        marker=dict(size=5, color="#4F6BED")
    ))

    trend_fig.add_trace(go.Scatter(
        x=forecast_plot_df["Month"],
        y=forecast_plot_df["Forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(width=4, dash="dash", color="#E4572E"),
        marker=dict(size=8, color="#E4572E")
    ))

    if not forecast_plot_df.empty:
        forecast_start = forecast_plot_df["Month"].min()
        ymax_candidates = []
        if not actual_plot_df.empty:
            ymax_candidates.append(actual_plot_df["Actual"].max())
        if not forecast_plot_df.empty:
            ymax_candidates.append(forecast_plot_df["Forecast"].max())
        ymax = max(ymax_candidates) if ymax_candidates else 0

        trend_fig.add_vline(
            x=forecast_start,
            line_width=2,
            line_dash="dot",
            line_color="gray"
        )
        trend_fig.add_annotation(
            x=forecast_start,
            y=ymax,
            text="Forecast Starts",
            showarrow=True,
            arrowhead=1,
            ay=-40
        )

    trend_fig.update_layout(
        title="Monthly Claim Cost - Actual vs Forecast",
        xaxis_title="Month",
        yaxis_title="Claim Cost",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    trend_chart = trend_fig.to_html(full_html=False)

    claim_type_df = (
        encounters.groupby("ENCOUNTERCLASS")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    claim_type_fig = px.pie(
        claim_type_df,
        names="ENCOUNTERCLASS",
        values="Count",
        title="Claim Type Distribution"
    )
    claim_type_fig.update_layout(
        template="plotly_white",
        height=450
    )
    claim_type_chart = claim_type_fig.to_html(full_html=False)

    payer_summary = (
        encounters.groupby("PAYER")
        .agg(Total_Claims=("PAYER", "count"))
        .reset_index()
    )

    payer_summary["Total Claim Cost"] = payer_summary["Total_Claims"] * 100
    payer_summary["Coverage Amount"] = payer_summary["Total Claim Cost"] * 0.75
    payer_summary["Denied Amount"] = payer_summary["Total Claim Cost"] - payer_summary["Coverage Amount"]
    payer_summary["Coverage %"] = (
        payer_summary["Coverage Amount"] / payer_summary["Total Claim Cost"] * 100
    ).round(2)

    leakage_df = payer_summary.sort_values("Denied Amount", ascending=False).head(10)

    leakage_fig = px.bar(
        leakage_df,
        x="Denied Amount",
        y="PAYER",
        orientation="h",
        title="Top Payers by Revenue Leakage"
    )
    leakage_fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(categoryorder="total ascending")
    )
    leakage_chart = leakage_fig.to_html(full_html=False)

    payer_table = payer_summary.round(2).to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    return render_template(
        "insurance_dashboard.html",
        total_claims=total_claims,
        total_actual=total_actual,
        total_forecast=total_forecast,
        trend_chart=trend_chart,
        claim_type_chart=claim_type_chart,
        leakage_chart=leakage_chart,
        payer_table=payer_table
    )

# -----------------------------
# PHARMA DASHBOARD
# -----------------------------
@app.route("/pharma-dashboard")
def pharma_dashboard():
    disease_actual = pd.read_csv(DATA_DIR / "disease_actual_claims.csv")
    sarima_forecast = pd.read_csv(DATA_DIR / "sarima_forecast.csv")

    def pick_col(df, options):
        lower_map = {c.lower(): c for c in df.columns}
        for opt in options:
            if opt.lower() in lower_map:
                return lower_map[opt.lower()]
        return None

    actual_disease_col = pick_col(disease_actual, ["DISEASE", "Disease"])
    actual_month_col2 = pick_col(disease_actual, ["MonthStart", "Month", "monthstart", "month"])
    actual_claim_col2 = pick_col(disease_actual, ["InsuranceClaimAmount", "ClaimAmount", "ActualClaimAmount"])

    forecast_disease_col = pick_col(sarima_forecast, ["DISEASE", "Disease"])
    forecast_month_col2 = pick_col(sarima_forecast, ["MonthStart", "Month", "monthstart", "month"])
    forecast_claim_col2 = pick_col(sarima_forecast, ["ForecastInsuranceClaimAmount", "ForecastClaimAmount", "Forecast"])

    actual_df2 = disease_actual[[actual_disease_col, actual_month_col2, actual_claim_col2]].copy()
    actual_df2.columns = ["DISEASE", "Month", "Actual"]
    actual_df2["Month"] = pd.to_datetime(actual_df2["Month"], errors="coerce")
    actual_df2["Actual"] = pd.to_numeric(actual_df2["Actual"], errors="coerce")
    actual_df2 = actual_df2.dropna(subset=["Month"])

    forecast_df2 = sarima_forecast[[forecast_disease_col, forecast_month_col2, forecast_claim_col2]].copy()
    forecast_df2.columns = ["DISEASE", "Month", "Forecast"]
    forecast_df2["Month"] = pd.to_datetime(forecast_df2["Month"], errors="coerce")
    forecast_df2["Forecast"] = pd.to_numeric(forecast_df2["Forecast"], errors="coerce")
    forecast_df2 = forecast_df2.dropna(subset=["Month"])

    selected_disease = request.args.get("disease", "All")
    disease_list = sorted(actual_df2["DISEASE"].dropna().unique())

    total_diseases = f"{len(actual_df2):,}"
    total_actual_claims = f"{actual_df2['Actual'].fillna(0).sum():,.2f}"
    total_forecast_claims = f"{forecast_df2['Forecast'].fillna(0).sum():,.2f}"

    top_diseases = (
        actual_df2.groupby("DISEASE")["Actual"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    # Top disease burden chart
    burden_df = (
        actual_df2.groupby("DISEASE")["Actual"]
        .sum()
        .reset_index()
        .sort_values("Actual", ascending=False)
        .head(10)
    )

    disease_burden_chart_fig = px.bar(
        burden_df,
        x="Actual",
        y="DISEASE",
        orientation="h",
        title="Top Disease Burden by Historical Claims"
    )
    disease_burden_chart_fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(categoryorder="total ascending")
    )
    disease_burden_chart = disease_burden_chart_fig.to_html(full_html=False)

    # Forecast summary table
    forecast_summary = (
        forecast_df2.groupby("DISEASE")["Forecast"]
        .sum()
        .reset_index()
        .sort_values("Forecast", ascending=False)
        .round(2)
    )
    forecast_summary_table = forecast_summary.to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    # Opportunity Matrix
    payer_summary = (
        encounters.groupby("PAYER")
        .agg(Total_Claims=("PAYER", "count"))
        .reset_index()
    )

    payer_summary["Total Claim Exposure"] = payer_summary["Total_Claims"] * 4000
    payer_summary["Revenue Leakage"] = payer_summary["Total_Claims"] * 1000
    payer_summary["Market Pressure"] = payer_summary["Total_Claims"] * 0.8

    payer_summary = payer_summary.sort_values("Total Claim Exposure", ascending=False).head(10)

    opportunity_fig = px.scatter(
        payer_summary,
        x="Total Claim Exposure",
        y="Revenue Leakage",
        size="Market Pressure",
        color="PAYER",
        hover_name="PAYER",
        title="Opportunity Matrix: Revenue Leakage vs Market Exposure",
        size_max=60
    )
    opportunity_fig.update_layout(
        template="plotly_white",
        height=500
    )
    opportunity_matrix_chart = opportunity_fig.to_html(full_html=False)

    # SARIMA chart only
    if selected_disease == "All":
        sarima_target = top_diseases[0] if top_diseases else None
    else:
        sarima_target = selected_disease

    sarima_actual = actual_df2[actual_df2["DISEASE"] == sarima_target].sort_values("Month")
    sarima_future = forecast_df2[forecast_df2["DISEASE"] == sarima_target].sort_values("Month")

    sarima_fig = go.Figure()

    sarima_fig.add_trace(go.Scatter(
        x=sarima_actual["Month"],
        y=sarima_actual["Actual"],
        mode="lines",
        name="Historical Demand",
        line=dict(width=3, color="#0F2D5C")
    ))

    sarima_fig.add_trace(go.Scatter(
        x=sarima_future["Month"],
        y=sarima_future["Forecast"],
        mode="lines",
        name="SARIMA Forecast",
        line=dict(width=3, color="#25C2A0")
    ))

    sarima_fig.update_layout(
        title=f"SARIMA Monthly Demand Forecast: {sarima_target}" if sarima_target else "SARIMA Monthly Demand Forecast",
        xaxis_title="Month",
        yaxis_title="Value",
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )
    sarima_chart = sarima_fig.to_html(full_html=False)

    return render_template(
        "pharma_dashboard.html",
        total_diseases=total_diseases,
        total_actual_claims=total_actual_claims,
        total_forecast_claims=total_forecast_claims,
        disease_burden_chart=disease_burden_chart,
        forecast_summary_table=forecast_summary_table,
        disease_list=disease_list,
        selected_disease=selected_disease,
        opportunity_matrix_chart=opportunity_matrix_chart,
        sarima_chart=sarima_chart
    )

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)