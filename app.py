from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from chatbot import get_chatbot_response

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
disease_actual_master = pd.read_csv(DATA_DIR / "disease_actual_claims.csv")
sarima_forecast_master = pd.read_csv(DATA_DIR / "sarima_forecast.csv")
conditions = pd.read_csv(DATA_DIR / "conditions.csv")

# -----------------------------
# CLEAN CORE TABLES
# -----------------------------
encounters["PAYER"] = encounters.get("PAYER", "Unknown")
encounters["PAYER"] = encounters["PAYER"].fillna("Unknown")

encounters["ENCOUNTERCLASS"] = encounters.get("ENCOUNTERCLASS", "Unknown")
encounters["ENCOUNTERCLASS"] = encounters["ENCOUNTERCLASS"].fillna("Unknown")

encounters["START_DT"] = pd.to_datetime(
    encounters.get("START"), errors="coerce", utc=True
).dt.tz_localize(None)

encounters["TOTAL_CLAIM_COST"] = pd.to_numeric(
    encounters.get("TOTAL_CLAIM_COST"), errors="coerce"
).fillna(0)

encounters["PAYER_COVERAGE"] = pd.to_numeric(
    encounters.get("PAYER_COVERAGE"), errors="coerce"
).fillna(0)

conditions["DESCRIPTION"] = conditions.get("DESCRIPTION", "Unknown")
conditions["DESCRIPTION"] = conditions["DESCRIPTION"].fillna("Unknown")
conditions["PATIENT"] = conditions.get("PATIENT", pd.Series(dtype="object"))

disease_actual_master["MonthStart"] = pd.to_datetime(
    disease_actual_master["MonthStart"], errors="coerce"
)
disease_actual_master["InsuranceClaimAmount"] = pd.to_numeric(
    disease_actual_master["InsuranceClaimAmount"], errors="coerce"
).fillna(0)

sarima_forecast_master["MonthStart"] = pd.to_datetime(
    sarima_forecast_master["MonthStart"], errors="coerce"
)
sarima_forecast_master["ForecastInsuranceClaimAmount"] = pd.to_numeric(
    sarima_forecast_master["ForecastInsuranceClaimAmount"], errors="coerce"
).fillna(0)

forecast_df["MonthStart"] = pd.to_datetime(
    forecast_df["MonthStart"], errors="coerce"
)
forecast_df["ForecastClaimCost"] = pd.to_numeric(
    forecast_df["ForecastClaimCost"], errors="coerce"
).fillna(0)

# -----------------------------
# FILTER HELPERS
# -----------------------------
def parse_start_date(value):
    if not value:
        return None
    dt = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(dt) else dt


def parse_end_date(value):
    if not value:
        return None
    dt = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(dt) else dt


def apply_date_filter(df, col_name, start_date=None, end_date=None):
    result = df.copy()
    if start_date is not None:
        result = result[result[col_name] >= start_date]
    if end_date is not None:
        result = result[result[col_name] <= end_date]
    return result


def short_label(text, keep=12):
    text = str(text)
    return text if len(text) <= keep else f"{text[:keep]}..."


# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
insurance_payer_values = sorted(
    encounters["PAYER"].dropna().astype(str).unique().tolist()
)
insurance_payer_options = [
    {"value": p, "label": short_label(p, keep=12)} for p in insurance_payer_values
]

insurance_claim_types = sorted(
    encounters["ENCOUNTERCLASS"].dropna().astype(str).unique().tolist()
)

insurance_diseases = sorted(
    conditions["DESCRIPTION"].dropna().astype(str).unique().tolist()
)

pharma_diseases = sorted(
    disease_actual_master["DISEASE"].dropna().astype(str).unique().tolist()
)

# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return render_template(
        "home.html",
        show_filters=False,
        page_type="home"
    )

# -----------------------------
# INSURANCE DASHBOARD
# -----------------------------
@app.route("/insurance-dashboard")
def insurance_dashboard():
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")
    selected_payers = request.args.getlist("payer")
    selected_claim_types = request.args.getlist("claim_type")
    selected_disease = request.args.get("disease", "All")

    start_date = parse_start_date(start_date_str)
    end_date = parse_end_date(end_date_str)

    filtered_encounters = encounters.copy()

    if selected_payers:
        filtered_encounters = filtered_encounters[
            filtered_encounters["PAYER"].isin(selected_payers)
        ]

    if selected_claim_types:
        filtered_encounters = filtered_encounters[
            filtered_encounters["ENCOUNTERCLASS"].isin(selected_claim_types)
        ]

    if selected_disease and selected_disease != "All":
        patient_ids = (
            conditions[conditions["DESCRIPTION"] == selected_disease]["PATIENT"]
            .dropna()
            .unique()
        )
        filtered_encounters = filtered_encounters[
            filtered_encounters["PATIENT"].isin(patient_ids)
        ]

    filtered_encounters = apply_date_filter(
        filtered_encounters,
        "START_DT",
        start_date=start_date,
        end_date=end_date
    )

    # KPI values
    total_claims = f"{len(filtered_encounters):,}"
    total_actual = f"{filtered_encounters['TOTAL_CLAIM_COST'].sum():,.2f}"

    base_forecast_df = apply_date_filter(
        forecast_df.copy(),
        "MonthStart",
        start_date=start_date,
        end_date=end_date
    )

    if selected_disease and selected_disease != "All":
        insurance_disease_forecast = sarima_forecast_master.copy()
        insurance_disease_forecast = insurance_disease_forecast[
            insurance_disease_forecast["DISEASE"] == selected_disease
        ]
        insurance_disease_forecast = apply_date_filter(
            insurance_disease_forecast,
            "MonthStart",
            start_date=start_date,
            end_date=end_date
        )

        total_forecast = f"{insurance_disease_forecast['ForecastInsuranceClaimAmount'].sum():,.2f}"
        forecast_plot_df = insurance_disease_forecast.rename(
            columns={
                "MonthStart": "Month",
                "ForecastInsuranceClaimAmount": "Forecast"
            }
        )[["Month", "Forecast"]].sort_values("Month")
    else:
        total_forecast = f"{base_forecast_df['ForecastClaimCost'].sum():,.2f}"
        forecast_plot_df = base_forecast_df.rename(
            columns={
                "MonthStart": "Month",
                "ForecastClaimCost": "Forecast"
            }
        )[["Month", "Forecast"]].sort_values("Month")

    # Actual monthly trend
    actual_plot_df = (
        filtered_encounters.dropna(subset=["START_DT"])
        .assign(MonthPeriod=filtered_encounters["START_DT"].dt.to_period("M"))
        .groupby("MonthPeriod")["TOTAL_CLAIM_COST"]
        .sum()
        .reset_index()
    )

    if not actual_plot_df.empty:
        actual_plot_df["Month"] = pd.to_datetime(actual_plot_df["MonthPeriod"].astype(str))
        actual_plot_df.rename(columns={"TOTAL_CLAIM_COST": "Actual"}, inplace=True)
        actual_plot_df = actual_plot_df[["Month", "Actual"]].sort_values("Month")
    else:
        actual_plot_df = pd.DataFrame(columns=["Month", "Actual"])

    trend_fig = go.Figure()

    if not actual_plot_df.empty:
        trend_fig.add_trace(go.Scatter(
            x=actual_plot_df["Month"],
            y=actual_plot_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(width=2.5, color="#4F6BED"),
            marker=dict(size=4, color="#4F6BED")
        ))

    if not forecast_plot_df.empty:
        trend_fig.add_trace(go.Scatter(
            x=forecast_plot_df["Month"],
            y=forecast_plot_df["Forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3, dash="dash", color="#E4572E"),
            marker=dict(size=5, color="#E4572E")
        ))

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
            ay=-35
        )

    trend_fig.update_layout(
        title="Monthly Claim Cost - Actual vs Forecast",
        xaxis_title="Month",
        yaxis_title="Claim Cost",
        template="plotly_white",
        height=280,
        hovermode="x unified",
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    trend_chart = trend_fig.to_html(full_html=False)

    # Claim type chart
    claim_type_df = (
        filtered_encounters.groupby("ENCOUNTERCLASS")
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
        height=240,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    claim_type_chart = claim_type_fig.to_html(full_html=False)

    # Payer summary and leakage
    payer_summary = (
        filtered_encounters.groupby("PAYER")
        .agg(
            Total_Claims=("PAYER", "count"),
            Total_Claim_Cost=("TOTAL_CLAIM_COST", "sum"),
            Coverage_Amount=("PAYER_COVERAGE", "sum")
        )
        .reset_index()
    )

    payer_summary["Denied Amount"] = (
        payer_summary["Total_Claim_Cost"] - payer_summary["Coverage_Amount"]
    )
    payer_summary["Coverage %"] = (
        payer_summary["Coverage_Amount"]
        / payer_summary["Total_Claim_Cost"].replace(0, pd.NA)
        * 100
    ).round(2).fillna(0)

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
        height=240,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(categoryorder="total ascending")
    )
    leakage_chart = leakage_fig.to_html(full_html=False)

    payer_table_df = payer_summary.rename(columns={
        "Total_Claim_Cost": "Total Claim Cost",
        "Coverage_Amount": "Coverage Amount"
    }).sort_values("Denied Amount", ascending=False).head(10).round(2)

    payer_table = payer_table_df.to_html(
        classes="table table-hover table-sm align-middle",
        index=False,
        border=0
    )

    return render_template(
        "insurance_dashboard.html",
        total_claims=total_claims,
        total_actual=total_actual,
        total_forecast=total_forecast,
        trend_chart=trend_chart,
        claim_type_chart=claim_type_chart,
        leakage_chart=leakage_chart,
        payer_table=payer_table,
        show_filters=True,
        page_type="insurance",
        insurance_payer_options=insurance_payer_options,
        insurance_claim_types=insurance_claim_types,
        insurance_diseases=insurance_diseases,
        selected_payers=selected_payers,
        selected_claim_types=selected_claim_types,
        selected_disease=selected_disease,
        start_date=start_date_str,
        end_date=end_date_str
    )

# -----------------------------
# PHARMA DASHBOARD
# -----------------------------
@app.route("/pharma-dashboard")
def pharma_dashboard():
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")
    selected_disease = request.args.get("disease", "All")
    forecast_view = request.args.get("forecast_view", "both")

    start_date = parse_start_date(start_date_str)
    end_date = parse_end_date(end_date_str)

    actual_df2 = disease_actual_master.copy()
    forecast_df2 = sarima_forecast_master.copy()

    actual_df2 = apply_date_filter(
        actual_df2, "MonthStart", start_date=start_date, end_date=end_date
    )
    forecast_df2 = apply_date_filter(
        forecast_df2, "MonthStart", start_date=start_date, end_date=end_date
    )

    if selected_disease and selected_disease != "All":
        actual_df2 = actual_df2[actual_df2["DISEASE"] == selected_disease]
        forecast_df2 = forecast_df2[forecast_df2["DISEASE"] == selected_disease]

    total_diseases = f"{len(actual_df2):,}"
    total_actual_claims = f"{actual_df2['InsuranceClaimAmount'].sum():,.2f}"
    total_forecast_claims = f"{forecast_df2['ForecastInsuranceClaimAmount'].sum():,.2f}"

    top_diseases = (
        actual_df2.groupby("DISEASE")["InsuranceClaimAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    # Top disease burden
    burden_df = (
        actual_df2.groupby("DISEASE")["InsuranceClaimAmount"]
        .sum()
        .reset_index()
        .sort_values("InsuranceClaimAmount", ascending=False)
        .head(10)
    )

    disease_burden_chart_fig = px.bar(
        burden_df,
        x="InsuranceClaimAmount",
        y="DISEASE",
        orientation="h",
        title="Top Disease Burden by Historical Claims"
    )
    disease_burden_chart_fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(categoryorder="total ascending")
    )
    disease_burden_chart = disease_burden_chart_fig.to_html(full_html=False)

    # Forecast summary
    forecast_summary = (
        forecast_df2.groupby("DISEASE")["ForecastInsuranceClaimAmount"]
        .sum()
        .reset_index()
        .rename(columns={"ForecastInsuranceClaimAmount": "Forecast"})
        .sort_values("Forecast", ascending=False)
        .head(8)
        .round(2)
    )

    forecast_summary["Forecast"] = forecast_summary["Forecast"].apply(
        lambda x: f"${x:,.0f}"
    )

    forecast_summary_table = forecast_summary.to_html(
        classes="table table-hover table-sm align-middle",
        index=False,
        border=0
    )

    # Opportunity matrix
    filtered_encounters = apply_date_filter(
        encounters.copy(),
        "START_DT",
        start_date=start_date,
        end_date=end_date
    )

    if selected_disease and selected_disease != "All":
        patient_ids = (
            conditions[conditions["DESCRIPTION"] == selected_disease]["PATIENT"]
            .dropna()
            .unique()
        )
        filtered_encounters = filtered_encounters[
            filtered_encounters["PATIENT"].isin(patient_ids)
        ]

    payer_summary = (
        filtered_encounters.groupby("PAYER")
        .agg(
            Total_Claim_Exposure=("TOTAL_CLAIM_COST", "sum"),
            Coverage_Amount=("PAYER_COVERAGE", "sum"),
            Total_Claims=("PAYER", "count")
        )
        .reset_index()
    )

    payer_summary["Revenue Leakage"] = (
        payer_summary["Total_Claim_Exposure"] - payer_summary["Coverage_Amount"]
    )
    payer_summary["Market Pressure"] = payer_summary["Total_Claims"]
    payer_summary = payer_summary.sort_values("Total_Claim_Exposure", ascending=False).head(10)

    opportunity_fig = px.scatter(
        payer_summary,
        x="Total_Claim_Exposure",
        y="Revenue Leakage",
        size="Market Pressure",
        color="PAYER",
        hover_name="PAYER",
        title="Opportunity Matrix: Revenue Leakage vs Market Exposure",
        size_max=55
    )
    opportunity_fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    opportunity_matrix_chart = opportunity_fig.to_html(full_html=False)

    # SARIMA chart
    if selected_disease == "All":
        sarima_target = top_diseases[0] if top_diseases else None
    else:
        sarima_target = selected_disease

    sarima_actual = actual_df2[
        actual_df2["DISEASE"] == sarima_target
    ].sort_values("MonthStart")

    sarima_future = forecast_df2[
        forecast_df2["DISEASE"] == sarima_target
    ].sort_values("MonthStart")

    sarima_fig = go.Figure()

    if forecast_view in ["both", "actual"]:
        sarima_fig.add_trace(go.Scatter(
            x=sarima_actual["MonthStart"],
            y=sarima_actual["InsuranceClaimAmount"],
            mode="lines",
            name="Historical Demand",
            line=dict(width=2.5, color="#0F2D5C")
        ))

    if forecast_view in ["both", "forecast"]:
        sarima_fig.add_trace(go.Scatter(
            x=sarima_future["MonthStart"],
            y=sarima_future["ForecastInsuranceClaimAmount"],
            mode="lines",
            name="SARIMA Forecast",
            line=dict(width=2.5, color="#25C2A0")
        ))

    sarima_fig.update_layout(
        title=f"SARIMA Monthly Demand Forecast: {sarima_target}" if sarima_target else "SARIMA Monthly Demand Forecast",
        xaxis_title="Month",
        yaxis_title="Value",
        template="plotly_white",
        height=260,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    sarima_chart = sarima_fig.to_html(full_html=False)

    return render_template(
        "pharma_dashboard.html",
        total_diseases=total_diseases,
        total_actual_claims=total_actual_claims,
        total_forecast_claims=total_forecast_claims,
        disease_burden_chart=disease_burden_chart,
        forecast_summary_table=forecast_summary_table,
        disease_list=pharma_diseases,
        selected_disease=selected_disease,
        forecast_view=forecast_view,
        opportunity_matrix_chart=opportunity_matrix_chart,
        sarima_chart=sarima_chart,
        show_filters=True,
        page_type="pharma",
        start_date=start_date_str,
        end_date=end_date_str
    )

# -----------------------------
# CHATBOT
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a question."})

    try:
        response = get_chatbot_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)