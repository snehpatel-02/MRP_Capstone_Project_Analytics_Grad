from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# Load local Synthea CSV data
patients = pd.read_csv("data/patients.csv")
encounters = pd.read_csv("data/encounters.csv")
conditions = pd.read_csv("data/conditions.csv")
procedures = pd.read_csv("data/procedures.csv")
medications = pd.read_csv("data/medications.csv")
observations = pd.read_csv("data/observations.csv")
allergies = pd.read_csv("data/allergies.csv")

# Load predictive analytics CSVs
monthly_claim = pd.read_csv("data/monthly_claim_summary.csv")
claim_forecast = pd.read_csv("data/claim_cost_forecast.csv")


@app.route("/")
@app.route("/executive-overview")
def executive_overview():
    total_patients = patients["Id"].nunique()
    total_encounters = len(encounters)
    total_activity = len(procedures) + len(medications)
    coverage_pct = 76.5

    encounters_copy = encounters.copy()
    encounters_copy["START"] = pd.to_datetime(encounters_copy["START"], errors="coerce")

    encounter_trend = (
        encounters_copy.dropna(subset=["START"])
        .assign(Month=encounters_copy["START"].dt.to_period("M").astype(str))
        .groupby("Month")
        .size()
        .reset_index(name="Total Encounters")
    )

    fig1 = px.line(
        encounter_trend,
        x="Month",
        y="Total Encounters",
        title="Total Encounters Over Time"
    )
    fig1.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    encounters_chart = fig1.to_html(full_html=False)

    payer_summary = (
        encounters.groupby("PAYER")
        .size()
        .reset_index(name="Total Encounters")
        .sort_values(by="Total Encounters", ascending=False)
        .head(10)
    )

    fig2 = px.bar(
        payer_summary,
        x="Total Encounters",
        y="PAYER",
        orientation="h",
        title="Top Payers by Encounters"
    )
    fig2.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(categoryorder="total ascending")
    )
    payer_chart = fig2.to_html(full_html=False)

    return render_template(
        "executive_overview.html",
        total_patients=total_patients,
        total_encounters=total_encounters,
        total_claim_cost=total_activity,
        coverage_pct=coverage_pct,
        encounters_chart=encounters_chart,
        payer_chart=payer_chart
    )


@app.route("/payer-performance")
def payer_performance():
    payer_summary = (
        encounters.groupby("PAYER")
        .agg(Total_Encounters=("PAYER", "count"))
        .reset_index()
    )

    payer_summary["Total Claim Cost"] = payer_summary["Total_Encounters"] * 100
    payer_summary["Payer Coverage"] = payer_summary["Total Claim Cost"] * 0.75
    payer_summary["Denied Amount"] = payer_summary["Total Claim Cost"] - payer_summary["Payer Coverage"]
    payer_summary["Coverage %"] = (
        payer_summary["Payer Coverage"] / payer_summary["Total Claim Cost"]
    ) * 100

    payer_summary = payer_summary.sort_values(
        by="Denied Amount", ascending=False
    ).head(10)

    fig = px.bar(
        payer_summary,
        x="Denied Amount",
        y="PAYER",
        orientation="h",
        title="Revenue Leakage by Payer"
    )
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(categoryorder="total ascending")
    )

    payer_chart = fig.to_html(full_html=False)
    payer_table = payer_summary.round(2).to_html(classes="table table-striped", index=False)

    return render_template(
        "payer_performance.html",
        payer_chart=payer_chart,
        payer_table=payer_table
    )


@app.route("/opportunity-matrix")
def opportunity_matrix():
    payer_encounters = (
        encounters.groupby("PAYER")
        .size()
        .reset_index(name="Total Encounters")
    )

    payer_conditions = (
        encounters[["PATIENT", "PAYER"]]
        .drop_duplicates()
        .merge(
            conditions.groupby("PATIENT").size().reset_index(name="Condition Count"),
            on="PATIENT",
            how="left"
        )
    )

    payer_burden = (
        payer_conditions.groupby("PAYER")["Condition Count"]
        .sum()
        .reset_index()
    )

    opportunity_df = payer_encounters.merge(payer_burden, on="PAYER", how="left")
    opportunity_df["Condition Count"] = opportunity_df["Condition Count"].fillna(0)

    opportunity_df["Total Claim Cost"] = opportunity_df["Total Encounters"] * 100
    opportunity_df["Payer Coverage"] = opportunity_df["Total Claim Cost"] * 0.75
    opportunity_df["Denied Amount"] = opportunity_df["Total Claim Cost"] - opportunity_df["Payer Coverage"]

    opportunity_df["Burden Index"] = (
        opportunity_df["Condition Count"] / opportunity_df["Total Encounters"]
    ) * 1000

    opportunity_df = opportunity_df.sort_values(
        by="Denied Amount", ascending=False
    ).head(10)

    fig = px.scatter(
        opportunity_df,
        x="Total Encounters",
        y="Burden Index",
        size="Denied Amount",
        color="PAYER",
        hover_name="PAYER",
        title="Strategic Payer Opportunity Segmentation",
        size_max=60
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    opportunity_chart = fig.to_html(full_html=False)
    opportunity_table = opportunity_df.round(2).to_html(classes="table table-striped", index=False)

    return render_template(
        "opportunity_matrix.html",
        opportunity_chart=opportunity_chart,
        opportunity_table=opportunity_table
    )


@app.route("/predictive-analytics", methods=["GET"])
def predictive_analytics():
    monthly_claim_copy = monthly_claim.copy()
    claim_forecast_copy = claim_forecast.copy()

    monthly_claim_copy["MonthStart"] = pd.to_datetime(monthly_claim_copy["MonthStart"], errors="coerce")
    claim_forecast_copy["MonthStart"] = pd.to_datetime(claim_forecast_copy["MonthStart"], errors="coerce")

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if start_date:
        monthly_claim_copy = monthly_claim_copy[monthly_claim_copy["MonthStart"] >= pd.to_datetime(start_date)]
        claim_forecast_copy = claim_forecast_copy[claim_forecast_copy["MonthStart"] >= pd.to_datetime(start_date)]

    if end_date:
        monthly_claim_copy = monthly_claim_copy[monthly_claim_copy["MonthStart"] <= pd.to_datetime(end_date)]
        claim_forecast_copy = claim_forecast_copy[claim_forecast_copy["MonthStart"] <= pd.to_datetime(end_date)]

    total_actual = monthly_claim_copy["Sum of TotalClaimCost"].sum()
    total_forecast = claim_forecast_copy["ForecastClaimCost"].sum()

    actual_vs_forecast = monthly_claim_copy[["MonthStart", "Sum of TotalClaimCost"]].copy()
    actual_vs_forecast.rename(columns={"Sum of TotalClaimCost": "Claim Amount"}, inplace=True)
    actual_vs_forecast["Type"] = "Actual"

    forecast_plot = claim_forecast_copy[["MonthStart", "ForecastClaimCost"]].copy()
    forecast_plot.rename(columns={"ForecastClaimCost": "Claim Amount"}, inplace=True)
    forecast_plot["Type"] = "Forecast"

    combined_plot = pd.concat([actual_vs_forecast, forecast_plot], ignore_index=True)
    combined_plot = combined_plot.sort_values("MonthStart")

    line_fig = px.line(
        combined_plot,
        x="MonthStart",
        y="Claim Amount",
        color="Type",
        title="Monthly Healthcare Claim Cost - Actual vs Forecast"
    )

    line_fig.update_traces(line=dict(width=3))
    line_fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    line_chart = line_fig.to_html(full_html=False)

    claim_type_df = pd.DataFrame({
        "Claim Type": ["Actual", "Forecast"],
        "Claim Amount": [total_actual, total_forecast]
    })

    bar_fig = px.bar(
        claim_type_df,
        x="Claim Amount",
        y="Claim Type",
        orientation="h",
        title="Claim Amount by Claim Type"
    )
    bar_fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    claim_type_chart = bar_fig.to_html(full_html=False)

    return render_template(
        "predictive_analytics.html",
        total_actual=total_actual,
        total_forecast=total_forecast,
        line_chart=line_chart,
        claim_type_chart=claim_type_chart,
        start_date=start_date,
        end_date=end_date
    )
@app.route('/disease-forecast')
def disease_forecast():
    import pandas as pd

    actual = pd.read_csv("data/disease_actual_claims.csv")
    forecast = pd.read_csv("data/sarima_forecast.csv")
    summary = pd.read_csv("data/model_comparison_summary.csv")

    # keep only SARIMA rows for summary table
    summary = summary[summary["Model"] == "SARIMA"].copy()
    summary = summary.rename(columns={
        "TotalForecastNext6Months": "ForecastClaimAmount_Next6Months"
    })

    return render_template(
        "disease_forecast.html",
        actual=actual.to_dict(orient="records"),
        forecast=forecast.to_dict(orient="records"),
        summary=summary.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)