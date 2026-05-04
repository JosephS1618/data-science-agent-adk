import os
from google.adk.agents import Agent
from .tools import (
    forecast_time_series,
    predict_probability,
    cluster_entities,
    calculate_rfm_scores,
    detect_anomalies_isolation_forest,
    calculate_z_scores,
    extract_topics,
    get_top_ngrams,
    calculate_correlation_matrix,
    run_linear_regression,
    generate_cohort_retention_matrix
)

MODEL = os.getenv("ADK_MODEL", "gemini-2.0-flash")

stats_agent_instruction = """
You are the Statistical Sub-Agent, the Mathematical Engine of the system.
Your responsibility is to perform domain-specific data science analyses using your strict, deterministic toolkit.

You have access to a suite of advanced mathematical tools for predictions, clustering, anomaly detection, NLP, and regression.

**Semantic Data Dictionary (CRITICAL - DO NOT HALLUCINATE FILENAMES):**
- If the analysis involves 2025 store transactions, sales data, or revenue, you MUST use the exact filename: `clean_sample.parquet`
- If the analysis involves customer data, users, or subscriptions, you MUST use the exact filename: `clean_customers.parquet`
NEVER invent or guess filenames (e.g., do not use `2025_store_transactions.parquet`).

When receiving a request, you must:
1. Determine the appropriate tool from your toolkit to answer the query.
2. Invoke the tool with the correct file path and parameters.
3. Return the exact statistical metrics, p-values, and summary insights provided by the tool.

DO NOT invent statistical results. Only return what your tools calculate.

Your final response to the Root Agent MUST explicitly name the exact Python tool you used (e.g., `predict_probability`) and the exact column names you passed as arguments, followed by the statistical output.
"""

stats_agent = Agent(
    model=MODEL,
    name="stats_agent",
    instruction=stats_agent_instruction,
    tools=[
        forecast_time_series,
        predict_probability,
        cluster_entities,
        calculate_rfm_scores,
        detect_anomalies_isolation_forest,
        calculate_z_scores,
        extract_topics,
        get_top_ngrams,
        calculate_correlation_matrix,
        run_linear_regression,
        generate_cohort_retention_matrix
    ]
)
