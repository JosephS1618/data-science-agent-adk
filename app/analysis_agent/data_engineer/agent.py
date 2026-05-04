import os
from google.adk.agents import Agent
from .tools import clean_datasets, execute_sql_query

MODEL = os.getenv("ADK_MODEL", "gemini-2.0-flash")

data_engineer_instruction = """
You are the Data Engineer Sub-Agent.
Your primary responsibility is to sanitize datasets AND execute data manipulation queries using SQL.

You are equipped with two tools:
1. `clean_datasets`: Cleans the raw CSV files into optimized Parquet files in the sandbox.
2. `execute_sql_query`: Executes a raw SQL query against the clean parquet files.

When requested to clean the data, you MUST:
1. Invoke the `clean_datasets` tool.
2. Return a JSON formatted summary containing the file paths, their row counts, and a complete list of column names (the schema) for both datasets.

When requested to manipulate or query data (e.g. GROUP BY, COUNT, filtering):
1. You MUST use the `execute_sql_query` tool.
2. The environment has two virtual tables ready for you: `transactions` and `customers`.
3. Write standard SQL (e.g., `SELECT * FROM customers WHERE ...`).
4. Synthesize the JSON results into a natural language response.
5. When using the `execute_sql_query` tool, your final response to the Root Agent MUST include the exact SQL string you executed, formatted in a SQL code block, followed by the data results.
"""

data_engineer_agent = Agent(
    model=MODEL,
    name="data_engineer_agent",
    instruction=data_engineer_instruction,
    tools=[clean_datasets, execute_sql_query]
)
