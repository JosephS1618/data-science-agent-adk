import os
from google.adk.agents import Agent
from .tools import clean_datasets

MODEL = os.getenv("ADK_MODEL", "gemini-2.0-flash")

data_engineer_instruction = """
You are the Data Engineer Sub-Agent.
Your primary responsibility is to sanitize and prepare datasets for analysis.
You are equipped with the `clean_datasets` tool, which cleans the raw CSV files into optimized Parquet files in the sandbox.

When requested to clean the data, you MUST:
1. Invoke the `clean_datasets` tool.
2. Return a JSON formatted summary of the operation containing the paths to the clean datasets and their row counts.
"""

data_engineer_agent = Agent(
    model=MODEL,
    name="data_engineer_agent",
    instruction=data_engineer_instruction,
    tools=[clean_datasets]
)
