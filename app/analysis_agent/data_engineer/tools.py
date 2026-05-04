import pandas as pd
import os

SANDBOX_DIR = "/Users/josephsong/Desktop/Projects/Personal/adk-stats-agent-demo-v2/_sandbox"

def clean_datasets(transaction_filename: str = "sample.csv", customer_filename: str = "customers-1000.csv") -> dict:
    """
    Cleans and normalizes the provided transaction and customer datasets, saving them as Parquet files.
    
    Args:
        transaction_filename: The name of the transaction CSV file in the sandbox.
        customer_filename: The name of the customer CSV file in the sandbox.
        
    Returns:
        A dictionary containing the status and sandbox paths to the cleaned Parquet files.
    """
    trans_path = os.path.join(SANDBOX_DIR, transaction_filename)
    cust_path = os.path.join(SANDBOX_DIR, customer_filename)
    
    # Clean transactions
    df_trans = pd.read_csv(trans_path)
    df_trans['date'] = pd.to_datetime(df_trans['date'], errors='coerce')
    df_trans.dropna(subset=['date'], inplace=True)
    numeric_cols = df_trans.select_dtypes(include=['number']).columns
    df_trans[numeric_cols] = df_trans[numeric_cols].fillna(0)
    string_cols = df_trans.select_dtypes(include=['object', 'string']).columns
    df_trans[string_cols] = df_trans[string_cols].fillna("Unknown")
    clean_trans_path = os.path.join(SANDBOX_DIR, "clean_sample.parquet")
    df_trans.to_parquet(clean_trans_path)
    
    # Clean customers
    df_cust = pd.read_csv(cust_path)
    if 'Subscription Date' in df_cust.columns:
        df_cust['Subscription Date'] = pd.to_datetime(df_cust['Subscription Date'], errors='coerce')
    df_cust.fillna("Unknown", inplace=True)
    clean_cust_path = os.path.join(SANDBOX_DIR, "clean_customers.parquet")
    df_cust.to_parquet(clean_cust_path)
    
    return {
        "status": "success",
        "transactions_path": clean_trans_path,
        "customers_path": clean_cust_path,
        "transactions_rows": len(df_trans),
        "customers_rows": len(df_cust),
        "transactions_columns": df_trans.columns.tolist(),
        "customers_columns": df_cust.columns.tolist()
    }
