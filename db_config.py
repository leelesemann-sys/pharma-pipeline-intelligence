"""
Centralized database configuration.
Reads credentials from environment variable AZURE_SQL_CONN_STR.

For local development, create a .env file (gitignored) with:
    AZURE_SQL_CONN_STR=Driver={ODBC Driver 17 for SQL Server};Server=tcp:pharma-pipeline-sql.database.windows.net,1433;Database=pharma_pipeline_db;Uid=pharmaadmin;Pwd={YOUR_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;

For Streamlit Cloud, set it in .streamlit/secrets.toml or the app's Secrets UI.
"""
import os

CONN_STR = os.environ.get(
    "AZURE_SQL_CONN_STR",
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:pharma-pipeline-sql.database.windows.net,1433;"
    "Database=pharma_pipeline_db;"
    "Uid=pharmaadmin;"
    "Pwd={SET_AZURE_SQL_CONN_STR_ENV_VAR};"
    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
)
