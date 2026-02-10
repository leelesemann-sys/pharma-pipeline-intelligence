"""
Database connection utilities for Pharma Pipeline Dashboard.
Provides cached connection and query execution using pyodbc + Azure SQL.

Azure SQL Serverless auto-pauses after inactivity and can take up to 60s
to resume. During resume, connections may fail with:
  - 40613: "Database not currently available"
  - 18456: "Login failed" (transient during resume)
This module handles both with retry + exponential backoff.
"""

import streamlit as st
import pyodbc
import pandas as pd
import os
import time

# Connection Timeout=120 gives Azure SQL Serverless enough time to auto-resume
DEFAULT_CONN_STR = os.environ.get(
    "AZURE_SQL_CONN_STR",
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=tcp:pharma-pipeline-sql.database.windows.net,1433;"
    "Database=pharma_pipeline_db;"
    "Uid=pharmaadmin;"
    "Pwd={SET_AZURE_SQL_CONN_STR_ENV_VAR};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=120;"
)

MAX_RETRIES = 5
BASE_DELAY = 5  # seconds, doubles each retry (5, 10, 20, 40, 80)

# Module-level connection (not Streamlit-cached to avoid caching failures)
_conn = None


def _get_conn_str():
    """Resolve connection string: secrets > env > default.
    Streamlit secrets take priority (deployment), then env var, then default.
    Note: env var AZURE_SQL_CONN_STR may contain stale credentials —
    Streamlit secrets or the hardcoded default are preferred.
    """
    try:
        conn_str = st.secrets.get("AZURE_SQL_CONN_STR", None)
        if conn_str:
            return conn_str
    except Exception:
        pass
    return DEFAULT_CONN_STR


def _is_retryable(err_str):
    """Check if an error is retryable (Azure SQL Serverless resume)."""
    retryable_codes = ["40613", "18456", "08S01", "08001", "40197", "40501", "49918"]
    retryable_msgs = ["not currently available", "communication link", "login failed"]
    return (
        any(code in err_str for code in retryable_codes)
        or any(msg in err_str.lower() for msg in retryable_msgs)
    )


def get_connection():
    """Get or create a DB connection. Re-creates on failure."""
    global _conn
    if _conn is not None:
        try:
            # Quick health check
            _conn.cursor().execute("SELECT 1").fetchone()
            return _conn
        except Exception:
            # Connection is broken, recreate
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    # Create new connection with retry
    conn_str = _get_conn_str()
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = pyodbc.connect(conn_str)
            conn.cursor().execute("SELECT 1").fetchone()
            _conn = conn
            return _conn
        except Exception as e:
            last_err = e
            err_str = str(e)
            if _is_retryable(err_str) and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)
            elif not _is_retryable(err_str):
                break
    # All retries failed
    _conn = None
    raise ConnectionError(f"Database connection failed after {MAX_RETRIES} attempts: {last_err}")


def _execute_with_retry(query, params=None):
    """Execute a query with connection retry logic. Raises on failure."""
    p = list(params) if isinstance(params, tuple) else params
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = get_connection()
            if p:
                return pd.read_sql(query, conn, params=p)
            return pd.read_sql(query, conn)
        except ConnectionError:
            raise  # propagate — don't cache failures
        except Exception as e:
            last_err = e
            # Force new connection on next attempt
            global _conn
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
            err_str = str(e)
            if _is_retryable(err_str) and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)
            elif not _is_retryable(err_str):
                break
    raise ConnectionError(f"Query failed after {MAX_RETRIES} attempts: {last_err}")


@st.cache_data(ttl=3600)
def _run_query_cached(query, params=None):
    """
    Inner cached function. Raises on failure so Streamlit
    does NOT cache empty DataFrames from connection errors.
    """
    return _execute_with_retry(query, params)


def run_query(query, params=None):
    """
    Safe cached query execution. Returns a pandas DataFrame.
    On DB failure: shows st.error and returns empty DataFrame,
    but the failure is NOT cached (next page load retries).
    params should be a tuple for hashability (Streamlit cache requirement).
    """
    try:
        return _run_query_cached(query, params)
    except ConnectionError as e:
        st.error(str(e))
        return pd.DataFrame()


def run_query_uncached(query, params=None):
    """Uncached query for real-time data needs."""
    try:
        return _execute_with_retry(query, params)
    except ConnectionError as e:
        st.error(str(e))
        return pd.DataFrame()
