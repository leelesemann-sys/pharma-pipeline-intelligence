"""
Database connection utilities for Pharma Pipeline Dashboard.
Provides cached connection and query execution using pymssql + Azure SQL.

pymssql uses the TDS protocol directly — no ODBC driver needed.
This makes it compatible with Streamlit Cloud (Linux) without extra system packages.

Azure SQL Serverless auto-pauses after inactivity and can take up to 60s
to resume. During resume, connections may fail with:
  - 40613: "Database not currently available"
  - 18456: "Login failed" (transient during resume)
This module handles both with retry + exponential backoff.
"""

import streamlit as st
import pymssql
import pandas as pd
import os
import time
import re

# Default connection parameters (fallback for local dev)
_DEFAULT_SERVER = "pharma-pipeline-sql.database.windows.net"
_DEFAULT_DATABASE = "pharma_pipeline_db"
_DEFAULT_USER = "pharmaadmin"
_DEFAULT_PASSWORD = ""  # Must be set via secrets or env

MAX_RETRIES = 5
BASE_DELAY = 5  # seconds, doubles each retry (5, 10, 20, 40, 80)

# Module-level connection (not Streamlit-cached to avoid caching failures)
_conn = None


def _parse_odbc_conn_str(conn_str):
    """Parse an ODBC-style connection string into pymssql parameters.
    Supports both ODBC format (Key=Value;...) and pymssql-native secrets.
    """
    # If it looks like an ODBC connection string
    if ";" in conn_str and "=" in conn_str:
        parts = {}
        for part in conn_str.split(";"):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                parts[key.strip().lower()] = val.strip()

        server = parts.get("server", _DEFAULT_SERVER)
        # Strip tcp: prefix and port suffix for pymssql
        server = re.sub(r"^tcp:", "", server)
        # pymssql uses server:port format, keep port if present
        # e.g. "pharma-pipeline-sql.database.windows.net,1433" → remove ,1433 (default port)
        server = server.replace(",1433", "")

        return {
            "server": server,
            "database": parts.get("database", _DEFAULT_DATABASE),
            "user": parts.get("uid", _DEFAULT_USER),
            "password": parts.get("pwd", _DEFAULT_PASSWORD).strip("{}"),
        }
    return None


def _get_conn_params():
    """Resolve connection parameters: secrets > env > default.
    Streamlit secrets take priority (deployment), then env var, then default.

    Supports three formats in secrets:
      1. AZURE_SQL_CONN_STR as ODBC string (auto-parsed)
      2. Individual keys: db_server, db_database, db_user, db_password
      3. Environment variable AZURE_SQL_CONN_STR (ODBC string)
    """
    # Try Streamlit secrets first
    try:
        # Format 1: ODBC connection string in secrets
        conn_str = st.secrets.get("AZURE_SQL_CONN_STR", None)
        if conn_str:
            parsed = _parse_odbc_conn_str(conn_str)
            if parsed:
                return parsed

        # Format 2: Individual secret keys
        server = st.secrets.get("db_server", None)
        if server:
            return {
                "server": server,
                "database": st.secrets.get("db_database", _DEFAULT_DATABASE),
                "user": st.secrets.get("db_user", _DEFAULT_USER),
                "password": st.secrets.get("db_password", _DEFAULT_PASSWORD),
            }
    except Exception:
        pass

    # Try environment variable
    env_str = os.environ.get("AZURE_SQL_CONN_STR", "")
    if env_str:
        parsed = _parse_odbc_conn_str(env_str)
        if parsed:
            return parsed

    # Fallback defaults (won't work without password)
    return {
        "server": _DEFAULT_SERVER,
        "database": _DEFAULT_DATABASE,
        "user": _DEFAULT_USER,
        "password": _DEFAULT_PASSWORD,
    }


def _is_retryable(err_str):
    """Check if an error is retryable (Azure SQL Serverless resume)."""
    retryable_codes = ["40613", "18456", "08S01", "08001", "40197", "40501", "49918"]
    retryable_msgs = ["not currently available", "communication link", "login failed",
                      "adaptive server is unavailable", "connection timed out"]
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
            cursor = _conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return _conn
        except Exception:
            # Connection is broken, recreate
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    # Create new connection with retry
    params = _get_conn_params()
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = pymssql.connect(
                server=params["server"],
                user=params["user"],
                password=params["password"],
                database=params["database"],
                login_timeout=120,
                timeout=120,
                tds_version="7.4",
                conn_properties="",
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
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
