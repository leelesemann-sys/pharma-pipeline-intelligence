"""Shared configuration for ML pipeline."""
import os

# --- DB ---
DB_CONN_STR = os.environ.get(
    "AZURE_SQL_CONN_STR",
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:pharma-pipeline-sql.database.windows.net,1433;"
    "Database=pharma_pipeline_db;"
    "Uid=pharmaadmin;"
    "Pwd={SET_AZURE_SQL_CONN_STR_ENV_VAR};"
    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
)

# --- Versioning ---
FEATURE_VERSION = "v1.0"
MODEL_VERSION = "v1.0"
PIPELINE_VERSION = "v1.0"

# --- MLflow ---
MLFLOW_TRACKING_DIR = "mlruns"
MLFLOW_EXPERIMENT_FEATURES = "pharma_pipeline_feature_engineering"
MLFLOW_EXPERIMENT_TRAINING = "pharma_pipeline_model_training"

# --- Paths ---
LOG_DIR = "logs"
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"

# --- Feature Engineering ---
TEMPORAL_LEAK_CHECK = True
IMPUTATION_STRATEGY = "median"
BATCH_INSERT_SIZE = 1000
CORRELATION_THRESHOLD = 0.95

# --- Phase Mapping ---
PHASE_NUMERIC = {
    "early_phase1": 0.5, "phase1": 1.0, "phase1_phase2": 1.5,
    "phase2": 2.0, "phase2_phase3": 2.5, "phase3": 3.0, "phase4": 4.0,
}
PHASE_ORDER = ["early_phase1", "phase1", "phase1_phase2", "phase2", "phase2_phase3", "phase3", "phase4"]

# Next phase mapping for target variable
NEXT_PHASE = {
    "early_phase1": ["phase1", "phase1_phase2", "phase2", "phase2_phase3", "phase3", "phase4"],
    "phase1": ["phase1_phase2", "phase2", "phase2_phase3", "phase3", "phase4"],
    "phase1_phase2": ["phase2", "phase2_phase3", "phase3", "phase4"],
    "phase2": ["phase2_phase3", "phase3", "phase4"],
    "phase2_phase3": ["phase3", "phase4"],
    "phase3": ["phase4"],  # phase4 = approved/post-marketing
}

# MoA Generation mapping
MOA_GENERATION = {
    "Sulfonylharnstoff": 1, "Thiazolidinedione (PPAR-gamma)": 1, "Meglitinid": 1,
    "Alpha-Glucosidase-Inhibitor": 1, "Bile Acid Sequestrant": 1, "Dopamin-Agonist (D2)": 1,
    "DPP-4 Inhibitor": 2, "SGLT2 Inhibitor": 2, "GLP-1 Receptor Agonist": 2,
    "Insulin (Basal)": 2, "Insulin (Rapid-acting)": 2, "Insulin (Ultra-long-acting)": 2,
    "Insulin (Weekly)": 2, "Biguanide": 2, "SGLT1/SGLT2 Inhibitor": 2,
    "GIP/GLP-1 Dual Agonist": 3, "GLP-1/Glucagon Dual Agonist": 3,
    "GLP-1/GIP/Glucagon Triple Agonist": 3, "Oral GLP-1 RA (Small Molecule)": 3,
    "Amylin Analogue": 3, "THR-beta Agonist": 3, "FXR Agonist": 3,
}

# --- Model Training ---
RANDOM_SEED = 42
CV_SPLITS = [
    {"train_before": "2018-01-01", "test_after": "2018-01-01", "test_before": "2019-01-01"},
    {"train_before": "2019-01-01", "test_after": "2019-01-01", "test_before": "2020-01-01"},
    {"train_before": "2020-01-01", "test_after": "2020-01-01", "test_before": "2021-01-01"},
    {"train_before": "2021-01-01", "test_after": "2021-01-01", "test_before": "2022-01-01"},
    {"train_before": "2022-01-01", "test_after": "2022-01-01", "test_before": "2099-01-01"},
]
PRIMARY_SPLIT_CUTOFF = "2020-01-01"
