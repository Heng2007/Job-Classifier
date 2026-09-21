"""Project-wide constants: filesystem paths, category names, database filename.

Responsible for: the single place every other module reads paths and the
category list from, so nothing hardcodes a directory or a category string.

Inputs: none — this module holds literals only, no logic and no environment
reads.

Outputs: path constants for the data, model and database locations; the 8
category names used by the classifier; the database filename.

How I'll know it works: every module imports its paths and categories from
here, and changing a path in this file changes it everywhere.
"""

# --- Paths ---
PROJECT_ROOT = "."
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
DB_DIR = "db"
MODELS_DIR = "models"
SCHEMA_PATH = "db/schema.sql"
SKILLS_PROFILE_PATH = "my_skills.json"
SEED_LABELS_PATH = "seed_labels.csv"
GREENHOUSE_TOKENS_PATH = "data/raw/greenhouse_tokens.txt"
KAGGLE_INPUT_DATA = "data/raw/postings.csv"        
KAGGLE_OUTPUT_DATA = "data/raw/kaggle.csv"
UOFT_INPUT_DATA = "data/raw/uoft_workstudy_summer2025.csv"
UOFT_OUTPUT_DATA = "data/raw/uoft.csv"
GREENHOUSE_OUTPUT_DATA = "data/raw/greenhouse.csv"
PROCESSED_OUTPUT_DATA = "data/processed/all_postings.csv"
HAND_LABELED_DATA = "data/processed/hand_label.xlsx"
HAND_LABEL_TOPUP_DATA = "data/processed/hand_label_topup.xlsx"
HAND_LABELED_DONE = "data/processed/hand_label_me.xlsx"
HAND_LABEL_TOPUP_DONE = "data/processed/hand_label_topup_me.xlsx"
LLM_LABELED_DATA = "data/processed/llm_labels.csv"
LABEL_CORRECTIONS_DATA = "data/processed/label_corrections.csv"
FINAL_LABELS_DATA = "data/processed/labels_final.csv"


# --- Database ---
DB_FILENAME = "jobintel.sqlite"
DB_PATH = "db/jobintel.sqlite"

# --- Categories (5-way classifier) ---
# Collapsed from 8 on 2026-08-23, after labeling all 6,636 postings revealed
# four classes under 1% of the data: Data engineering 62, NLP / LLM 48,
# Data analyst 46, Quant / finance 10. Macro-F1 weights every class equally,
# so a 10-example class measures sampling noise, not the model.
CATEGORIES = [
    "Machine learning / AI",          # was Machine learning + NLP / LLM
    "Data engineering / analytics",   # was Data engineering + Data analyst
    "Software engineering",
    "Research assistant",
    "Not relevant",                   # absorbed Quant / finance (10 postings)
]

NOT_RELEVANT_CATEGORY = "Not relevant"

# --- Sources ---
SOURCES = ["greenhouse", "kaggle", "uoft"]

# --- Cleaning thresholds ---
MIN_DESCRIPTION_LENGTH = 200

# --- Sampling / reproducibility ---
KAGGLE_SAMPLE_SIZE = 1200
RANDOM_SEED = 42            
# Batch Number
BATCH_SIZE = 40    
#Leaning Rate
LEARNING_RATE = 0.01
#Model Name
MODEL_NAME_MLP = "Multilayer Perceptron"
MODEL_NAME_LOG = "Logistic Regression"

# --- Junior-eligibility title regexes (Streamlit tab 2 filter) ---
JUNIOR_TITLE_PATTERN = r"\b(?:intern|interns|internship|internships|junior|new grad|new graduate|research assistant)\b"
SENIOR_TITLE_PATTERN = r"\b(?:senior|sr|staff|principal|lead)\b"

# --- Near-miss threshold (match_scoring) ---
NEAR_MISS_MAX_MISSING_SKILLS = 2

# --- External endpoints ---
GREENHOUSE_API_BASE = "https://boards-api.greenhouse.io/v1/boards"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
