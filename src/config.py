import os

from dotenv import load_dotenv


load_dotenv()

# --- Project Root ---
# Uses the structure from your previous version
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_TABLE_NAME = "interventions" 
# --- File Paths ---
# Uses the paths from your previous version
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "GPT_Input_DB(Sheet1).csv")
STRUCTURED_DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "structured_db.sqlite")
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "vector_store")
KNOWLEDGE_GAP_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_gaps.log") # Kept from intermediate versions

# --- LLM & Embedding Models ---
# UPDATED based on recommendations
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
VECTOR_COLLECTION_NAME = "road_safety_bge_combined" # Updated to reflect model/strategy

# --- NEW: Flag for Vector DB Text ---
# Controls embedding strategy as recommended
USE_COMBINED_TEXT_FOR_EMBEDDING = True
