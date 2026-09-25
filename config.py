from urllib.request import localhost

from dotenv import load_dotenv
import os


load_dotenv()


# =========================
# Secrets
# =========================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# External Services
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_LOCAL_HOST = "localhost"
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")


# =========================
# Pinecone
# =========================

PINECONE_INDEX_NAME1 = "legal-engine"
PINECONE_INDEX_NAME2 = "semantic-cache"


# =========================
# Cache Configuration
# =========================

SEMANTIC_CACHE_THRESHOLD = 0.7
REDIS_CACHE_TTL = 1000


# =========================
# LLM Configuration
# =========================

OLLAMA = "ollama"
GEMINI = "gemini"
GROQ = "groq"

LLM_PROVIDER = GROQ

GEMINI_MODEL = "gemini-3.8-flash"
OLLAMA_MODEL = "llama3.2:3b"
GROQ_MODEL = "openai/gpt-oss-20b"