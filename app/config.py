import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# Configuration Management
# ============================================================================
# This module manages all configuration for the AI Interviewer application.
# Configuration can be set via environment variables or defaults are used.
# 
# To use custom values, create a .env file in the project root or set
# environment variables directly in your system/shell.

# ============================================================================
# Load Environment Variables from .env File
# ============================================================================

# Determine base directory first
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)

# ============================================================================
# Base Directory Configuration
# ============================================================================
"""Project root directory path"""

# ============================================================================
# ChromaDB Configuration
# ============================================================================

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(BASE_DIR / "data" / "chroma")
)
"""ChromaDB persistence directory - stores embeddings and vector data"""

# ============================================================================
# Embedding Model Configuration
# ============================================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2"
)
"""
Sentence Transformer model for embedding generation.
Options: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
See: https://www.sbert.net/docs/pretrained_models.html
"""

# ============================================================================
# Text Chunking Configuration
# ============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
"""Size of text chunks for embedding (characters)"""

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
"""Overlap between consecutive chunks (characters)"""

# ============================================================================
# Groq LLM Configuration (IMPORTANT: Set your API key here or in .env)
# ============================================================================

GROK_API_KEY = os.getenv("GROK_API_KEY")
"""
Groq API Key for LLM calls.

⚠️ IMPORTANT: Set this via environment variable BEFORE running the app.

Options:
1. Create a .env file in project root:
   GROK_API_KEY=gsk_your_actual_key_here

2. Set system environment variable:
   # Windows PowerShell:
   $env:GROK_API_KEY="gsk_your_actual_key_here"
   
   # Linux/Mac:
   export GROK_API_KEY="gsk_your_actual_key_here"

3. The header X-Grok-API-Key can still override this for per-request customization.

Get your API key from: https://console.groq.com/keys
"""

if not GROK_API_KEY:
    raise ValueError(
        "❌ GROK_API_KEY environment variable is not set!\n"
        "Please set your Groq API key before running the application.\n"
        "\nQuick Setup:\n"
        "1. Set environment variable: GROK_API_KEY=gsk_your_key\n"
        "2. Or create .env file with: GROK_API_KEY=gsk_your_key\n"
        "3. Or install python-dotenv: pip install python-dotenv\n"
        "\nGet your key from: https://console.groq.com/keys"
    )

GROK_MODEL = os.getenv("GROK_MODEL", "grok-2")
"""Groq model to use for LLM calls. Options: grok-2, grok-beta, etc."""

# ============================================================================
# Application Configuration
# ============================================================================

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
"""Enable debug mode for development"""

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
"""Allowed CORS origins. Use "*" to allow all (dev only)"""
