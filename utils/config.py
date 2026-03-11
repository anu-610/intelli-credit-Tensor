import os
from dotenv import load_dotenv

def load_config():
    """Loads environment variables."""
    load_dotenv()
    
    # Check if keys are loaded
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not found in environment variables.")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not found in environment variables.")

load_config()
