"""
Configuration Module for Production Support RAG System
Centralized configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config/.env
config_dir = Path(__file__).parent.parent / "config"
env_path = config_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Config:
    """
    Configuration settings for the RAG system
    """
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent  # Go up to project root from source/
    
    # Excel file configuration
    EXCEL_PATH = os.getenv("EXCEL_PATH", str(BASE_DIR / "Input.xlsx"))
    
    # Database configuration
    DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "chroma_db"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "prod_support")
    
    # Embedding model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Retrieval settings
    TOP_K = int(os.getenv("TOP_K", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.5"))
    
    # Google API configuration (optional)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Streamlit configuration
    PAGE_TITLE = "Production Support Assistant"
    PAGE_ICON = "🤖"
    
    # Response templates
    NO_ANSWER_RESPONSE = "I do not have the answer for this."
    
    # System prompts
    SYSTEM_PROMPT = """You are a polite and helpful production support assistant. 
Answer the user's question based ONLY on the following context from the knowledge base.

Context:
{context}

User Question: {query}

Instructions:
- Provide accurate and to-the-point answers
- Be polite and professional
- If the context doesn't contain the answer, respond with: "I do not have the answer for this."
- Do not make up information outside the provided context
- Keep your response concise and relevant

Answer:"""
    
    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """
        Get absolute path from relative path
        
        Args:
            relative_path: Relative path from base directory
            
        Returns:
            Absolute path as string
        """
        return str(cls.BASE_DIR / relative_path)
    
    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration settings
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if Excel file exists
        if not os.path.exists(cls.EXCEL_PATH):
            errors.append(f"Excel file not found: {cls.EXCEL_PATH}")
        
        # Validate TOP_K value
        if cls.TOP_K < 1 or cls.TOP_K > 10:
            errors.append(f"TOP_K must be between 1 and 10, got: {cls.TOP_K}")
        
        # Validate similarity threshold
        if cls.SIMILARITY_THRESHOLD < 0:
            errors.append(f"SIMILARITY_THRESHOLD must be positive, got: {cls.SIMILARITY_THRESHOLD}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def print_config(cls):
        """
        Print current configuration
        """
        print("="*50)
        print("Configuration Settings")
        print("="*50)
        print(f"Excel Path: {cls.EXCEL_PATH}")
        print(f"Database Path: {cls.DB_PATH}")
        print(f"Collection Name: {cls.COLLECTION_NAME}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Top K: {cls.TOP_K}")
        print(f"Similarity Threshold: {cls.SIMILARITY_THRESHOLD}")
        print(f"Google API Key: {'Set' if cls.GOOGLE_API_KEY else 'Not set'}")
        print(f"Gemini Model: {cls.GEMINI_MODEL}")
        print("="*50)


if __name__ == "__main__":
    # Print and validate configuration
    Config.print_config()
    
    is_valid, errors = Config.validate_config()
    if is_valid:
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
