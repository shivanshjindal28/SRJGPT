from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Google API Key
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Upload directory
    upload_dir: str = "uploads"
    
    # Maximum file size (10MB)
    max_file_size: int = 10 * 1024 * 1024
    
    # Allowed file types
    allowed_file_types: list = [".pdf", ".docx", ".png", ".jpg", ".jpeg"]

# Create settings instance
settings = Settings() 