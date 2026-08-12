import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# プロジェクトルートの .env ファイルを自動ロード
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    QA_JSON_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "globallogic_qa.json")
    COLLECTION_NAME: str = "globallogic_qa_ruri_v3"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
