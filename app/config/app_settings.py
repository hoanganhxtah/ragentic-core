from pydantic_settings import BaseSettings

from app.config.env.server_config import ServerSettings
from app.config.env.llm_config import LLMSettings, APIKeySettings
from app.config.env.web_search_config import WebSearchSettings
from app.config.env.agent_config import AgentSettings
from app.config.env.rag_client_config import RAGClientSettings


class ApplicationSettings(BaseSettings):
    """
    Core application metadata settings.
    Only reads variables prefixed with APP_ from .env
    """

    APP_NAME: str = "Super-RAgentic Bot"
    APP_VERSION: str = "0.1.0"
    APP_SUMMARY: str = "Multi-Agentic AI Assistant API"
    APP_DESCRIPTION: str = (
        "A production-ready Agent API supporting multi-agent"
    )
    APP_URL: str = "http://localhost:8001"
    APP_DEBUG: bool = True

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Singleton instances — import these directly wherever needed
# ---------------------------------------------------------------------------

app_settings = ApplicationSettings()
server_settings = ServerSettings()
llm_settings = LLMSettings()
api_key_settings = APIKeySettings()
web_search_settings = WebSearchSettings()
agent_settings = AgentSettings()
rag_client_settings = RAGClientSettings()
