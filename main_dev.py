import logging
import uvicorn
from app.config.app_settings import app_settings, server_settings

if __name__ == "__main__":
    _log = logging.getLogger(__name__)
    _log.info("Starting %s v%s ...", app_settings.APP_NAME, app_settings.APP_VERSION)
    uvicorn.run(
        "app.main:app",
        host=server_settings.HOST,
        port=server_settings.PORT,
        reload=server_settings.RELOAD,
        env_file=".env"
    )
