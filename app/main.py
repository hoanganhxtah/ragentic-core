import logging

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    logging.getLogger(__name__).warning(
        "truststore not installed; TLS may fail behind an inspecting proxy."
    )

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.search_api import router
import app.api.search_api as routes_module
import app.api.agent_api as agent_routes
from app.agent.engine import ReActAgentEngine
from app.retrieval import build_retriever
from app.config.app_settings import app_settings, server_settings, llm_settings
from app.config.env.postgres_config import init_postgres_pool, close_postgres_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Starting up...")

    # ── Retriever: HTTP client tới rag_service ──────────────────────────────
    # app/ không còn mở vector store nào. rag_service sở hữu store + ingestion;
    # ingest dữ liệu bằng POST {RAG_SERVICE_URL}/api/v1/ingest, không phải ở đây.
    retriever = build_retriever()
    routes_module.retriever = retriever

    stats = retriever.stats()
    if "error" in stats:
        _log.warning(
            "rag_service chưa truy cập được (%s). Agent vẫn khởi động; các lượt "
            "tra cứu knowledge base sẽ trả về rỗng cho tới khi service sẵn sàng.",
            stats["error"],
        )
    else:
        _log.info(
            "rag_service ready (docs=%s | collection=%s | embedding=%s)",
            stats.get("doc_count"),
            stats.get("collection"),
            stats.get("embedding_model"),
        )

    # ── PostgreSQL pool + tạo bảng từ ORM model ─────────────────────────
    try:
        await init_postgres_pool()
        # Tạo bảng chat_messages nếu chưa có (đọc từ ORM entity, không raw SQL)
        from sqlalchemy.ext.asyncio import create_async_engine as _create_engine
        from app.entity.postgres_models import Base
        from app.config.env.postgres_config import PostgresSettings
        _sa_engine = _create_engine(PostgresSettings().get_sqlalchemy_url())
        async with _sa_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _sa_engine.dispose()
        _log.info("Chat history storage ready (table: chat_messages)")
    except Exception as exc:
        _log.warning("PostgreSQL unavailable (%s) — chat history will not be saved", exc)

    # ── Agent engine (MemorySaver cho trong-phiên) ────────────────────────
    agent_engine = await ReActAgentEngine.create(retriever=retriever)
    agent_routes.agent_engine = agent_engine
    _log.info("Agent engine ready (provider=%s)", llm_settings.PROVIDER)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    await close_postgres_pool()
    # RemoteRetriever giữ một httpx.Client (connection pool) cần đóng lại.
    close = getattr(retriever, "close", None)
    if callable(close):
        close()
    _log.info("Shutting down...")


_log.info("Initializing %s v%s", app_settings.APP_NAME, app_settings.APP_VERSION)

app = FastAPI(
    title=app_settings.APP_NAME,
    summary=app_settings.APP_SUMMARY,
    description=app_settings.APP_DESCRIPTION,
    version=app_settings.APP_VERSION,
    docs_url=f"{server_settings.CONTEXT_PATH}/docs",
    redoc_url=f"{server_settings.CONTEXT_PATH}/redoc",
    openapi_url=f"{server_settings.CONTEXT_PATH}/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Nguyen Hoang Anh",
        "url": "https://github.com/anhnh2292",
        "email": "nguyenhoanganh2820@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://github.com/anhnh2292/RAG_Project/blob/main/LICENSE",
    },
    terms_of_service="https://example.com/terms/",
)

app.include_router(router)
app.include_router(agent_routes.router)


@app.get("/health")
def health_check():
    return {"status": "UP"}
