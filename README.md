# 🤖 Agent Service — Super-RAGentic Bot

> **Multi-Agent AI Assistant API** — FastAPI microservice powering the orchestration layer of the Super-RAGentic system.

[![M8ven Score](https://m8ven.ai/badge/mcp/hoanganhxtah-ragentic-core-ylqa12?v=9d2a4a64b9593682a38262f919d818c9)](https://m8ven.ai/mcp/hoanganhxtah-ragentic-core-ylqa12?s=readme)
[![M8ven Score](https://m8ven.ai/badge/mcp/hoanganhxtah/ragentic-core)](https://m8ven.ai/mcp/hoanganhxtah/ragentic-core?s=readme)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc Multi-Agent](#-kiến-trúc-multi-agent)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Cấu hình biến môi trường](#-cấu-hình-biến-môi-trường)
- [API Endpoints](#-api-endpoints)
- [LLM Providers được hỗ trợ](#-llm-providers-được-hỗ-trợ)
- [Triển khai với Docker](#-triển-khai-với-docker)
- [Tích hợp với các service khác](#-tích-hợp-với-các-service-khác)
- [Tác giả](#-tác-giả)

---

## 🔍 Tổng quan

`agent_service` là microservice trung tâm của hệ thống **Super-RAGentic Bot**, chịu trách nhiệm:

- **Điều phối multi-agent** qua mô hình Supervisor sử dụng LangGraph
- **Kết nối nhiều LLM providers** (Gemini, OpenAI, Anthropic, Groq, AWS Bedrock)
- **Streaming real-time** qua SSE (Server-Sent Events)
- **Lưu trữ lịch sử hội thoại** vào PostgreSQL
- **Tích hợp RAG** (Retrieval-Augmented Generation) qua HTTP client tới `rag_service`
- **Tích hợp MCP Server** cho banking tools
- **Tìm kiếm web** qua Tavily API

> **Lưu ý:** `agent_service` **không** trực tiếp mở vector store hay load embedding model. Toàn bộ retrieval và ingestion được ủy quyền cho `rag_service` (microservice riêng).

---

## 🏗 Kiến trúc Multi-Agent

Hệ thống sử dụng mô hình **Supervisor pattern** với LangGraph:

```
                        ┌─────────────┐
                        │    START    │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                   ┌────┤  Supervisor  ├────┐
                   │    └──────┬──────┘    │
                   │           │           │
          handoff_to_     no tool call   handoff_to_
          rag_agent      (final answer)  web_agent
                   │           │           │
            ┌──────▼──────┐    │    ┌──────▼──────┐
            │  RAG Agent  │    │    │  Web Agent  │
            └──────┬──────┘    │    └──────┬──────┘
                   │           │           │
                   └───────────┤───────────┘
                               │
                        ┌──────▼──────┐
                        │  Validator  │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │     END     │
                        └─────────────┘
```

### Agents

| Agent | Vai trò | Tools |
|-------|---------|-------|
| **Supervisor** | Điều phối, quyết định route, tổng hợp câu trả lời cuối | `handoff_to_rag_agent`, `handoff_to_web_agent` |
| **RAG Agent** | Tra cứu knowledge base nội bộ (qua `rag_service`) | `search_vectordb`, MCP banking tools |
| **Web Agent** | Tìm kiếm thông tin trên web | `web_search` (Tavily) |
| **Validator** | Đánh giá chất lượng câu trả lời (PASS / FAIL / AMBIGUOUS) | — |

### Routing Logic

- `PASS` → Trả kết quả cho người dùng
- `FAIL` / `AMBIGUOUS` + còn retry budget → Retry (gọi lại sub-agent)
- `FAIL` / `AMBIGUOUS` + hết retry budget → Tổng hợp best-effort answer
- Max hops = **6** lần bàn giao trước khi buộc kết thúc

---

## 📁 Cấu trúc thư mục

```
agent_service/
├── app/
│   ├── main.py                         # FastAPI application + lifespan
│   ├── requirements.txt                # Dependencies (production)
│   │
│   ├── agent/                          # 🤖 Multi-agent orchestration
│   │   ├── engine.py                   # ReActAgentEngine — run() & stream()
│   │   ├── graph.py                    # LangGraph supervisor graph
│   │   ├── state.py                    # AgentState definition
│   │   │
│   │   ├── agents/                     # Agent factories
│   │   │   ├── supervisor.py           # Supervisor agent
│   │   │   ├── rag_agent.py            # RAG agent (knowledge base)
│   │   │   ├── web_agent.py            # Web search agent
│   │   │   └── validator.py            # Answer quality validator
│   │   │
│   │   ├── tools/                      # Agent tools
│   │   │   ├── registry.py             # Tool registry & builder
│   │   │   ├── vectordb_tool.py        # Vector DB search tool
│   │   │   ├── web_search_tool.py      # Tavily web search tool
│   │   │   ├── banking_mcp_client.py   # MCP client for banking tools
│   │   │   └── handoff.py              # Agent handoff tools
│   │   │
│   │   └── prompts/                    # System prompts
│   │       ├── __init__.py             # Prompt exports
│   │       ├── chat.py                 # Chat prompt
│   │       ├── supervisor.py           # Supervisor prompt
│   │       ├── rag_agent.py            # RAG agent prompt
│   │       ├── web_agent.py            # Web agent prompt
│   │       └── validator.py            # Validator prompt
│   │
│   ├── api/                            # 🌐 REST API layer
│   │   ├── agent_api.py                # /agent & /agent-stream endpoints
│   │   └── search_api.py               # /search/vector & /search/web
│   │
│   ├── config/                         # ⚙️ Configuration
│   │   ├── app_settings.py             # Singleton settings instances
│   │   └── env/                        # Pydantic settings classes
│   │       ├── server_config.py        # Server host/port/context path
│   │       ├── llm_config.py           # LLM provider & model settings
│   │       ├── agent_config.py         # Agent behavior settings
│   │       ├── web_search_config.py    # Tavily & proxy settings
│   │       ├── postgres_config.py      # PostgreSQL connection pool
│   │       └── rag_client_config.py    # RAG service HTTP client config
│   │
│   ├── llm/                            # 🧠 LLM abstraction
│   │   └── llm_factory.py              # Multi-provider chat model factory
│   │
│   ├── entity/                         # 📦 ORM models
│   │   └── postgres_models.py          # SQLAlchemy: chat_messages table
│   │
│   ├── repository/                     # 💾 Data access layer
│   │   └── chat_history_repository.py  # Chat history CRUD
│   │
│   ├── retrieval/                      # 🔎 Retrieval abstraction
│   │   ├── __init__.py                 # build_retriever() factory
│   │   ├── base.py                     # Retriever interface
│   │   ├── factory.py                  # Retriever factory
│   │   └── remote.py                   # RemoteRetriever (HTTP → rag_service)
│   │
│   └── schemas/                        # 📐 Pydantic request/response models
│       ├── agent.py                    # AgentRequest, AgentResponse, AgentStep
│       └── chat.py                     # SearchRequest, SearchResponse
│
├── main_dev.py                         # Dev entrypoint (uvicorn + hot reload)
├── Dockerfile                          # Production Docker image
├── docker-compose.yml                  # Docker Compose config
├── requirements.txt                    # Meta requirements (all sub-services)
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
└── .dockerignore                       # Docker ignore rules
```

---

## 💻 Yêu cầu hệ thống

- **Python** ≥ 3.12
- **PostgreSQL** (tùy chọn — cho lưu trữ lịch sử hội thoại)
- **rag_service** đang chạy (tùy chọn — agent sẽ fallback nếu không có)
- **MCP Server** (tùy chọn — cho banking tools)

---

## 🚀 Cài đặt & Chạy

### 1. Clone repository

```bash
git clone https://github.com/hoanganhxtah/ragentic-core.git
cd ragentic-core
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r app/requirements.txt
```

### 4. Cấu hình biến môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env với API keys và cấu hình phù hợp
```

### 5. Chạy development server

```bash
python main_dev.py
```

Server sẽ khởi động tại `http://localhost:8001` với hot reload enabled.

### 6. Truy cập API docs

- **Swagger UI:** http://localhost:8001/api/docs
- **ReDoc:** http://localhost:8001/api/redoc
- **Health Check:** http://localhost:8001/health

---

## ⚙️ Cấu hình biến môi trường

Tham khảo file [`.env.example`](.env.example) để biết đầy đủ các biến. Dưới đây là các nhóm chính:

### LLM Provider

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `LLM_PROVIDER` | Provider: `gemini` \| `openai` \| `anthropic` \| `openai-compatible` \| `bedrock` \| `groq` | `gemini` |
| `LLM_MODEL` | Tên model | `gemini-3.1-flash-lite-preview` |
| `LLM_TEMPERATURE` | Nhiệt độ sampling | `0.2` |
| `LLM_MAX_TOKENS` | Số token tối đa | `1024` |
| `LLM_API_KEY` | API key chung (hoặc dùng key riêng theo provider) | — |

### Server

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `SERVER_HOST` | Bind host | `0.0.0.0` |
| `SERVER_PORT` | Bind port | `8001` |
| `SERVER_CONTEXT_PATH` | API prefix | `/api` |

### PostgreSQL (Conversation Memory)

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `DB_POSTGRES_URL` | Connection URL (nếu set thì ưu tiên dùng) | — |
| `DB_POSTGRES_HOST` | Hostname | `localhost` |
| `DB_POSTGRES_PORT` | Port | `5432` |
| `DB_POSTGRES_DATABASE` | Database name | `postgres` |

### RAG Service Client

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `RAG_CLIENT_SERVICE_URL` | URL tới rag_service | `http://host.docker.internal:8002` |
| `RAG_CLIENT_CONTEXT_PATH` | API prefix | `/api/v1` |
| `RAG_CLIENT_TIMEOUT` | Request timeout (seconds) | `10.0` |

### Web Search

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `WEB_SEARCH_PROVIDER` | Search provider | `tavily` |
| `TAVILY_API_KEY` | Tavily API key | — |
| `WEB_SEARCH_MAX_RESULTS` | Số kết quả tối đa | `5` |

### MCP Server

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `MCP_SERVER_URL` | URL tới Banking MCP Server | `http://host.docker.internal:8003/mcp` |

---

## 🌐 API Endpoints

### `GET /health`

Health check endpoint.

```json
{ "status": "UP" }
```

### `POST /agent`

Multi-agent endpoint — gửi câu hỏi và nhận phản hồi đầy đủ (JSON).

**Request:**

```json
{
  "question": "Lãi suất tiết kiệm 12 tháng là bao nhiêu?",
  "thread_id": "conv-001",
  "user_id": "user-123",
  "channel": "web",
  "channel_id": "ch-001",
  "session_id": "sess-abc",
  "include_steps": true
}
```

**Response:**

```json
{
  "user_id": "user-123",
  "channel": "web",
  "channel_id": "ch-001",
  "thread_id": "conv-001",
  "session_id": "sess-abc",
  "answer": "Lãi suất tiết kiệm 12 tháng hiện tại là ...",
  "steps": [
    {
      "agent": "rag_agent",
      "tool": "search_vectordb",
      "status": "tool_end",
      "content": "..."
    }
  ]
}
```

### `POST /agent-stream`

Multi-agent endpoint với **SSE streaming** — token được trả về ngay khi LLM sinh ra.

**SSE Events:**

```
data: {"type": "token", "content": "Lãi suất", "agent": "supervisor"}
data: {"type": "step", "status": "tool_start", "agent": "rag_agent", "tool": "search_vectordb", "content": "..."}
data: {"type": "step", "status": "tool_end", "agent": "rag_agent", "tool": "search_vectordb", "content": "..."}
data: {"type": "token", "content": " tiết kiệm", "agent": "supervisor"}
data: {"type": "done", "answer": "...", "time_response": 2.35}
```

### `POST /search/vector`

Truy vấn trực tiếp vector database (qua `rag_service`). Trả top-k kết quả thô, không áp score threshold.

### `POST /search/web`

Tìm kiếm web qua Tavily. Trả raw results.

---

## 🧠 LLM Providers được hỗ trợ

| Provider | Package | Alias |
|----------|---------|-------|
| **Google Gemini** | `langchain-google-genai` | `gemini`, `google` |
| **OpenAI** | `langchain-openai` | `openai`, `oai` |
| **OpenAI-Compatible** | `langchain-openai` | `openai-compatible`, `openai_compatible` |
| **Anthropic (Claude)** | `langchain-anthropic` | `anthropic`, `claude` |
| **Groq** | `langchain-groq` | `groq` |
| **AWS Bedrock** | `langchain-aws` | `bedrock`, `aws`, `aws-bedrock` |

Chuyển đổi provider chỉ cần thay đổi `LLM_PROVIDER` và `LLM_MODEL` trong `.env`. Các package không sử dụng sẽ không được import (lazy loading).

---

## 🐳 Triển khai với Docker

### Build & Run

```bash
docker compose up --build -d
```

### Chỉ build image

```bash
docker build -t agent-service:latest .
```

### Docker Compose config

- **Port:** `8001` (có thể override bằng `SERVER_PORT` trong `.env`)
- **Health check:** `GET http://localhost:8001/health` mỗi 15s
- **Restart policy:** `unless-stopped`
- **Security:** Chạy dưới non-root user `agent`

---

## 🔗 Tích hợp với các service khác

`agent_service` là một phần của hệ thống microservices **Super-RAGentic**:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│     UI      │────▶│  Agent Service   │────▶│ RAG Service │
│  (port 3000)│     │   (port 8001)    │     │ (port 8002) │
└─────────────┘     └────────┬─────────┘     └─────────────┘
                             │
                    ┌────────▼─────────┐
                    │   MCP Server     │
                    │   (port 8003)    │
                    └──────────────────┘
```

| Service | Port | Vai trò |
|---------|------|---------|
| **agent_service** | 8001 | Điều phối agent, API chính |
| **rag_service** | 8002 | Vector store, embedding, ingestion |
| **mcp_server** | 8003 | Banking tools (MCP protocol) |
| **UI** | 3000 | Giao diện người dùng |

---

## 👤 Tác giả

**Nguyen Hoang Anh**

- GitHub: [@hoanganhxtah](https://github.com/hoanganhxtah)
- Email: nguyenhoanganh2820@gmail.com

---

## 📄 License

[Apache 2.0](https://github.com/anhnh2292/RAG_Project/blob/main/LICENSE)
