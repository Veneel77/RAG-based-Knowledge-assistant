# 🤖 RAG Knowledge Assistant - Production Grade

A **Production-Ready Retrieval-Augmented Generation (RAG) Knowledge Assistant** that enables intelligent Q&A over your documents using **free/open-source tools** and **Google Gemini API**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

### 🔐 **Authentication & Security**
- JWT-based authentication
- User-specific document storage
- Secure password hashing with bcrypt
- Token-based session management

### 📄 **Document Management**
- Upload **PDF, TXT, DOCX** files
- Automatic text extraction
- Intelligent text chunking with overlap
- Per-user document isolation

### 🧠 **RAG Pipeline**
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector Store**: FAISS for fast similarity search
- **Generation**: Google Gemini API (free tier)
- **Retrieval**: Top-K semantic search with source citations

### 💬 **Chat Interface**
- Clean, modern Streamlit UI
- Real-time Q&A with document context
- Source attribution with similarity scores
- Conversation history tracking

### 🐳 **Production Ready**
- Docker & Docker Compose support
- Centralized logging with rotation
- Environment-based configuration
- Comprehensive error handling
- Database migrations ready

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  Streamlit (Port 8501)
│   (Streamlit)   │  - Auth UI
└────────┬────────┘  - Upload UI
         │           - Chat Interface
         │ REST API
         ▼
┌─────────────────┐
│   Backend       │  FastAPI (Port 8000)
│   (FastAPI)     │  - JWT Auth
└────────┬────────┘  - Document Processing
         │           - RAG Pipeline
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
┌─────────┐         ┌──────────┐
│ SQLite  │         │  FAISS   │
│   DB    │         │  Index   │
└─────────┘         └──────────┘
    │                     │
    ▼                     ▼
 Metadata           Vector Store
 (Users,            (Embeddings)
  Docs,
  Chunks,
  History)
```

---

## 📁 Project Structure

```
RAG-based-Knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── config.py            # Configuration management
│   │   ├── auth/
│   │   │   ├── auth.py          # Authentication logic
│   │   │   └── jwt.py           # JWT token handling
│   │   ├── rag/
│   │   │   ├── loader.py        # Document loading (PDF/TXT/DOCX)
│   │   │   ├── chunker.py       # Text chunking
│   │   │   ├── embedder.py      # Embedding generation
│   │   │   ├── vector_store.py  # FAISS vector store
│   │   │   ├── retriever.py     # Retrieval orchestration
│   │   │   └── generator.py     # Gemini response generation
│   │   ├── api/
│   │   │   ├── auth_api.py      # Auth endpoints
│   │   │   ├── upload.py        # Upload endpoints
│   │   │   ├── query.py         # Query endpoints
│   │   │   └── history.py       # History endpoints
│   │   ├── models/
│   │   │   └── database.py      # SQLAlchemy models
│   │   └── utils/
│   │       └── logger.py        # Centralized logging
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/                        # SQLite DB & FAISS index
├── uploads/                     # Uploaded documents
├── logs/                        # Application logs
├── docker-compose.yml           # Docker orchestration
├── env.example                  # Environment template
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** or **conda**
- **Git**
- **Google Gemini API Key** (Free: https://makersuite.google.com/app/apikey)

### Option 1: Local Setup (Recommended for Development)

#### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/RAG-based-Knowledge-assistant.git
cd RAG-based-Knowledge-assistant
```

#### 2️⃣ Set Up Environment Variables

```bash
# Copy example env file
cp env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your-actual-api-key-here
```

**Get your FREE Gemini API Key:**
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and paste into `.env` file

#### 3️⃣ Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### 4️⃣ Install Frontend Dependencies

```bash
cd ../frontend
pip install -r requirements.txt
cd ..
```

#### 5️⃣ Run Backend

```bash
# From project root
cd backend
python -m backend.app.main
```

Backend will start at: **http://localhost:8000**
- API Docs: **http://localhost:8000/docs**

#### 6️⃣ Run Frontend (New Terminal)

```bash
# From project root
cd frontend
streamlit run app.py
```

Frontend will start at: **http://localhost:8501**

---

### Option 2: Docker Setup (Recommended for Production)

#### 1️⃣ Prerequisites

- **Docker Desktop** installed
- **Docker Compose** installed

#### 2️⃣ Setup Environment

```bash
# Copy and edit .env file
cp env.example .env
# Add your GEMINI_API_KEY in .env
```

#### 3️⃣ Build and Run

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

#### 4️⃣ Access Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

#### 5️⃣ Stop Services

```bash
docker-compose down

# Remove volumes as well
docker-compose down -v
```

---

## 📖 Usage Guide

### 1. **Create Account**
- Open http://localhost:8501
- Go to **Register** tab
- Enter username, email, and password
- Click **Register**

### 2. **Upload Documents**
- After login, use the sidebar
- Click **Browse files** under "Upload Documents"
- Select PDF, TXT, or DOCX files (max 10MB)
- Click **Upload**
- Wait for processing (chunking + embedding)

### 3. **Ask Questions**
- Type your question in the chat input
- Click **Ask** or press Enter
- View the AI-generated answer
- Expand **View Sources** to see citations with similarity scores

### 4. **Manage Documents**
- View all uploaded documents in sidebar
- Click on a document to see details
- Delete documents as needed

---

## 🔧 Configuration

All configuration is managed through environment variables (`.env` file):

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (REQUIRED) | - |
| `SECRET_KEY` | JWT secret (change in production!) | auto-generated |
| `CHUNK_SIZE` | Text chunk size in characters | 500 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `TOP_K_RESULTS` | Number of chunks to retrieve | 5 |
| `MAX_FILE_SIZE` | Max upload size in bytes | 10485760 (10MB) |
| `EMBEDDING_MODEL` | HuggingFace model for embeddings | all-MiniLM-L6-v2 |

See `env.example` for full list of options.

---

## 🧪 Testing

### Run Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Test API Endpoints Manually

```bash
# Health check
curl http://localhost:8000/health

# Register user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"password123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"password123"}'
```

---

## 🎯 How RAG Works in This Project

### End-to-End Flow

```
1. DOCUMENT INGESTION
   └─> Upload PDF/TXT/DOCX
   └─> Extract text (PyPDF2, python-docx)
   └─> Split into chunks (500 chars, 50 overlap)
   └─> Generate embeddings (sentence-transformers)
   └─> Store in FAISS index
   └─> Save metadata in SQLite

2. QUERY PROCESSING
   └─> User asks question
   └─> Generate query embedding
   └─> Search FAISS for top-K similar chunks
   └─> Retrieve chunk content + metadata

3. RESPONSE GENERATION
   └─> Build context from retrieved chunks
   └─> Create prompt with context + query
   └─> Send to Gemini API
   └─> Return answer + source citations
   └─> Save to history

4. DISPLAY
   └─> Show answer in chat UI
   └─> Display source documents
   └─> Show similarity scores
```

### Why This Stack?

| Component | Why? |
|-----------|------|
| **FastAPI** | Async, fast, auto-docs, modern Python |
| **Streamlit** | Quick beautiful UI without frontend code |
| **FAISS** | Fast similarity search, works offline, free |
| **sentence-transformers** | Best open-source embeddings, no API costs |
| **Gemini API** | Free tier, powerful, Google-backed |
| **SQLite** | Zero-config, file-based, perfect for MVP |
| **JWT** | Stateless auth, scalable, industry standard |

---

## 📊 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login user |
| GET | `/auth/me` | Get current user profile |

### Document Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload/` | Upload and process document |
| GET | `/upload/documents` | List user's documents |
| DELETE | `/upload/documents/{id}` | Delete document |

### Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query/` | Ask question and get answer |
| GET | `/query/health` | Check query service health |

### History

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/history/` | Get query history |
| DELETE | `/history/{id}` | Delete history item |
| DELETE | `/history/` | Clear all history |

**Full API documentation**: http://localhost:8000/docs

---

## 🚨 Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is in use
# Windows
netstat -ano | findstr :8000

# Kill process if needed (replace PID)
taskkill /PID <PID> /F

# Or use different port in .env
API_PORT=8001
```

### "GEMINI_API_KEY not found" error

1. Ensure `.env` file exists in project root
2. Check `GEMINI_API_KEY=...` is set (no quotes, no spaces)
3. Restart backend after changing `.env`

### Embeddings download slow/failing

```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### SQLite database locked

```bash
# Delete and recreate database
rm data/knowledge_assistant.db
# Restart backend to recreate
```

### Docker build fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

---

## 🔒 Security Considerations

### For Production Deployment:

1. **Change SECRET_KEY**: Generate a strong random key
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Use HTTPS**: Deploy behind reverse proxy (nginx, Caddy)

3. **Rate Limiting**: Add rate limiting middleware

4. **CORS**: Configure proper CORS origins in `main.py`

5. **Database**: Migrate from SQLite to PostgreSQL for production

6. **Environment Variables**: Use secrets management (AWS Secrets Manager, Azure Key Vault)

7. **API Keys**: Never commit `.env` to git (already in `.gitignore`)

---

## 🎓 Interview Talking Points

### Architecture Decisions

- **Why FastAPI?** Async support, automatic OpenAPI docs, type hints, high performance
- **Why FAISS?** Production-grade, used by Facebook, handles millions of vectors
- **Why Sentence Transformers?** SOTA embeddings, open-source, no API costs
- **Why Gemini?** Free tier generous, Google-backed reliability

### Scalability

- **Current**: Single server, SQLite, in-memory FAISS
- **Scale to 10K users**: 
  - PostgreSQL for database
  - Redis for caching
  - Separate vector service
  - Load balancer for API
- **Scale to 1M users**:
  - Kubernetes orchestration
  - Managed vector DB (Pinecone, Weaviate)
  - CDN for frontend
  - Distributed caching

### Trade-offs

| Decision | Pro | Con |
|----------|-----|-----|
| SQLite | Zero config, fast for small data | Not suitable for concurrent writes |
| FAISS | Fast, free, offline | Requires rebuilding for deletions |
| Gemini Free Tier | No cost, powerful | Rate limits, requires internet |
| Streamlit | Quick to build | Less customizable than React |

---

## 🛣️ Roadmap

- [ ] Add support for more file types (CSV, JSON, HTML)
- [ ] Implement conversation memory (multi-turn)
- [ ] Add document summarization
- [ ] Support for multiple languages
- [ ] Advanced filters (by date, document type)
- [ ] Analytics dashboard
- [ ] Export chat history
- [ ] Multi-modal support (images in PDFs)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **Streamlit** - Instant UI framework
- **FAISS** - Vector similarity search (Meta AI)
- **Sentence Transformers** - Embedding models (UKP Lab)
- **Google Gemini** - LLM API
- **Hugging Face** - Model hosting

---

## 📄 Additional Documentation

- **[DEPLOYMENT_PROOF.md](DEPLOYMENT_PROOF.md)** - Complete proof of working deployment with test results
- **[WORKING_CONFIGURATION.md](WORKING_CONFIGURATION.md)** - Verified working configuration and setup guide

---

## ✅ Verified Working

**Status**: 🟢 **PRODUCTION READY**  
**Last Verified**: December 18, 2025  
**Gemini Model**: `gemini-2.5-flash` (Latest stable)

This project has been fully tested and verified working. See [DEPLOYMENT_PROOF.md](DEPLOYMENT_PROOF.md) for complete test results.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Veneel77/RAG-based-Knowledge-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Veneel77/RAG-based-Knowledge-assistant/discussions)

---

## 🌟 Show Your Support

If you find this project helpful, please give it a ⭐️ on GitHub!

---

**Built with ❤️ for the AI Community**

*Production-ready • Interview-ready • Free & Open Source*
