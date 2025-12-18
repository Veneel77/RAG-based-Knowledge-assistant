# 🤖 RAG Knowledge Assistant

A **production-grade** Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over your documents using **Google Gemini AI** and **FAISS vector search**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌟 Features

### Core Functionality
- 📄 **Multi-format Document Upload** - PDF, TXT, DOCX support
- 🔍 **Semantic Search** - FAISS-powered vector similarity search
- 🤖 **AI-Powered Responses** - Google Gemini API integration
- 📚 **Source Citations** - Transparent answers with document references
- 💬 **Chat Interface** - Intuitive Streamlit-based UI
- 📊 **Conversation History** - Track all your queries and responses

### Production Features
- 🔐 **JWT Authentication** - Secure user management
- 🗄️ **SQLite Database** - User data and document metadata storage
- 🐳 **Docker Support** - Easy deployment with Docker Compose
- 📝 **Comprehensive Logging** - Structured logging for debugging
- ⚙️ **Configuration Management** - Environment-based settings
- 🧪 **Modular Architecture** - Clean, scalable code structure

---

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Streamlit  │ ◄─────► │   FastAPI    │ ◄─────► │   Gemini    │
│   Frontend  │         │   Backend    │         │     API     │
└─────────────┘         └──────────────┘         └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    SQLite    │
                        │   Database   │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    FAISS     │
                        │ Vector Store │
                        └──────────────┘
```

### RAG Pipeline Flow

1. **Document Ingestion**
   - Upload PDF/TXT/DOCX files
   - Extract text content
   - Split into overlapping chunks (configurable)
   - Generate embeddings using sentence-transformers
   - Store vectors in FAISS index
   - Save metadata in SQLite

2. **Query Processing**
   - Convert user query to embedding
   - Perform similarity search in FAISS
   - Retrieve top-k relevant chunks
   - Construct context-aware prompt
   - Generate response using Gemini API
   - Return answer with source citations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key (free tier available)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/RAG-based-Knowledge-assistant.git
cd RAG-based-Knowledge-assistant
```

### 2. Get Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a free API key
3. Copy the key for next step

### 3. Setup Environment
```bash
# Copy example environment file
cp env.example .env

# Edit .env file and add your Gemini API key
# GEMINI_API_KEY=your-api-key-here
```

### 4. Install Dependencies

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
cd frontend
pip install -r requirements.txt
cd ..
```

#### Option B: Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### 5. Run the Application

#### Without Docker:

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
streamlit run app.py
```

#### With Docker:
```bash
docker-compose up
```

### 6. Access the Application
- **Frontend:** http://localhost:8501
- **Backend API Docs:** http://localhost:8000/docs
- **Backend Health:** http://localhost:8000/health

---

## 📖 Usage Guide

### First Time Setup
1. Open the frontend at http://localhost:8501
2. Click "Register" tab
3. Create an account (username, email, password)
4. Login with your credentials

### Upload Documents
1. In the sidebar, click "Choose a file"
2. Select PDF, TXT, or DOCX file
3. Click "Upload"
4. Wait for processing (you'll see chunk count)

### Ask Questions
1. Type your question in the input box
2. Click "Ask" or press Enter
3. View the AI-generated answer
4. Expand "View Sources" to see document references

### Manage Documents
- View all uploaded documents in the sidebar
- Click on a document to see details
- Delete documents you no longer need

---

## 🛠️ Configuration

All configuration is managed through environment variables in the `.env` file:

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `SECRET_KEY` | JWT token secret | Change in production |
| `CHUNK_SIZE` | Text chunk size | 500 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `TOP_K_RESULTS` | Number of search results | 5 |
| `MAX_FILE_SIZE` | Max upload size (bytes) | 10MB |

See `env.example` for all available options.

---

## 📁 Project Structure

```
RAG-based-Knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration management
│   │   ├── auth/                # Authentication module
│   │   │   ├── auth.py          # Auth dependencies
│   │   │   └── jwt.py           # JWT token handling
│   │   ├── rag/                 # RAG pipeline
│   │   │   ├── loader.py        # Document loading
│   │   │   ├── chunker.py       # Text chunking
│   │   │   ├── embedder.py      # Embedding generation
│   │   │   ├── vector_store.py  # FAISS management
│   │   │   ├── retriever.py     # Similarity search
│   │   │   └── generator.py     # Gemini integration
│   │   ├── api/                 # API endpoints
│   │   │   ├── auth_api.py      # Auth endpoints
│   │   │   ├── upload.py        # Upload endpoints
│   │   │   ├── query.py         # Query endpoints
│   │   │   └── history.py       # History endpoints
│   │   ├── models/              # Database models
│   │   │   └── database.py      # SQLAlchemy models
│   │   └── utils/               # Utilities
│   │       └── logger.py        # Logging setup
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Backend Docker image
├── frontend/
│   ├── app.py                   # Streamlit application
│   ├── requirements.txt         # Frontend dependencies
│   └── Dockerfile               # Frontend Docker image
├── data/                        # Database and vector store
├── uploads/                     # Uploaded documents
├── logs/                        # Application logs
├── docker-compose.yml           # Docker Compose config
├── env.example                  # Example environment file
└── README_NEW.md                # This file
```

---

## 🔒 Security

### Best Practices Implemented
- ✅ JWT-based authentication
- ✅ Password hashing with bcrypt
- ✅ Environment-based secrets
- ✅ File type validation
- ✅ File size limits
- ✅ User-specific data isolation

### Production Recommendations
1. Change `SECRET_KEY` to a strong random value
2. Use HTTPS in production
3. Implement rate limiting
4. Add input sanitization
5. Enable CORS only for trusted origins
6. Regular security audits

---

## 🧪 Testing

### Run Tests
```bash
cd backend
pytest tests/
```

### Manual Testing
1. Use FastAPI interactive docs at http://localhost:8000/docs
2. Test all endpoints with different payloads
3. Verify error handling

---

## 🐳 Docker Deployment

### Build and Run
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables
Edit `docker-compose.yml` or create `.env` file for Docker environment variables.

---

## 📊 API Documentation

### Authentication Endpoints
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `GET /auth/me` - Get current user profile

### Document Endpoints
- `POST /upload/` - Upload document
- `GET /upload/documents` - List user's documents
- `DELETE /upload/documents/{id}` - Delete document

### Query Endpoints
- `POST /query/` - Query documents
- `GET /query/health` - Check query service health

### History Endpoints
- `GET /history/` - Get query history
- `DELETE /history/{id}` - Delete history item
- `DELETE /history/` - Clear all history

Full API documentation available at http://localhost:8000/docs

---

## 🎯 How It Works

### RAG (Retrieval-Augmented Generation)

RAG combines information retrieval with text generation:

1. **Retrieval Phase**
   - Your query is converted to a vector embedding
   - FAISS finds the most similar document chunks
   - Relevant context is extracted

2. **Augmentation Phase**
   - Retrieved chunks are formatted as context
   - A prompt is constructed with context + query
   - Conversation history is included if available

3. **Generation Phase**
   - Gemini API receives the augmented prompt
   - AI generates a response based on context
   - Answer includes source citations

### Why RAG?
- ✅ Provides answers grounded in your documents
- ✅ Reduces AI hallucinations
- ✅ Allows citing sources
- ✅ Works with private/proprietary data
- ✅ No need to fine-tune models

---

## 🔧 Troubleshooting

### Backend won't start
- Check if port 8000 is available
- Verify `GEMINI_API_KEY` in `.env`
- Check logs in `logs/` directory

### Frontend won't start
- Check if port 8501 is available
- Ensure backend is running
- Verify API_BASE_URL in frontend

### Document upload fails
- Check file size (default max: 10MB)
- Verify file format (PDF, TXT, DOCX only)
- Check disk space for uploads

### Query returns no results
- Ensure documents are uploaded and processed
- Check if vector store was created
- Verify Gemini API key is valid

### Docker issues
- Run `docker-compose down -v` to clean volumes
- Rebuild with `docker-compose build --no-cache`
- Check Docker logs: `docker-compose logs`

---

## 🚧 Limitations & Future Improvements

### Current Limitations
- SQLite (not suitable for high concurrency)
- No real-time collaboration
- Basic error recovery
- Single language support

### Planned Improvements
- [ ] PostgreSQL support
- [ ] Redis caching
- [ ] Multi-language support
- [ ] Advanced chunking strategies
- [ ] Hybrid search (keyword + semantic)
- [ ] PDF OCR support
- [ ] Role-based access control
- [ ] API rate limiting
- [ ] Monitoring dashboard
- [ ] Kubernetes deployment

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern Python web framework
- **Streamlit** - Rapid UI development
- **Google Gemini** - Powerful AI generation
- **FAISS** - Efficient vector search
- **Sentence Transformers** - State-of-the-art embeddings

---

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/RAG-based-Knowledge-assistant/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/RAG-based-Knowledge-assistant/discussions)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star on GitHub!

---

**Made with ❤️ for the AI community**

