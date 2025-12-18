@echo off
REM Run script for RAG Knowledge Assistant (Windows)

echo Starting RAG Knowledge Assistant...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start backend
echo Starting backend server...
start "RAG Backend" cmd /k "cd backend && python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a bit for backend to start
timeout /t 5 /nobreak

REM Start frontend
echo Starting frontend...
start "RAG Frontend" cmd /k "cd frontend && streamlit run app.py"

echo.
echo Application started!
echo.
echo Frontend: http://localhost:8501
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Close the terminal windows to stop the servers.
echo.
pause

