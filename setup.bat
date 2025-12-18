@echo off
REM Setup script for RAG Knowledge Assistant (Windows)

echo Setting up RAG Knowledge Assistant...

REM Check Python version
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
cd backend
pip install --upgrade pip
pip install -r requirements.txt
cd ..

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
pip install -r requirements.txt
cd ..

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy env.example .env
    echo WARNING: Please edit .env file and add your GEMINI_API_KEY
) else (
    echo .env file already exists
)

REM Create necessary directories
echo Creating data directories...
if not exist data mkdir data
if not exist uploads mkdir uploads
if not exist logs mkdir logs

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file and add your GEMINI_API_KEY
echo    Get it from: https://makersuite.google.com/app/apikey
echo 2. Run run.bat to start the application
echo.
pause

