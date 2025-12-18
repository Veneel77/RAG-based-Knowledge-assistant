#!/bin/bash
# Setup script for RAG Knowledge Assistant

echo "🚀 Setting up RAG Knowledge Assistant..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
pip install -r requirements.txt
cd ..

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file and add your GEMINI_API_KEY"
else
    echo ".env file already exists"
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data uploads logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GEMINI_API_KEY"
echo "   Get it from: https://makersuite.google.com/app/apikey"
echo "2. Run ./run.sh to start the application"
echo ""

