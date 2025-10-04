@echo off
REM Text Summarizer API Startup Script for Windows
REM This script ensures the server starts with the correct configuration

echo 🚀 Starting Text Summarizer API Server...

REM Check if .env file exists, if not create it with defaults
if not exist .env (
    echo 📝 Creating .env file with default configuration...
    (
        echo # Text Summarizer API Configuration
        echo OLLAMA_HOST=http://127.0.0.1:11434
        echo OLLAMA_MODEL=llama3.2:latest
        echo OLLAMA_TIMEOUT=30
        echo SERVER_HOST=0.0.0.0
        echo SERVER_PORT=8000
        echo LOG_LEVEL=INFO
        echo API_KEY_ENABLED=false
        echo RATE_LIMIT_ENABLED=false
        echo RATE_LIMIT_REQUESTS=60
        echo RATE_LIMIT_WINDOW=60
        echo MAX_TEXT_LENGTH=32000
        echo MAX_TOKENS_DEFAULT=256
    ) > .env
    echo ✅ .env file created with default values
)

REM Check if Ollama is running
echo 🔍 Checking Ollama service...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama is running and accessible
) else (
    echo ❌ Ollama is not running or not accessible at http://127.0.0.1:11434
    echo    Please start Ollama first:
    echo    - Download and install Ollama from https://ollama.ai
    echo    - Start the Ollama application
    pause
    exit /b 1
)

REM Start the server
echo 🌟 Starting FastAPI server...
echo    Server will be available at: http://localhost:8000
echo    API docs will be available at: http://localhost:8000/docs
echo    Press Ctrl+C to stop the server
echo.

REM Load environment variables and start uvicorn
for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
)

uvicorn app.main:app --host %SERVER_HOST% --port %SERVER_PORT% --reload
