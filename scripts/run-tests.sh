#!/bin/bash

# Comprehensive test runner for Text Summarizer API
# This script runs all tests and provides detailed reporting

set -e  # Exit on any error

echo "🧪 Text Summarizer API - Test Suite"
echo "===================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Check if pytest is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python to run tests."
    exit 1
fi

if ! python -c "import pytest" &> /dev/null; then
    echo "❌ pytest not found. Please install pytest to run tests."
    echo "   Run: pip install pytest pytest-asyncio"
    exit 1
fi

# Function to run tests with different configurations
run_test_suite() {
    local test_type="$1"
    local test_args="$2"
    local description="$3"
    
    echo "🔍 $description"
    echo "----------------------------------------"
    
    if python -m pytest $test_args; then
        echo "✅ $description - PASSED"
        echo ""
        return 0
    else
        echo "❌ $description - FAILED"
        echo ""
        return 1
    fi
}

# Track overall success
overall_success=true

# Run different test suites
echo "📊 Running comprehensive test suite..."
echo ""

# 1. Unit tests (fast)
if ! run_test_suite "unit" "tests/test_services.py tests/test_config.py tests/test_schemas.py tests/test_errors.py tests/test_logging.py tests/test_middleware.py" "Unit Tests"; then
    overall_success=false
fi

# 2. API tests
if ! run_test_suite "api" "tests/test_api.py tests/test_api_errors.py" "API Tests"; then
    overall_success=false
fi

# 3. Integration tests
if ! run_test_suite "integration" "tests/test_502_prevention.py" "502 Prevention Tests"; then
    overall_success=false
fi

# 4. Startup script tests
if ! run_test_suite "startup" "tests/test_startup_script.py" "Startup Script Tests"; then
    overall_success=false
fi

# 5. Main application tests
if ! run_test_suite "main" "tests/test_main.py" "Main Application Tests"; then
    overall_success=false
fi

# 6. All tests together (comprehensive)
echo "🔍 Running All Tests Together"
echo "----------------------------------------"
if python -m pytest tests/ -v --tb=short; then
    echo "✅ All Tests Together - PASSED"
    echo ""
else
    echo "❌ All Tests Together - FAILED"
    echo ""
    overall_success=false
fi

# Final report
echo "📋 Test Summary"
echo "==============="
if [ "$overall_success" = true ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "✅ Your code is ready for:"
    echo "   • Committing to git"
    echo "   • Deploying to production"
    echo "   • Code review"
    echo ""
    echo "🚀 Text Summarizer API is production-ready!"
    exit 0
else
    echo "❌ SOME TESTS FAILED!"
    echo ""
    echo "🔧 Please fix the failing tests before:"
    echo "   • Committing to git"
    echo "   • Deploying to production"
    echo "   • Code review"
    echo ""
    echo "💡 Run individual test files to debug:"
    echo "   python -m pytest tests/test_services.py -v"
    echo "   python -m pytest tests/test_api.py -v"
    echo "   python -m pytest tests/test_502_prevention.py -v"
    echo ""
    exit 1
fi