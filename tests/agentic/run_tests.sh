#!/bin/bash
# Run all SDK tests

echo "🧪 Running Disseqt Agentic SDK Tests"
echo "====================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ensure package and dev dependencies are installed
if ! python3 -c "import pytest_cov" 2>/dev/null; then
    echo "📦 Installing package and dev dependencies..."
    pip install -e ".[dev]" > /dev/null 2>&1
fi

# Ensure package is installed in editable mode (required for src layout)
if ! python3 -c "import disseqt_agentic_sdk" 2>/dev/null; then
    echo "📦 Installing package in editable mode..."
    pip install -e . > /dev/null 2>&1
fi

# Run pytest with coverage
pytest tests/ -v --cov=disseqt_agentic_sdk --cov-report=term-missing

echo ""
echo "✅ Tests complete!"
