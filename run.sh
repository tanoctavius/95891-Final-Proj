#!/bin/bash
# Quick start script for the Multi-Modal Attribute Extraction & Retrieval App

echo "🚀 Starting Multi-Modal Attribute Extraction & Retrieval App..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "📦 Installing dependencies..."
    source venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "✅ Virtual environment found."
fi

# Activate virtual environment and run
echo "🚀 Activating virtual environment and starting Streamlit..."
echo ""

source venv/bin/activate
streamlit run app.py

