#!/bin/bash

echo "🚀 Starting Parallel Base Models Training Launcher..."
python3 orchestrator.py

echo "✅ Training Completed."

echo "🧪 Running Base Models Test Suites..."
python3 test_suite.py