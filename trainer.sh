#!/bin/bash

echo "🚀 Starting Parallel Training Launcher..."
python3 parallel_train_launcher.py

echo "✅ Training Completed."

echo "🧪 Running All Test Suites..."
python3 run_all_tests.py

echo "🎉 All processes completed successfully."