#!/bin/bash
# Quick demo script for presentation

echo "🎯 Starting Voice Phishing Detection Demo..."

# Check if server is running
if ! pgrep -f "python server.py" > /dev/null; then
    echo "📡 Starting server..."
    python server.py &
    sleep 3
fi

echo "🚀 Running presentation demo..."
python presentation_demo.py

echo "📊 Running performance benchmark..."
python demo_parallel_inference.py

echo "✅ Demo completed!"