#!/bin/bash

echo "🚀 Setting up Customer Churn Predictor..."

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "❌ Error: data/ directory not found!"
    echo "Please ensure you have the required data files in the data/ directory:"
    echo "- Tabla_01_English_Unique_postEDA.csv"
    echo "- Tabla_02_Clients_English.csv"
    echo "- Tabla_01_English_20211020.csv"
    echo "- Tabla_01_English.csv"
    echo "- Tabla_01_test_English.csv"
    exit 1
fi

# Check if artifacts directory exists, create if not
if [ ! -d "artifacts" ]; then
    echo "📁 Creating artifacts directory..."
    mkdir artifacts
fi

# Train the model
echo "🤖 Training Random Forest model..."
python main.py

if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully!"
    echo "🎯 You can now run the app with: streamlit run app.py"
    echo "🐳 Or build Docker image: docker build -t streamlitchurnapp:latest -f docker/Dockerfile ."
else
    echo "❌ Model training failed. Please check your data files and dependencies."
    exit 1
fi
