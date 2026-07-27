# Automatic Modeling Workflow

An enterprise-grade automatic modeling demonstration platform integrating a FastAPI backend with an interactive web frontend.

## 🚀 Project Overview
This project provides an end-to-end demonstration of data uploading, automated feature engineering, model training (powered by AutoGluon), and comprehensive model evaluation & visualization.

## 📂 Project Structure
- /backend: FastAPI service for data processing and model orchestration.
    - main.py: Backend API entry point.
    - modeling_manager.py: Workflow controller.
    - model/: Core modeling logic and configuration.
    - models/: Persistent storage for trained model artifacts.
- /workflow-demo: Frontend visualization interface.
    - index.html: Web interface built with Tailwind CSS.

## 🛠 Tech Stack
- **Backend**: Python 3.11, FastAPI, AutoGluon, pandas
- **Frontend**: HTML, Tailwind CSS, JavaScript (Fetch API)

## ⚙️ Setup & Execution
1. **Backend Service**:
   Navigate to the ackend directory:
   `bash
   python main.py
   `
   (Ensure the API is running on port 8001)

2. **Frontend Service**:
   From the project root:
   `bash
   python -m http.server 8080
   `
   Access the demo at: http://localhost:8080/workflow-demo/index.html

## 💡 Workflow
1. **Configuration**: Upload your CSV dataset and configure the train/test split ratio.
2. **Modeling**: Monitor real-time system logs and progress bars during the training process.
3. **Analysis**: View model performance metrics (Accuracy, KS, AR), feature importance (IV ranking), and AI-driven expert insights.

## 📜 License
MIT
