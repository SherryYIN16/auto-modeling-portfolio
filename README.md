# Automatic Modeling Workflow

An enterprise-grade automatic modeling demonstration platform integrating a FastAPI backend with an interactive web frontend.

## 🚀 Project Overview
This project provides an end-to-end demonstration of data uploading, automated feature engineering, model training (powered by AutoGluon), and comprehensive model evaluation & visualization.

## 📸 Workflow Preview

### 1. Configuration Page
The entry point for uploading data and configuring model parameters.
![Setup Configuration](./initial_page1.png)

### 2. Modeling Progress Page
Real-time monitoring of system logs and the automated modeling pipeline.
![Modeling Progress](./modeling_page2.png)

### 3. Model Analysis Report
Comprehensive visualization of model performance and AI-driven insights.
![Model Analysis](./report_page3.png)

## 📂 Project Structure
- /backend: FastAPI service for data processing and model orchestration.
- /workflow-demo: Frontend visualization interface (index.html).

## 🛠 Tech Stack
- **Backend**: Python 3.11, FastAPI, AutoGluon, pandas
- **Frontend**: HTML, Tailwind CSS, JavaScript (Fetch API)

## ⚙️ Setup & Execution
1. **Backend Service**:
   `bash
   python backend/main.py
   `
2. **Frontend Service**:
   `bash
   python -m http.server 8080
   `
   Access at: http://localhost:8080/workflow-demo/index.html

## 📜 License
MIT
