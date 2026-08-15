# Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

A low-latency (<200 ms target) voice-driven Retrieval-Augmented Generation (RAG) system built on the **AI4Bharat MSMARCO-XI** dataset.

## Architecture & Technology Stack
- **STT Engine**: Sarvam AI Speech-to-Text
- **LLM Provider**: Google Gemini (`gemini-1.5-flash` / `gemini-2.5-flash`)
- **Embedding Model**: `intfloat/multilingual-e5-small`
- **Vector Storage**: FAISS (Local In-Memory Similarity Search Index)
- **Backend Framework**: Python FastAPI + Async Uvicorn
- **Frontend UI**: React + Vite (Glassmorphism & Voice Interface)
- **Evaluation**: Latency Analytics Dashboard (P50 / P70 / P100 Metrics)

## Directory Structure
```text
├── backend/            # FastAPI application & server logic
│   ├── app/            # Routes, services, model harness & guardrails
│   └── requirements.txt
├── frontend/           # React + Vite user interface
│   ├── src/            # Audio recording, visualizer & state
│   └── package.json
├── data/               # MSMARCO-XI dataset subsets & FAISS index
├── scripts/            # Dataset exploration & chunking scripts
├── tests/              # Benchmark & latency testing suites
└── docs/               # Architecture design & decision notes
```

## Quick Start (Local Development)

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.
