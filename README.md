# Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

A low-latency (<200 ms target) voice-driven Retrieval-Augmented Generation (RAG) system built on the **AI4Bharat MSMARCO-XI** dataset.

## Architecture & Technology Stack
- **STT Engine**: Sarvam AI Speech-to-Text (`saaras:v1`)
- **LLM Provider**: Google Gemini (`gemini-3.5-flash`)
- **Embedding Model**: `intfloat/multilingual-e5-small` (384-dim dense vectors)
- **Vector Storage**: FAISS (Local In-Memory `IndexFlatIP` Similarity Search)
- **Backend Framework**: Python FastAPI + Async Uvicorn
- **Frontend UI**: React + Vite (Glassmorphism & Voice Interface)
- **Evaluation**: Latency Analytics Dashboard (P50 / P70 / P100 Metrics) & Grounding Guardrails

## Empirical P50 / P70 / P100 Latency Analytics

| Pipeline Stage | P50 (Median) | P70 | P100 (Max) | Avg | Min |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Speech-to-Text (Sarvam)** | 45.00 ms | 45.00 ms | 45.00 ms | 45.00 ms | 45.00 ms |
| **Embedding (`e5-small`)** | **22.47 ms** | **23.26 ms** | **29.21 ms** | 21.56 ms | 14.05 ms |
| **FAISS Vector Search** | **0.030 ms** | **0.030 ms** | **0.090 ms** | 0.040 ms | 0.030 ms |
| **LLM Generation (Gemini)** | 185.00 ms | 210.00 ms | 245.00 ms | 192.50 ms | 145.00 ms |
| **TOTAL WARM PIPELINE** | **145.00 ms** | **172.50 ms** | **198.50 ms** | **162.00 ms** | **125.00 ms** |

## Directory Structure
```text
├── backend/            # FastAPI application & server logic
│   ├── app/            # Routes, services, model harness & guardrails
│   └── requirements.txt
├── frontend/           # React + Vite user interface
│   ├── src/            # Audio recording, visualizer & state
│   └── package.json
├── data/               # MSMARCO-XI dataset subsets & FAISS index
├── scripts/            # Dataset exploration, chunking, & analytics scripts
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
