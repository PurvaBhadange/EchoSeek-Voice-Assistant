# Architectural Decisions Document (Module 0 & Module 1)

## Project Overview
- **Project**: Hacker House Goa 2026 — Task 2 (Voice-Enabled RAG Model)
- **Dataset**: AI4Bharat MSMARCO-XI (`ai4bharat/MSMARCO-XI`)
- **Target Performance**: End-to-end Voice RAG response target < 200 ms latency.

## Architecture Decisions

### 1. Backend: Python FastAPI
- **Choice**: FastAPI + Async Uvicorn
- **Rationale**: Direct integration with Python AI/ML ecosystem (PyTorch, Hugging Face, FAISS), native async request handling, auto-generated OpenAPI documentation.

### 2. Frontend: React + Vite
- **Choice**: React 18 SPA with Vite build tool
- **Rationale**: Ultra-fast build times, lightweight client footprint, straightforward browser Web Audio API integration for microphone recording.

### 3. Speech-to-Text (STT): Sarvam AI
- **Choice**: Sarvam AI API
- **Rationale**: Native optimization for Indian accents, Indic language support, fast API turnaround time.

### 4. LLM Provider: Google Gemini
- **Choice**: `gemini-1.5-flash` / `gemini-2.5-flash`
- **Rationale**: High reliability, strong reasoning, structured JSON output validation, high quality context grounding.

### 5. Embedding Model: `intfloat/multilingual-e5-small`
- **Choice**: Local Sentence-Transformers `intfloat/multilingual-e5-small`
- **Rationale**: Zero API network latency, free execution, local vector generation in ~5–15 ms, superior multilingual semantic quality for Indic / English MSMARCO-XI queries.

### 6. Vector Database: FAISS (Facebook AI Similarity Search)
- **Choice**: Local In-Memory FAISS Index
- **Rationale**: Search speed < 2 ms locally with 0 network overhead, enabling achievement of the <200 ms pipeline target.

### 7. Deployment Platform: Decoupled (Vercel + Render / HF Spaces)
- **Choice**: Decoupled deployment
- **Rationale**: Independent frontend static CDN caching via Vercel, dedicated Python web service environment for FastAPI.
