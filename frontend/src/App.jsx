import React, { useState, useEffect } from 'react';
import { Mic, Activity, Database, Sparkles, CheckCircle2, XCircle } from 'lucide-react';

function App() {
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error('Backend not responding');
        return res.json();
      })
      .then((data) => {
        setHealthStatus(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="container">
      <header className="header">
        <span className="badge">
          <Sparkles size={14} /> Hacker House Goa 2026 — Task 2
        </span>
        <h1 className="title">Voice-Enabled RAG Model</h1>
        <p className="subtitle">
          Ultra Low-Latency (&lt;200 ms) Speech Retrieval & Answers over MSMARCO-XI
        </p>
      </header>

      <main>
        <div className="glass-card" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
          <div style={{ margin: '1.5rem 0' }}>
            <button 
              className="badge" 
              style={{ padding: '0.85rem 1.75rem', fontSize: '1rem', cursor: 'pointer', background: 'linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(79, 172, 254, 0.2))' }}
              onClick={() => alert('Microphone pipeline will be built in Module 8!')}
            >
              <Mic size={18} /> Module 2 Ready: Click to Test UI
            </button>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Project architecture & environment setup initialized successfully.
          </p>
        </div>

        <div className="glass-card">
          <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--accent-cyan)" /> System Health Check
          </h2>

          {loading && <p style={{ color: 'var(--text-muted)' }}>Connecting to FastAPI backend...</p>}
          
          {error && (
            <div className="status-item" style={{ color: '#ef4444' }}>
              <span>Backend Status</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <XCircle size={16} /> Disconnected (Start FastAPI server)
              </span>
            </div>
          )}

          {healthStatus && (
            <div className="status-grid">
              <div className="glass-card" style={{ padding: '1rem' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Backend Service</div>
                <div style={{ marginTop: '0.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div className="status-dot"></div> {healthStatus.status.toUpperCase()}
                </div>
              </div>

              <div className="glass-card" style={{ padding: '1rem' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Embedding Model</div>
                <div style={{ marginTop: '0.25rem', fontWeight: 600 }}>
                  {healthStatus.embedding_model}
                </div>
              </div>

              <div className="glass-card" style={{ padding: '1rem' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Sarvam STT Key</div>
                <div style={{ marginTop: '0.25rem', fontWeight: 600, color: healthStatus.sarvam_configured ? '#22c55e' : '#f59e0b' }}>
                  {healthStatus.sarvam_configured ? 'Configured' : 'Missing in .env'}
                </div>
              </div>

              <div className="glass-card" style={{ padding: '1rem' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Gemini LLM Key</div>
                <div style={{ marginTop: '0.25rem', fontWeight: 600, color: healthStatus.gemini_configured ? '#22c55e' : '#f59e0b' }}>
                  {healthStatus.gemini_configured ? 'Configured' : 'Missing in .env'}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
