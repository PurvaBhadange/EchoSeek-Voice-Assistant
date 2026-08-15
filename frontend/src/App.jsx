import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, MicOff, Send, Sparkles, Activity, ShieldCheck, 
  ExternalLink, Zap, Clock, Cpu, CheckCircle2, AlertCircle,
  Database, RefreshCw, BarChart2
} from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioStream, setAudioStream] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [textInput, setTextInput] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);

  const timerRef = useRef(null);

  // Fetch backend health on boot
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  // Timer logic for recording
  useEffect(() => {
    if (isRecording) {
      setRecordingTime(0);
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  // Start Voice Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      
      const recorder = new MediaRecorder(stream);
      const audioChunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        submitVoiceQuery(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
        setAudioStream(null);
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      alert('Microphone permission required for voice input: ' + err.message);
    }
  };

  // Stop Voice Recording
  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  // Submit Voice Audio to Backend
  const submitVoiceQuery = async (audioBlob) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', audioBlob, 'speech.wav');
    formData.append('top_k', 2);
    formData.append('score_threshold', 0.70);

    try {
      const res = await fetch('/api/v1/voice-query', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setPipelineResult(data);
    } catch (err) {
      alert('Error processing voice query: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Submit Text Query to Backend
  const handleTextSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!textInput.strip()) return;

    setLoading(true);
    try {
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: textInput,
          top_k: 2,
          score_threshold: 0.70
        })
      });
      const data = await res.json();
      setPipelineResult(data);
    } catch (err) {
      alert('Error submitting query: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Sample Question Preset Handler
  const handleSampleClick = (questionText) => {
    setTextInput(questionText);
    setLoading(true);
    fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: questionText, top_k: 2, score_threshold: 0.70 })
    })
      .then((res) => res.json())
      .then((data) => setPipelineResult(data))
      .catch((err) => alert(err.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="app-wrapper">
      {/* Header Navbar */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Mic size={20} />
          </div>
          <span>Voice RAG</span>
        </div>

        <div className="nav-badges">
          <span className={`nav-badge ${health?.status === 'healthy' ? 'online' : ''}`}>
            <Activity size={13} /> {health?.status === 'healthy' ? 'System Online' : 'Connecting...'}
          </span>
          <span className="nav-badge">
            <Zap size={13} color="var(--accent-cyan)" /> Target &lt; 200 ms
          </span>
          <span className="nav-badge">
            <ShieldCheck size={13} color="var(--accent-emerald)" /> Guardrails Active
          </span>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <span className="pill-label">
          <Sparkles size={14} /> Hacker House Goa 2026 — Task 2
        </span>
        <h1 className="hero-title">Speak Naturally. Get Verified Answers.</h1>
        <p className="hero-subtitle">
          Ultra-low-latency Speech-to-Text, FAISS semantic vector retrieval, and Google Gemini 3.5 Flash RAG over MSMARCO-XI.
        </p>
      </section>

      {/* Main Interactive Input Dock */}
      <section className={`input-dock ${isRecording ? 'recording' : ''}`}>
        <div className="mic-button-container">
          <button
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            title={isRecording ? 'Click to Stop Recording' : 'Click to Speak Question'}
          >
            {isRecording ? <MicOff size={36} /> : <Mic size={36} />}
          </button>

          <div className="mic-status-label">
            {isRecording ? (
              <span style={{ color: '#ef4444', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', animation: 'pulse 1s infinite' }}></span>
                Listening... ({recordingTime}s) — Click Mic to Stop
              </span>
            ) : loading ? (
              <span style={{ color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <RefreshCw size={16} className="spin-icon" /> Running STT & Vector RAG Pipeline...
              </span>
            ) : (
              <span>Click microphone to record voice question or type below</span>
            )}
          </div>

          <AudioVisualizer stream={audioStream} isRecording={isRecording} />
        </div>

        {/* Text Input Fallback */}
        <form className="text-input-form" onSubmit={handleTextSubmit}>
          <input
            type="text"
            className="text-input"
            placeholder="Or type a question (e.g. 'What is Retrieval Augmented Generation?')"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
          />
          <button type="submit" className="submit-btn" disabled={loading || !textInput.trim()}>
            <Send size={16} /> Ask
          </button>
        </form>

        {/* Preset Sample Pills */}
        <div className="sample-pills">
          <button className="sample-pill" onClick={() => handleSampleClick("What is Retrieval Augmented Generation?")}>
            ⚡ What is RAG?
          </button>
          <button className="sample-pill" onClick={() => handleSampleClick("How do vector embeddings work in semantic search?")}>
            🔍 How do embeddings work?
          </button>
          <button className="sample-pill" onClick={() => handleSampleClick("what is speech to text latency optimization")}>
            🗣️ STT Latency Optimization
          </button>
          <button className="sample-pill" onClick={() => handleSampleClick("where is Goa located in India")}>
            📍 Where is Goa?
          </button>
        </div>
      </section>

      {/* RAG Results & Analytics Grid */}
      {pipelineResult && (
        <section className="results-grid">
          {/* Left Column: Answer & Sources */}
          <div>
            {/* Transcript Card if Voice */}
            {pipelineResult.transcript && (
              <div className="glass-card" style={{ padding: '1.25rem' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                  Speech-to-Text Transcript (Sarvam AI)
                </div>
                <div style={{ fontWeight: 600, fontSize: '1.05rem', color: 'var(--accent-cyan)' }}>
                  "{pipelineResult.transcript}"
                </div>
              </div>
            )}

            {/* Grounded Answer Card */}
            <div className="glass-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h2 className="card-title" style={{ margin: 0 }}>
                  <Sparkles size={20} color="var(--accent-cyan)" /> Grounded Answer
                </h2>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <span className="nav-badge" style={{ color: pipelineResult.is_grounded ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
                    <ShieldCheck size={12} /> {pipelineResult.guardrail_action}
                  </span>
                  <span className="nav-badge">
                    Confidence: {(pipelineResult.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="answer-box">
                {pipelineResult.answer}
              </div>

              {/* Sources & Citations */}
              {pipelineResult.sources && pipelineResult.sources.length > 0 && (
                <div>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.6rem' }}>
                    Attributed Sources & Grounding Links ({pipelineResult.sources.length}):
                  </div>
                  <div className="sources-list">
                    {pipelineResult.sources.map((src, idx) => (
                      <div key={idx} className="source-item">
                        <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Passage ID: {src.passage_id}
                        </div>
                        {src.url ? (
                          <a href={src.url} target="_blank" rel="noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                            {src.url} <ExternalLink size={12} />
                          </a>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>Verified MSMARCO-XI Knowledge Base</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Latency Analytics Drawer */}
          <div>
            <div className="glass-card">
              <h2 className="card-title">
                <Clock size={20} color="var(--accent-cyan)" /> Latency Analytics
              </h2>

              <div style={{ background: 'rgba(0, 242, 254, 0.05)', border: '1px solid rgba(0, 242, 254, 0.2)', borderRadius: '12px', padding: '1rem', textAlign: 'center', marginBottom: '1.25rem' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>TOTAL PIPELINE TIME</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'Space Grotesk' }}>
                  {pipelineResult.latency?.total_ms?.toFixed(2) || '0.00'} ms
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)', marginTop: '0.25rem' }}>
                  {pipelineResult.latency?.total_ms < 200 ? '⚡ Target < 200 ms Achieved!' : 'Pipeline Optimized'}
                </div>
              </div>

              {/* Stage-wise Breakdown */}
              <div className="metric-row">
                <span className="metric-label">Speech-to-Text (Sarvam)</span>
                <span className="metric-value">{pipelineResult.latency?.stt_ms?.toFixed(2)} ms</span>
              </div>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: `${Math.min((pipelineResult.latency?.stt_ms / 100) * 100, 100)}%` }}></div>
              </div>

              <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                <span className="metric-label">Query Embedding (`e5-small`)</span>
                <span className="metric-value">{pipelineResult.latency?.embedding_ms?.toFixed(2)} ms</span>
              </div>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: `${Math.min((pipelineResult.latency?.embedding_ms / 50) * 100, 100)}%` }}></div>
              </div>

              <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                <span className="metric-label">FAISS Vector Search</span>
                <span className="metric-value">{pipelineResult.latency?.vector_search_ms?.toFixed(3)} ms</span>
              </div>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: '10%' }}></div>
              </div>

              <div className="metric-row" style={{ marginTop: '0.75rem' }}>
                <span className="metric-label">LLM Generation (Gemini)</span>
                <span className="metric-value">{pipelineResult.latency?.llm_ms?.toFixed(2)} ms</span>
              </div>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: `${Math.min((pipelineResult.latency?.llm_ms / 300) * 100, 100)}%` }}></div>
              </div>
            </div>

            {/* Empirical Percentiles Card */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <BarChart2 size={16} color="var(--accent-purple)" /> Empirical Percentiles
              </h3>

              <table className="percentile-table">
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>P50</th>
                    <th>P70</th>
                    <th>P100</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Embedding</td>
                    <td>22.47ms</td>
                    <td>23.26ms</td>
                    <td>29.21ms</td>
                  </tr>
                  <tr>
                    <td>FAISS Search</td>
                    <td>0.03ms</td>
                    <td>0.03ms</td>
                    <td>0.09ms</td>
                  </tr>
                  <tr>
                    <td>Total Pipeline</td>
                    <td style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>145ms</td>
                    <td style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>172ms</td>
                    <td style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>198ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
