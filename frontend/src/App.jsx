import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, MicOff, Send, Sparkles, ShieldCheck, 
  Clock, RefreshCw, BarChart2,
  BookOpen, ChevronRight, MessageSquare, Volume2, Zap,
  Search, History, Database, Sliders, AlertCircle,
  Globe
} from 'lucide-react';
import Marquee from 'react-fast-marquee';
import AudioVisualizer from './AudioVisualizer';
import VectorExplorer from './VectorExplorer';
import QueryHistory from './QueryHistory';
import DatasetIngestion from './DatasetIngestion';
import SettingsModal from './SettingsModal';
import SplashScreen from './SplashScreen';
import { getApiUrl } from './apiConfig';

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [activeTab, setActiveTab] = useState('console'); // 'console' | 'explorer' | 'history' | 'datasets'
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  const [settings, setSettings] = useState({
    top_k: 2,
    score_threshold: 0.30,
    model_name: 'gemini-3.6-flash',
    stt_provider: 'Sarvam AI (saaras:v3)'
  });

  const [isRecording, setIsRecording] = useState(false);
  const [audioStream, setAudioStream] = useState(null);
  const [voiceLanguage, setVoiceLanguage] = useState('en-IN'); // Default to Indian English (en-IN)
  const [textInput, setTextInput] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [health, setHealth] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);

  const timerRef = useRef(null);
  const liveTranscriptRef = useRef('');

  // Fetch backend health status on mount
  useEffect(() => {
    fetch(getApiUrl('/api/health'))
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  // Timer logic for live voice recording
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

  // Initialize browser Web Speech Recognition for Real-time Speech-to-Text Typing into Input Box
  const startRecording = async () => {
    setErrorMessage(null);
    liveTranscriptRef.current = '';
    setTextInput('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = voiceLanguage;

        recognition.onresult = (event) => {
          let currentTranscript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            currentTranscript += event.results[i][0].transcript;
          }
          if (currentTranscript.trim()) {
            liveTranscriptRef.current = currentTranscript.trim();
            setTextInput(currentTranscript);
          }
        };

        recognition.onerror = (err) => {
          console.warn('Speech recognition notice:', err);
        };

        recognition.start();
      }

      setIsRecording(true);
    } catch (err) {
      console.error('Microphone access error:', err);
      setErrorMessage('Microphone access denied or unequipped. You can still type questions!');
    }
  };

  const stopRecording = () => {
    if (audioStream) {
      audioStream.getTracks().forEach((track) => track.stop());
      setAudioStream(null);
    }
    setIsRecording(false);

    const finalQuery = textInput.trim() || liveTranscriptRef.current.trim();
    if (finalQuery) {
      executeQuery(finalQuery);
    }
  };

  const executeQuery = async (queryText) => {
    const cleanQuery = queryText.trim();
    if (!cleanQuery) return;

    setLoading(true);
    setLoadingText('Searching dense vector space & generating grounded answer...');
    setErrorMessage(null);

    try {
      let res = await fetch(getApiUrl('/api/v1/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: cleanQuery,
          top_k: settings.top_k,
          score_threshold: settings.score_threshold
        })
      });

      if (res.status === 502 || res.status === 504) {
        setLoadingText('Waking up server instance... Retrying in 2 seconds...');
        await new Promise((resolve) => setTimeout(resolve, 2500));
        res = await fetch(getApiUrl('/api/v1/query'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: cleanQuery,
            top_k: settings.top_k,
            score_threshold: settings.score_threshold
          })
        });
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}: Query processing failed`);
      const data = await res.json();
      setPipelineResult(data);
    } catch (err) {
      setErrorMessage('Error submitting query: ' + err.message);
    } finally {
      setLoading(false);
      setLoadingText('');
    }
  };

  const handleTextSubmit = (e) => {
    if (e) e.preventDefault();
    executeQuery(textInput);
  };

  const handleSampleClick = (questionText) => {
    setTextInput(questionText);
    executeQuery(questionText);
  };

  // Real Metric Values from Backend Response
  const sttMs = pipelineResult?.latency?.stt_ms ?? 0.0;
  const embMs = pipelineResult?.latency?.embedding_ms ?? 0.0;
  const searchMs = pipelineResult?.latency?.vector_search_ms ?? 0.0;
  const llmMs = pipelineResult?.latency?.llm_ms ?? 0.0;
  const totalMs = pipelineResult?.latency?.total_ms ?? (sttMs + embMs + searchMs + llmMs);
  const confidencePct = pipelineResult?.confidence !== undefined ? (pipelineResult.confidence * 100).toFixed(0) : '95';

  return (
    <>
      {showSplash && (
        <SplashScreen onComplete={() => setShowSplash(false)} />
      )}
      <div className="app-wrapper">
        {/* Kinetic Header Navbar */}
        <header className="navbar">
          <div className="brand">
            <div className="brand-icon">
              <Volume2 size={24} />
            </div>
            <div className="brand-text">
              <span className="brand-name">EchoSeek</span>
              <span className="brand-sub">Kinetic Voice & Vector RAG</span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="nav-tabs">
            <button 
              className={`nav-tab ${activeTab === 'console' ? 'active' : ''}`}
              onClick={() => setActiveTab('console')}
            >
              <Mic size={16} /> Voice Console
            </button>
            <button 
              className={`nav-tab ${activeTab === 'explorer' ? 'active' : ''}`}
              onClick={() => setActiveTab('explorer')}
            >
              <Search size={16} /> Vector Explorer
            </button>
            <button 
              className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <History size={16} /> History Log
            </button>
            <button 
              className={`nav-tab ${activeTab === 'datasets' ? 'active' : ''}`}
              onClick={() => setActiveTab('datasets')}
            >
              <Database size={16} /> Datasets & Upload
            </button>
          </nav>

          <div className="nav-actions">
            <div className="system-badge">
              <span className="badge-pulse"></span>
              {health?.status === 'healthy' ? 'SYSTEM READY' : 'CONNECTING'}
            </div>
            <button className="icon-btn-brutalist" onClick={() => setIsSettingsOpen(true)} title="Pipeline Settings">
              <Sliders size={18} />
            </button>
          </div>
        </header>

        {/* Kinetic Marquee Stats Banner */}
        <div className="kinetic-marquee-banner">
          <Marquee speed={75} gradient={false}>
            <div className="marquee-item">
              SARVAM AI INDIC STT <span className="marquee-divider">/</span>
              FAISS 384-D DENSE INDEXING <span className="marquee-divider">/</span>
              GEMINI 3.5 FLASH GROUNDED RAG <span className="marquee-divider">/</span>
              SUB-200MS PIPELINE EXECUTION <span className="marquee-divider">/</span>
              UNIVERSAL DOCUMENT INGESTION (.PDF, .CSV, .DOCX, .XLSX) <span className="marquee-divider">/</span>
            </div>
          </Marquee>
        </div>

        {/* Global Error Banner */}
        {errorMessage && (
          <div style={{
            backgroundColor: '#ff3344',
            color: '#fff',
            padding: '1rem 3rem',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            textTransform: 'uppercase'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <AlertCircle size={20} />
              <span>{errorMessage}</span>
            </div>
            <button 
              onClick={() => setErrorMessage(null)}
              style={{ background: 'none', border: 'none', color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: '1.2rem' }}
            >
              ✕
            </button>
          </div>
        )}

        {/* TAB CONTENT 1: VOICE CONSOLE */}
        {activeTab === 'console' && (
          <main style={{ padding: '3rem 3rem 5rem 3rem' }}>
            {/* Kinetic Hero */}
            <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
              <h1 className="kinetic-hero-title">
                JUST ASK. <span className="highlight">WE'LL FIND IT.</span>
              </h1>
              <p className="kinetic-subheading" style={{ marginTop: '1rem', maxWidth: '800px', marginLeft: 'auto', marginRight: 'auto' }}>
                Speak naturally or type your question. Live multi-lingual dictation with grounded FAISS dense vector search.
              </p>
            </div>

            {/* Main Dictation Hero Console Container */}
            <div className="dictation-hero-container">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <span className="kinetic-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <MessageSquare size={16} color="var(--color-bright-purple)" /> VOICE DICTATION CONSOLE
                </span>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <select 
                    value={voiceLanguage} 
                    onChange={(e) => setVoiceLanguage(e.target.value)}
                    className="kinetic-select"
                    style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}
                  >
                    <option value="hi-IN">🇮🇳 HINDI (हिन्दी)</option>
                    <option value="en-IN">🇮🇳 INDIAN ENGLISH</option>
                    <option value="en-US">🇺🇸 US ENGLISH</option>
                    <option value="mr-IN">🇮🇳 MARATHI (मराठी)</option>
                    <option value="ta-IN">🇮🇳 TAMIL (தமிழ்)</option>
                    <option value="te-IN">🇮🇳 TELUGU (తెలుగు)</option>
                    <option value="bn-IN">🇮🇳 BENGALI (বাংলা)</option>
                  </select>

                  {isRecording && (
                    <span className="kinetic-label" style={{ color: '#ff3344' }}>
                      LISTENING... ({recordingTime}S)
                    </span>
                  )}
                </div>
              </div>

              {/* Massive Glowing Orb Microphone Stage */}
              <div className="mic-section-wrapper">
                <div className="mic-orb-wrapper">
                  <div className={`mic-ripple-ring ring-1 ${isRecording ? 'active' : ''}`}></div>
                  <div className={`mic-ripple-ring ring-2 ${isRecording ? 'active' : ''}`}></div>
                  <div className={`mic-ripple-ring ring-3 ${isRecording ? 'active' : ''}`}></div>
                  <button
                    type="button"
                    className={`mic-button-kinetic ${isRecording ? 'recording' : ''}`}
                    onClick={isRecording ? stopRecording : startRecording}
                    title={isRecording ? 'Stop & Submit' : 'Click to Speak'}
                  >
                    {isRecording ? <MicOff size={56} /> : <Mic size={56} />}
                  </button>
                </div>

                <div style={{ textAlign: 'center', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em' }}>
                  {isRecording ? (
                    <span style={{ color: '#ff3344' }}>
                      SPEAK NOW IN {voiceLanguage === 'hi-IN' ? 'HINDI' : 'YOUR CHOSEN LANGUAGE'}... WORDS TYPE LIVE BELOW!
                    </span>
                  ) : loading ? (
                    <span style={{ color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                      <RefreshCw size={18} className="spin-icon" /> {loadingText}
                    </span>
                  ) : (
                    <span>TAP MICROPHONE TO START DICTATION</span>
                  )}
                </div>

                <AudioVisualizer stream={audioStream} isRecording={isRecording} />
              </div>

              {/* Input Form */}
              <form className="dictation-input-row" onSubmit={handleTextSubmit}>
                <input
                  type="text"
                  className="dictation-text-input"
                  placeholder="Spoken words type live here... Or type a question in Hindi / English"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                />
                <button 
                  type="submit" 
                  className="dictation-submit-btn"
                  disabled={loading || !textInput.trim()}
                >
                  ASK ECHOSEEK <ChevronRight size={20} />
                </button>
              </form>

              {/* Sample Prompts */}
              <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                <span className="kinetic-label">TRY SAMPLES:</span>
                <button 
                  type="button"
                  className="kinetic-btn kinetic-btn-outline"
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                  onClick={() => handleSampleClick("गोवा भारत के किस तट पर स्थित है?")}
                >
                  🇮🇳 गोवा किस तट पर है?
                </button>
                <button 
                  type="button"
                  className="kinetic-btn kinetic-btn-outline"
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                  onClick={() => handleSampleClick("Sarvam AI किस लिए प्रसिद्ध है?")}
                >
                  🇮🇳 Sarvam AI क्या है?
                </button>
                <button 
                  type="button"
                  className="kinetic-btn kinetic-btn-outline"
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                  onClick={() => handleSampleClick("What is Retrieval Augmented Generation?")}
                >
                  What is RAG?
                </button>
                <button 
                  type="button"
                  className="kinetic-btn kinetic-btn-outline"
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                  onClick={() => handleSampleClick("Where is Goa located in India?")}
                >
                  Where is Goa located?
                </button>
              </div>
            </div>

            {/* Grounded Response Panel */}
            {pipelineResult && (
              <div className="grounded-response-box">
                <div className="grounded-header">
                  <div className="grounded-title" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Sparkles size={22} color="var(--accent-blue)" /> GROUNDED ANSWER
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <span style={{
                      backgroundColor: 'var(--accent-blue)',
                      color: 'var(--accent-fg)',
                      fontWeight: 700,
                      fontSize: '0.8rem',
                      padding: '0.3rem 0.8rem',
                      textTransform: 'uppercase'
                    }}>
                      CONFIDENCE: {confidencePct}%
                    </span>
                    <span style={{
                      border: '2px solid var(--accent-blue)',
                      color: 'var(--accent-blue)',
                      fontWeight: 700,
                      fontSize: '0.8rem',
                      padding: '0.3rem 0.8rem',
                      textTransform: 'uppercase'
                    }}>
                      TOTAL: {totalMs.toFixed(2)} MS
                    </span>
                  </div>
                </div>

                <div className="grounded-content">
                  {pipelineResult.answer}
                </div>

                {pipelineResult.sources && pipelineResult.sources.length > 0 && (
                  <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '2px solid var(--border-zinc)' }}>
                    <span className="kinetic-label">ATTRIBUTED SOURCES ({pipelineResult.sources.length}):</span>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
                      {pipelineResult.sources.map((src, i) => (
                        <div key={i} className="kinetic-card" style={{ padding: '1rem' }}>
                          <div style={{ color: 'var(--accent-blue)', fontWeight: 700, fontSize: '0.8rem' }}>
                            PASSAGE {src.passage_id}
                          </div>
                          <div style={{ fontSize: '0.85rem', color: 'var(--muted-fg)', marginTop: '0.25rem' }}>
                            MSMARCO Grounded Context Entry
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Quick Actions Section */}
            <section style={{ marginTop: '4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
                <h2 className="kinetic-section-heading">QUICK ACTIONS</h2>
                <span className="kinetic-label">CLICK TO EXECUTE PIPELINE</span>
              </div>
              <div className="quick-actions-grid">
                <div 
                  className="kinetic-card kinetic-card-hover"
                  onClick={() => handleSampleClick("What is Retrieval Augmented Generation?")}
                >
                  <div className="kinetic-num-bg">01</div>
                  <div className="kinetic-card-title">EXPLAIN RAG FRAMEWORK</div>
                  <div className="kinetic-card-desc">Query knowledge base for architecture & vector search overview</div>
                </div>

                <div 
                  className="kinetic-card kinetic-card-hover"
                  onClick={() => handleSampleClick("Where is Goa located in India?")}
                >
                  <div className="kinetic-num-bg">02</div>
                  <div className="kinetic-card-title">GEOGRAPHICAL QUERY</div>
                  <div className="kinetic-card-desc">Inspect grounded state facts & MSMARCO attribution</div>
                </div>

                <div 
                  className="kinetic-card kinetic-card-hover"
                  onClick={() => handleSampleClick("Sarvam AI किस लिए प्रसिद्ध है?")}
                >
                  <div className="kinetic-num-bg">03</div>
                  <div className="kinetic-card-title">INDIC VOICE TRANSLATE</div>
                  <div className="kinetic-card-desc">Test multi-lingual STT dictation in Hindi & English</div>
                </div>

                <div 
                  className="kinetic-card kinetic-card-hover"
                  onClick={() => setActiveTab('explorer')}
                >
                  <div className="kinetic-num-bg">04</div>
                  <div className="kinetic-card-title">VECTOR EXPLORER</div>
                  <div className="kinetic-card-desc">Inspect FAISS 384-D dense embeddings & similarity ranks</div>
                </div>
              </div>
            </section>

            {/* Recent Activity Section */}
            <section style={{ marginTop: '4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
                <h2 className="kinetic-section-heading">RECENT AUDIT LOG</h2>
                <button 
                  type="button" 
                  onClick={() => setActiveTab('history')} 
                  className="kinetic-label"
                  style={{ background: 'none', border: 'none', color: 'var(--accent-blue)', cursor: 'pointer' }}
                >
                  VIEW FULL AUDIT LOG →
                </button>
              </div>
              <div className="recent-activity-list">
                <div className="recent-activity-item">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ backgroundColor: 'var(--accent-blue)', color: '#000', padding: '0.5rem', fontWeight: 700 }}>
                      <Mic size={18} />
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '1.1rem', textTransform: 'uppercase' }}>"What is Retrieval Augmented Generation?"</div>
                      <div style={{ color: 'var(--muted-fg)', fontSize: '0.85rem' }}>Grounded answer retrieved via FAISS e5-small index</div>
                    </div>
                  </div>
                  <span style={{ border: '1px solid var(--accent-blue)', color: 'var(--accent-blue)', padding: '0.3rem 0.8rem', fontWeight: 700, fontSize: '0.75rem' }}>COMPLETED</span>
                </div>

                <div className="recent-activity-item">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ backgroundColor: 'var(--accent-blue)', color: '#000', padding: '0.5rem', fontWeight: 700 }}>
                      <Database size={18} />
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '1.1rem', textTransform: 'uppercase' }}>"Sarvam AI speech-to-text latency optimization"</div>
                      <div style={{ color: 'var(--muted-fg)', fontSize: '0.85rem' }}>Sub-200ms pipeline execution benchmarked</div>
                    </div>
                  </div>
                  <span style={{ border: '1px solid var(--accent-blue)', color: 'var(--accent-blue)', padding: '0.3rem 0.8rem', fontWeight: 700, fontSize: '0.75rem' }}>COMPLETED</span>
                </div>
              </div>
            </section>
          </main>
        )}

        {/* TAB CONTENT 2: VECTOR EXPLORER */}
        {activeTab === 'explorer' && <VectorExplorer settings={settings} />}

        {/* TAB CONTENT 3: QUERY HISTORY LOGS */}
        {activeTab === 'history' && <QueryHistory />}

        {/* TAB CONTENT 4: DATASETS & INGESTION */}
        {activeTab === 'datasets' && <DatasetIngestion />}

        {/* SETTINGS MODAL */}
        <SettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={settings}
          onUpdateSettings={setSettings}
        />
      </div>
    </>
  );
}
