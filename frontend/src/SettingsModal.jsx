import React from 'react';
import { X, Sliders, Hash, ShieldCheck, Zap, RotateCcw } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, settings, onUpdateSettings }) {
  if (!isOpen) return null;

  const handleChange = (key, val) => {
    onUpdateSettings({ ...settings, [key]: val });
  };

  const handleReset = () => {
    onUpdateSettings({
      top_k: 2,
      score_threshold: 0.30,
      model_name: 'gemini-3.6-flash',
      embedding_model: 'intfloat/multilingual-e5-small',
      stt_provider: 'Sarvam AI (saaras:v3)'
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content-kinetic" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', borderBottom: '2px solid var(--border-zinc)', paddingBottom: '1rem' }}>
          <h3 className="kinetic-card-title" style={{ margin: 0 }}>
            PIPELINE CONFIGURATION
          </h3>
          <button className="icon-btn-brutalist" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', marginBottom: '2.5rem' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span className="kinetic-label">RETRIEVED TOP-K PASSAGES</span>
              <span className="kinetic-label" style={{ color: 'var(--accent-yellow)' }}>{settings.top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              value={settings.top_k}
              onChange={(e) => handleChange('top_k', Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-yellow)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span className="kinetic-label">SIMILARITY SCORE THRESHOLD</span>
              <span className="kinetic-label" style={{ color: 'var(--accent-yellow)' }}>{settings.score_threshold}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.80"
              step="0.05"
              value={settings.score_threshold}
              onChange={(e) => handleChange('score_threshold', Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-yellow)' }}
            />
          </div>

          <div>
            <span className="kinetic-label" style={{ display: 'block', marginBottom: '0.5rem' }}>LLM MODEL PROVIDER</span>
            <select
              className="kinetic-select"
              value={settings.model_name}
              onChange={(e) => handleChange('model_name', e.target.value)}
            >
              <option value="gemini-3.6-flash">GOOGLE GEMINI 3.6 FLASH (SUB-200MS)</option>
              <option value="gemini-3.5-flash">GOOGLE GEMINI 3.5 FLASH</option>
            </select>
          </div>

          <div>
            <span className="kinetic-label" style={{ display: 'block', marginBottom: '0.5rem' }}>SPEECH-TO-TEXT PROVIDER</span>
            <select
              className="kinetic-select"
              value={settings.stt_provider}
              onChange={(e) => handleChange('stt_provider', e.target.value)}
            >
              <option value="Sarvam AI (saaras:v1)">SARVAM AI SAARAS:V1 (INDIAN LANGUAGES)</option>
              <option value="WebSpeech (Browser Dictation)">WEB SPEECH API (REAL-TIME DICTATION)</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
          <button className="kinetic-btn kinetic-btn-outline" onClick={handleReset}>
            <RotateCcw size={16} /> RESET
          </button>
          <button className="kinetic-btn kinetic-btn-primary" onClick={onClose}>
            APPLY CONFIG
          </button>
        </div>
      </div>
    </div>
  );
}
