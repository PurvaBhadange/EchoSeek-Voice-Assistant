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
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">
            <Sliders size={18} color="var(--accent-cyan)" /> Pipeline Settings & Config
          </h3>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div className="setting-group">
            <label className="setting-label">
              <Sliders size={14} /> Retrieved Top-K Passages: <strong>{settings.top_k}</strong>
            </label>
            <p className="setting-desc">Number of vector passage chunks fetched from FAISS per query.</p>
            <input
              type="range"
              min="1"
              max="5"
              value={settings.top_k}
              onChange={(e) => handleChange('top_k', Number(e.target.value))}
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">
              <Hash size={14} /> Similarity Score Threshold: <strong>{settings.score_threshold}</strong>
            </label>
            <p className="setting-desc">Minimum cosine similarity score required for passage grounding.</p>
            <input
              type="range"
              min="0.10"
              max="0.80"
              step="0.05"
              value={settings.score_threshold}
              onChange={(e) => handleChange('score_threshold', Number(e.target.value))}
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">
              <Zap size={14} /> LLM Provider & Model
            </label>
            <select
              className="form-input"
              value={settings.model_name}
              onChange={(e) => handleChange('model_name', e.target.value)}
            >
              <option value="gemini-3.6-flash">Google Gemini 3.6 Flash (Sub-200ms)</option>
              <option value="gemini-3.5-flash">Google Gemini 3.5 Flash</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-label">
              <ShieldCheck size={14} /> Speech-to-Text Provider
            </label>
            <select
              className="form-input"
              value={settings.stt_provider}
              onChange={(e) => handleChange('stt_provider', e.target.value)}
            >
              <option value="Sarvam AI (saaras:v1)">Sarvam AI saaras:v1 (Indian Languages & English)</option>
              <option value="WebSpeech (Browser Dictation)">Web Speech API (Real-time Typing)</option>
            </select>
          </div>
        </div>

        <div className="modal-footer">
          <button className="action-btn secondary" onClick={handleReset}>
            <RotateCcw size={14} /> Reset Defaults
          </button>
          <button className="action-btn primary" onClick={onClose}>
            Apply Settings
          </button>
        </div>
      </div>
    </div>
  );
}
