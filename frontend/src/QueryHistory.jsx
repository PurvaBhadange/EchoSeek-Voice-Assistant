import React, { useState, useEffect } from 'react';
import { History, Trash2, Clock, CheckCircle, AlertTriangle, Mic, MessageSquare, RefreshCw } from 'lucide-react';
import { getApiUrl } from './apiConfig';

export default function QueryHistory() {
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch(getApiUrl('/api/v1/history'));
      if (res.ok) {
        const data = await res.json();
        setHistoryData(data.history || []);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all query history logs?')) return;
    try {
      await fetch(getApiUrl('/api/v1/history'), { method: 'DELETE' });
      setHistoryData([]);
    } catch (err) {
      alert('Failed to clear history');
    }
  };

  return (
    <div style={{ padding: '3rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '2rem' }}>
        <div>
          <h2 className="kinetic-section-heading">QUERY AUDIT LOG</h2>
          <p className="kinetic-subheading" style={{ marginTop: '0.5rem' }}>
            Historical record of voice & text dictation, grounding verification, and execution latency.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button onClick={fetchHistory} className="kinetic-btn kinetic-btn-outline" style={{ padding: '0.5rem 1rem' }}>
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} /> REFRESH
          </button>
          {historyData.length > 0 && (
            <button onClick={handleClearHistory} className="kinetic-btn kinetic-btn-outline" style={{ padding: '0.5rem 1rem', borderColor: '#ff3344', color: '#ff3344' }}>
              <Trash2 size={16} /> CLEAR LOGS
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="kinetic-card" style={{ textAlign: 'center', padding: '4rem' }}>
          <RefreshCw size={32} className="spin-icon" color="var(--accent-blue)" style={{ marginBottom: '1rem' }} />
          <div className="kinetic-label">LOADING AUDIT LOGS...</div>
        </div>
      ) : historyData.length === 0 ? (
        <div className="kinetic-card" style={{ textAlign: 'center', padding: '4rem' }}>
          <History size={48} color="var(--muted-fg)" style={{ marginBottom: '1rem' }} />
          <div className="kinetic-card-title">NO AUDIT LOGS RECORDED</div>
          <p className="kinetic-card-desc">Submit queries in the Voice Console to populate execution benchmarks.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {historyData.map((item, idx) => (
            <div key={item.id || idx} className="kinetic-card kinetic-card-hover" style={{ padding: '1.5rem 2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className="kinetic-label" style={{ backgroundColor: 'var(--accent-blue)', color: '#000', padding: '0.2rem 0.6rem' }}>
                    {item.type === 'voice' ? 'VOICE' : 'TEXT'}
                  </span>
                  <span className="kinetic-label">{item.timestamp}</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <span className="kinetic-label" style={{ color: item.is_grounded ? 'var(--accent-blue)' : '#ff3344' }}>
                    {item.is_grounded ? '✓ GROUNDED' : '⚠️ UNVERIFIED'}
                  </span>
                  <span className="kinetic-label" style={{ border: '1px solid var(--border-zinc)', padding: '0.2rem 0.6rem' }}>
                    TOTAL: {(item.latency?.total_ms || 0).toFixed(1)} MS
                  </span>
                </div>
              </div>

              <div style={{ fontSize: '1.3rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                "{item.query}"
              </div>
              <p className="kinetic-card-desc" style={{ marginBottom: '1rem' }}>
                {item.answer}
              </p>

              <div style={{ display: 'flex', gap: '1.5rem', borderTop: '1px solid var(--border-zinc)', paddingTop: '0.75rem', fontSize: '0.75rem', fontWeight: 700, color: 'var(--muted-fg)', textTransform: 'uppercase' }}>
                <span>STT: {(item.latency?.stt_ms || 0).toFixed(1)}ms</span>
                <span>EMB: {(item.latency?.embedding_ms || 0).toFixed(1)}ms</span>
                <span>FAISS: {(item.latency?.vector_search_ms || 0).toFixed(2)}ms</span>
                <span>LLM: {(item.latency?.llm_ms || 0).toFixed(1)}ms</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
