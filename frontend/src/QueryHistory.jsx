import React, { useState, useEffect } from 'react';
import { History, Trash2, Clock, CheckCircle, AlertTriangle, Mic, MessageSquare, RefreshCw } from 'lucide-react';

export default function QueryHistory() {
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/history');
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
      await fetch('/api/v1/history', { method: 'DELETE' });
      setHistoryData([]);
    } catch (err) {
      alert('Failed to clear history');
    }
  };

  return (
    <div className="tab-container">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <History size={22} color="var(--accent-cyan)" /> Query Session Audit Logs
          </h2>
          <p className="section-subtitle">
            Historical audit log of voice & text queries, grounding verification status, and per-stage latency breakdown.
          </p>
        </div>
        <div className="action-button-group">
          <button onClick={fetchHistory} className="action-btn secondary">
            <RefreshCw size={14} className={loading ? 'spin-icon' : ''} /> Refresh
          </button>
          {historyData.length > 0 && (
            <button onClick={handleClearHistory} className="action-btn danger">
              <Trash2 size={14} /> Clear Logs
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="loading-container">
          <RefreshCw size={24} className="spin-icon" color="var(--accent-cyan)" />
          <p>Loading query history logs...</p>
        </div>
      ) : historyData.length === 0 ? (
        <div className="empty-state">
          <History size={36} color="var(--text-muted)" style={{ marginBottom: '0.75rem' }} />
          <p>No queries recorded in this session yet.</p>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Ask a question via voice or text in the <strong>Voice RAG Console</strong> tab to generate audit logs.
          </span>
        </div>
      ) : (
        <div className="history-table-container">
          <table className="history-table">
            <thead>
              <tr>
                <th>Time & Type</th>
                <th>Query</th>
                <th>Status & Grounding</th>
                <th>Latency Breakdown</th>
                <th>Total Latency</th>
              </tr>
            </thead>
            <tbody>
              {historyData.map((item) => (
                <tr key={item.id}>
                  <td className="time-col">
                    <span className="type-badge">
                      {item.type === 'voice' ? <Mic size={12} /> : <MessageSquare size={12} />}
                      {item.type === 'voice' ? 'Voice' : 'Text'}
                    </span>
                    <div className="timestamp">{item.timestamp}</div>
                  </td>
                  <td className="query-col">
                    <div className="query-text">"{item.query}"</div>
                    <div className="answer-snippet">{item.answer}</div>
                  </td>
                  <td className="status-col">
                    <span className={`status-tag ${item.is_grounded ? 'grounded' : 'unverified'}`}>
                      {item.is_grounded ? <CheckCircle size={12} /> : <AlertTriangle size={12} />}
                      {item.guardrail_action || (item.is_grounded ? 'PASSED' : 'UNGROUNDED')}
                    </span>
                    <div className="confidence-val">Confidence: {((item.confidence || 0.95) * 100).toFixed(0)}%</div>
                  </td>
                  <td className="latency-breakdown-col">
                    <div className="micro-timing">STT: {(item.latency?.stt_ms || 0).toFixed(1)}ms</div>
                    <div className="micro-timing">Emb: {(item.latency?.embedding_ms || 0).toFixed(1)}ms</div>
                    <div className="micro-timing">FAISS: {(item.latency?.vector_search_ms || 0).toFixed(2)}ms</div>
                    <div className="micro-timing">LLM: {(item.latency?.llm_ms || 0).toFixed(1)}ms</div>
                  </td>
                  <td className="total-col">
                    <span className="total-badge">
                      <Clock size={12} /> {(item.latency?.total_ms || 0).toFixed(1)} ms
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
