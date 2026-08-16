import React, { useState } from 'react';
import { Search, Database, Sliders, Hash, ArrowUpRight, Zap, RefreshCw, FileText } from 'lucide-react';

export default function VectorExplorer({ settings }) {
  const [queryText, setQueryText] = useState('what is retrieval augmented generation');
  const [topK, setTopK] = useState(settings?.top_k || 4);
  const [threshold, setThreshold] = useState(settings?.score_threshold || 0.10);
  const [loading, setLoading] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!queryText.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/v1/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText.trim(),
          top_k: Number(topK),
          score_threshold: Number(threshold)
        })
      });

      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setSearchResult(data);
    } catch (err) {
      setError(err.message || 'Failed to perform vector search');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-container">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Search size={22} color="var(--accent-cyan)" /> Vector Search Explorer
          </h2>
          <p className="section-subtitle">
            Perform direct FAISS dense embedding similarity queries and inspect raw vector score rankings.
          </p>
        </div>
        <div className="badge-pill">
          <Database size={13} /> FAISS IndexFlatIP (384-dim)
        </div>
      </div>

      <div className="explorer-card">
        <form onSubmit={handleSearch} className="explorer-form">
          <div className="input-group flex-1">
            <Search size={18} className="search-icon" />
            <input
              type="text"
              className="explorer-input"
              placeholder="Enter search query to test FAISS cosine similarity..."
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
            />
          </div>

          <div className="controls-row">
            <div className="control-item">
              <label><Sliders size={13} /> Top K: {topK}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>

            <div className="control-item">
              <label><Hash size={13} /> Threshold: {threshold}</label>
              <input
                type="range"
                min="0.05"
                max="0.80"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </div>

            <button type="submit" className="action-btn primary" disabled={loading}>
              {loading ? <RefreshCw size={16} className="spin-icon" /> : <Zap size={16} />}
              Search Vectors
            </button>
          </div>
        </form>
      </div>

      {error && <div className="error-banner">⚠️ {error}</div>}

      {searchResult && (
        <div className="results-wrapper">
          <div className="metrics-bar">
            <span>Query: <strong>"{searchResult.query}"</strong></span>
            <span>Hits Found: <strong>{searchResult.total_hits}</strong></span>
            <span>Search Latency: <strong className="highlight-val">{searchResult.search_latency_ms} ms</strong></span>
          </div>

          <div className="hits-grid">
            {searchResult.hits.length === 0 ? (
              <div className="empty-state">
                No matching passages found above score threshold {threshold}. Try lowering threshold or changing query.
              </div>
            ) : (
              searchResult.hits.map((hit) => (
                <div key={hit.rank} className="hit-card">
                  <div className="hit-card-header">
                    <span className="hit-rank">#Rank {hit.rank}</span>
                    <span className="hit-score">Cosine Similarity: <strong>{(hit.score * 100).toFixed(1)}%</strong> ({hit.score})</span>
                  </div>
                  <div className="hit-body">
                    <FileText size={15} color="var(--accent-cyan)" style={{ float: 'left', marginRight: '8px', marginTop: '3px' }} />
                    <p className="hit-text">{hit.text}</p>
                  </div>
                  <div className="hit-footer">
                    <span className="passage-tag">Passage ID: {hit.passage_id}</span>
                    {hit.url && (
                      <a href={hit.url} target="_blank" rel="noopener noreferrer" className="hit-link">
                        Source Link <ArrowUpRight size={13} />
                      </a>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
