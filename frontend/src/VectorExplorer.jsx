import React, { useState } from 'react';
import { Search, Database, Sliders, Hash, ArrowUpRight, Zap, RefreshCw, FileText } from 'lucide-react';
import { getApiUrl } from './apiConfig';

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
      const res = await fetch(getApiUrl('/api/v1/search'), {
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
    <div style={{ padding: '3rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '2rem' }}>
        <div>
          <h2 className="kinetic-section-heading">VECTOR EXPLORER</h2>
          <p className="kinetic-subheading" style={{ marginTop: '0.5rem' }}>
            Perform direct FAISS 384-D dense similarity search and inspect cosine rankings.
          </p>
        </div>
        <span className="kinetic-label" style={{ border: '2px solid var(--border-zinc)', padding: '0.5rem 1rem' }}>
          FAISS INDEXFLATIP (384-D)
        </span>
      </div>

      <div className="kinetic-card" style={{ marginBottom: '2rem' }}>
        <form onSubmit={handleSearch}>
          <div style={{ marginBottom: '1.5rem' }}>
            <span className="kinetic-label">QUERY INPUT:</span>
            <input
              type="text"
              className="kinetic-input"
              placeholder="Enter query to compute cosine similarity..."
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', alignItems: 'center' }}>
            <div>
              <span className="kinetic-label">TOP K: {topK}</span>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent-blue)', marginTop: '0.5rem' }}
              />
            </div>

            <div>
              <span className="kinetic-label">THRESHOLD: {threshold}</span>
              <input
                type="range"
                min="0.05"
                max="0.80"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent-blue)', marginTop: '0.5rem' }}
              />
            </div>

            <button type="submit" className="kinetic-btn kinetic-btn-primary" disabled={loading}>
              {loading ? <RefreshCw size={18} className="spin-icon" /> : <Zap size={18} />}
              SEARCH VECTORS
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div style={{ backgroundColor: '#ff3344', color: '#fff', padding: '1rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '2rem' }}>
          ⚠️ {error}
        </div>
      )}

      {searchResult && (
        <div>
          <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem', backgroundColor: 'var(--muted-bg)', padding: '1rem 2rem', textTransform: 'uppercase', fontWeight: 700 }}>
            <span>QUERY: <strong style={{ color: 'var(--accent-blue)' }}>"{searchResult.query}"</strong></span>
            <span>HITS: <strong style={{ color: 'var(--accent-blue)' }}>{searchResult.total_hits}</strong></span>
            <span>LATENCY: <strong style={{ color: 'var(--accent-blue)' }}>{searchResult.search_latency_ms} MS</strong></span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {searchResult.hits.length === 0 ? (
              <div className="kinetic-card" style={{ textAlign: 'center', color: 'var(--muted-fg)' }}>
                NO MATCHING PASSAGES ABOVE THRESHOLD {threshold}. TRY LOWERING THRESHOLD.
              </div>
            ) : (
              searchResult.hits.map((hit) => (
                <div key={hit.rank} className="kinetic-card kinetic-card-hover">
                  <div className="kinetic-num-bg">0{hit.rank}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <span className="kinetic-label" style={{ color: 'var(--accent-blue)' }}>RANK #{hit.rank}</span>
                    <span className="kinetic-label">SCORE: {(hit.score * 100).toFixed(1)}%</span>
                  </div>
                  <p className="kinetic-card-desc" style={{ marginBottom: '1.5rem' }}>
                    "{hit.text}"
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-zinc)', paddingTop: '0.75rem' }}>
                    <span className="kinetic-label">PASSAGE #{hit.passage_id}</span>
                    {hit.url && (
                      <a href={hit.url} target="_blank" rel="noopener noreferrer" style={{ color: 'inherit', fontWeight: 700, fontSize: '0.8rem', textTransform: 'uppercase' }}>
                        LINK ↗
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
