import React, { useState, useEffect } from 'react';
import { Database, UploadCloud, FileText, Plus, CheckCircle2, AlertCircle, RefreshCw, BookOpen } from 'lucide-react';
import { getApiUrl } from './apiConfig';

export default function DatasetIngestion() {
  const [datasetsInfo, setDatasetsInfo] = useState(null);
  const [loadingInfo, setLoadingInfo] = useState(true);

  // Ingestion Form State
  const [docTitle, setDocTitle] = useState('');
  const [docText, setDocText] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  
  const [ingesting, setIngesting] = useState(false);
  const [successMsg, setSuccessMsg] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchDatasetInfo = async () => {
    setLoadingInfo(true);
    try {
      const res = await fetch(getApiUrl('/api/v1/datasets'));
      if (res.ok) {
        const data = await res.json();
        setDatasetsInfo(data);
      }
    } catch (err) {
      console.error('Error fetching datasets info:', err);
    } finally {
      setLoadingInfo(false);
    }
  };

  useEffect(() => {
    fetchDatasetInfo();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      if (!docTitle) setDocTitle(e.target.files[0].name);
    }
  };

  const handleIngest = async (e) => {
    if (e) e.preventDefault();
    if (!selectedFile && !docText.trim()) {
      setErrorMsg('Please select a file to upload or enter text content.');
      return;
    }

    setIngesting(true);
    setSuccessMsg(null);
    setErrorMsg(null);

    const formData = new FormData();
    if (selectedFile) {
      formData.append('file', selectedFile);
    }
    if (docTitle) formData.append('title', docTitle);
    if (docText) formData.append('text', docText);
    if (sourceUrl) formData.append('source_url', sourceUrl);

    try {
      const res = await fetch(getApiUrl('/api/v1/ingest'), {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (res.ok && data.status === 'success') {
        setSuccessMsg(`Ingested successfully! Added ${data.chunks_added} chunks. Total passages in index: ${data.total_index_passages}.`);
        setDocTitle('');
        setDocText('');
        setSourceUrl('');
        setSelectedFile(null);
        fetchDatasetInfo();
      } else {
        throw new Error(data.detail || 'Ingestion failed');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Error uploading document');
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div style={{ padding: '3rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '2rem' }}>
        <div>
          <h2 className="kinetic-section-heading">DATASETS & INGESTION</h2>
          <p className="kinetic-subheading" style={{ marginTop: '0.5rem' }}>
            Upload any document (.pdf, .docx, .csv, .xlsx, .json, .txt, .md) to chunk and embed into FAISS index.
          </p>
        </div>
        <button onClick={fetchDatasetInfo} className="kinetic-btn kinetic-btn-outline" style={{ padding: '0.5rem 1rem' }}>
          <RefreshCw size={16} className={loadingInfo ? 'spin-icon' : ''} /> REFRESH STATS
        </button>
      </div>

      {/* Dataset Overview Cards */}
      {datasetsInfo && (
        <div className="quick-actions-grid" style={{ marginBottom: '3rem' }}>
          <div className="kinetic-card">
            <div className="kinetic-num-bg">01</div>
            <div className="kinetic-card-title">ACTIVE DATASET</div>
            <div className="kinetic-card-desc">{datasetsInfo.dataset_name}</div>
          </div>
          <div className="kinetic-card">
            <div className="kinetic-num-bg">02</div>
            <div className="kinetic-card-title">INDEX TYPE</div>
            <div className="kinetic-card-desc" style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
              {datasetsInfo.vector_store_type} ({datasetsInfo.dimension}-D)
            </div>
          </div>
          <div className="kinetic-card">
            <div className="kinetic-num-bg">03</div>
            <div className="kinetic-card-title">INDEXED PASSAGES</div>
            <div className="kinetic-card-desc">{datasetsInfo.total_passages} Dense Vectors</div>
          </div>
        </div>
      )}

      {/* Upload Form */}
      <div className="kinetic-card" style={{ marginBottom: '3rem' }}>
        <h3 className="kinetic-card-title" style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Plus size={20} color="var(--accent-blue)" /> INGEST NEW DOCUMENT
        </h3>

        {successMsg && (
          <div style={{ backgroundColor: 'var(--accent-blue)', color: '#000', padding: '1rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '1.5rem' }}>
            ✓ {successMsg}
          </div>
        )}
        {errorMsg && (
          <div style={{ backgroundColor: '#ff3344', color: '#fff', padding: '1rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '1.5rem' }}>
            ⚠️ {errorMsg}
          </div>
        )}

        <form onSubmit={handleIngest}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '2rem', marginBottom: '1.5rem' }}>
            <div>
              <span className="kinetic-label">DOCUMENT TITLE / IDENTIFIER:</span>
              <input
                type="text"
                className="kinetic-input"
                placeholder="e.g. Goa Tourism Guide 2026"
                value={docTitle}
                onChange={(e) => setDocTitle(e.target.value)}
              />
            </div>

            <div>
              <span className="kinetic-label">SOURCE URL (OPTIONAL):</span>
              <input
                type="url"
                className="kinetic-input"
                placeholder="https://example.org/docs/goa-guide"
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
              />
            </div>
          </div>

          {/* Brutalist Drag-and-Drop Surface */}
          <div style={{ margin: '2rem 0' }}>
            <input
              type="file"
              id="file-upload"
              accept="*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label 
              htmlFor="file-upload" 
              className="kinetic-card kinetic-card-hover"
              style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center', 
                padding: '4rem 2rem', 
                cursor: 'pointer',
                textAlign: 'center'
              }}
            >
              <UploadCloud size={48} color="var(--accent-blue)" style={{ marginBottom: '1rem' }} />
              {selectedFile ? (
                <div style={{ fontSize: '1.2rem', fontWeight: 700, textTransform: 'uppercase' }}>
                  SELECTED FILE: <span style={{ color: 'var(--accent-blue)' }}>{selectedFile.name}</span> ({Math.round(selectedFile.size / 1024)} KB)
                </div>
              ) : (
                <div style={{ fontSize: '1.2rem', fontWeight: 700, textTransform: 'uppercase' }}>
                  DRAG & DROP ANY FILE HERE OR <span style={{ color: 'var(--accent-blue)' }}>CLICK TO BROWSE</span> (.PDF, .DOCX, .CSV, .XLSX, .JSON, .TXT)
                </div>
              )}
            </label>
          </div>

          <div style={{ margin: '2rem 0', textTransform: 'uppercase', fontWeight: 700, color: 'var(--muted-fg)', textAlign: 'center' }}>
            — OR PASTE RAW TEXT —
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <textarea
              className="kinetic-input"
              style={{ minHeight: '120px', resize: 'vertical' }}
              placeholder="Paste raw text or passage content to chunk & embed into FAISS..."
              value={docText}
              onChange={(e) => setDocText(e.target.value)}
            />
          </div>

          <button type="submit" className="kinetic-btn kinetic-btn-primary" style={{ width: '100%', padding: '1.25rem' }} disabled={ingesting}>
            {ingesting ? <RefreshCw size={20} className="spin-icon" /> : <Plus size={20} />}
            {ingesting ? 'CHUNKING, EMBEDDING & BUILDING INDEX...' : 'INGEST & EMBED INTO FAISS INDEX'}
          </button>
        </form>
      </div>

      {/* Dataset Sample Passages Viewer */}
      {datasetsInfo?.sample_passages && (
        <div>
          <h3 className="kinetic-section-heading" style={{ fontSize: '2rem', marginBottom: '1.5rem' }}>
            INDEXED PASSAGES ({datasetsInfo.sample_passages.length})
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {datasetsInfo.sample_passages.map((p) => (
              <div key={p.passage_id} className="kinetic-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <span className="kinetic-label" style={{ color: 'var(--accent-blue)' }}>PASSAGE #{p.passage_id}</span>
                  <span className="kinetic-label">{p.word_count} WORDS</span>
                </div>
                <p className="kinetic-card-desc">"{p.text_snippet}"</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
