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
    <div className="tab-container">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Database size={22} color="var(--accent-cyan)" /> Datasets & Document Ingestion
          </h2>
          <p className="section-subtitle">
            Upload any document (PDF, Word, CSV, Excel, JSON, Markdown, TXT, HTML) to chunk, embed via e5-small, and update your local FAISS index dynamically.
          </p>
        </div>
        <button onClick={fetchDatasetInfo} className="action-btn secondary">
          <RefreshCw size={14} className={loadingInfo ? 'spin-icon' : ''} /> Refresh Stats
        </button>
      </div>

      {/* Dataset Overview Cards */}
      {datasetsInfo && (
        <div className="dataset-stats-grid">
          <div className="stat-card">
            <div className="stat-label">Active Dataset</div>
            <div className="stat-value">{datasetsInfo.dataset_name}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Index Type & Dim</div>
            <div className="stat-value highlight">{datasetsInfo.vector_store_type} ({datasetsInfo.dimension}-D)</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Indexed Passages</div>
            <div className="stat-value">{datasetsInfo.total_passages} vectors</div>
          </div>
        </div>
      )}

      {/* Upload & Text Ingestion Form */}
      <div className="ingest-card">
        <h3 className="card-subtitle"><Plus size={16} /> Ingest New Document / Knowledge Passage</h3>

        {successMsg && (
          <div className="success-banner">
            <CheckCircle2 size={18} /> {successMsg}
          </div>
        )}
        {errorMsg && (
          <div className="error-banner">
            <AlertCircle size={18} /> {errorMsg}
          </div>
        )}

        <form onSubmit={handleIngest} className="ingest-form">
          <div className="form-grid">
            <div className="form-group">
              <label>Document Title / Identifier</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Goa Tourism Guide 2026"
                value={docTitle}
                onChange={(e) => setDocTitle(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Source URL (Optional Attribution)</label>
              <input
                type="url"
                className="form-input"
                placeholder="https://example.org/docs/goa-guide"
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
              />
            </div>
          </div>

          <div className="upload-dropzone">
            <input
              type="file"
              id="file-upload"
              accept="*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-upload" className="dropzone-label">
              <UploadCloud size={32} color="var(--accent-cyan)" />
              {selectedFile ? (
                <span className="file-name">Selected: <strong>{selectedFile.name}</strong> ({Math.round(selectedFile.size / 1024)} KB)</span>
              ) : (
                <span>Drag & drop any data file here, or <strong>click to browse</strong> (.pdf, .docx, .csv, .xlsx, .json, .txt, .md)</span>
              )}
            </label>
          </div>

          <div className="or-divider"><span>OR PASTE RAW TEXT</span></div>

          <div className="form-group">
            <textarea
              className="form-textarea"
              rows={4}
              placeholder="Paste raw text or passage content to chunk & embed into FAISS..."
              value={docText}
              onChange={(e) => setDocText(e.target.value)}
            />
          </div>

          <button type="submit" className="action-btn primary large" disabled={ingesting}>
            {ingesting ? <RefreshCw size={18} className="spin-icon" /> : <Plus size={18} />}
            {ingesting ? 'Chunking, Embedding & Building Index...' : 'Ingest & Embed into FAISS Index'}
          </button>
        </form>
      </div>

      {/* Dataset Sample Passages Viewer */}
      {datasetsInfo?.sample_passages && (
        <div className="passages-viewer">
          <h3 className="card-subtitle"><BookOpen size={16} /> Sample Passages in Active Index ({datasetsInfo.sample_passages.length})</h3>
          <div className="passages-list">
            {datasetsInfo.sample_passages.map((p) => (
              <div key={p.passage_id} className="passage-item">
                <div className="passage-header">
                  <span className="passage-id"><FileText size={13} /> {p.passage_id}</span>
                  <span className="word-count">{p.word_count} words</span>
                </div>
                <div className="passage-text">{p.text_snippet}</div>
                {p.url && <div className="passage-url">{p.url}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
