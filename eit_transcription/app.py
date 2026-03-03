"""
=============================================================================
EIT Transcription Web App
=============================================================================
A clean web interface for researchers to:
  - Upload audio files (WAV/MP3)
  - Transcribe EIT learner responses
  - View and download results as CSV

Run with:
    pip install flask openai-whisper
    python app.py

Then open: http://localhost:5000
=============================================================================
"""

import os
import sys
import json
import time
import wave
import struct
import re
import numpy as np
from pathlib import Path
from flask import (Flask, render_template_string, request,
                   jsonify, send_file, redirect, url_for)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from transcription_pipeline import EITPipeline, AudioPreprocessor, WhisperTranscriber

# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global pipeline instance (demo mode by default)
pipeline = EITPipeline(output_dir=app.config['OUTPUT_FOLDER'], use_demo=True)

# ---------------------------------------------------------------------------
# HTML Template (single-file app)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EIT Transcription System</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #4ade80;
    --accent2: #38bdf8;
    --text: #e2e8f0;
    --muted: #64748b;
    --error: #f87171;
    --warning: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    min-height: 100vh;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    border: 1px solid var(--accent);
    padding: 4px 10px;
    border-radius: 4px;
  }
  header h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
  }
  header p { font-size: 13px; color: var(--muted); }
  main { max-width: 1100px; margin: 0 auto; padding: 40px; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
  @media(max-width:768px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
  }
  .card h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 20px;
  }

  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(74,222,128,0.04);
  }
  .drop-zone .icon { font-size: 40px; margin-bottom: 12px; }
  .drop-zone p { color: var(--muted); font-size: 14px; }
  .drop-zone strong { color: var(--accent); }
  #fileInput { display: none; }

  .file-list { margin-top: 16px; }
  .file-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .file-item .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent2); flex-shrink: 0; }
  .file-item .name { flex: 1; color: var(--text); }
  .file-item .size { color: var(--muted); }

  label { font-size: 13px; color: var(--muted); display: block; margin-bottom: 6px; }
  select, input[type=text] {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 10px 12px;
    font-size: 14px;
    font-family: 'IBM Plex Sans', sans-serif;
    margin-bottom: 16px;
    outline: none;
    transition: border-color 0.2s;
  }
  select:focus, input:focus { border-color: var(--accent); }

  .checkbox-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
  .checkbox-row input[type=checkbox] { accent-color: var(--accent); width: 16px; height: 16px; }
  .checkbox-row label { margin: 0; color: var(--text); }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .btn-primary {
    background: var(--accent);
    color: #0f1117;
    width: 100%;
    justify-content: center;
  }
  .btn-primary:hover { background: #22c55e; }
  .btn-primary:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }
  .btn-secondary {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn-secondary:hover { border-color: var(--accent2); color: var(--accent2); }

  .progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 16px;
    overflow: hidden;
    display: none;
  }
  .progress-bar .fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    transition: width 0.3s;
    width: 0%;
  }

  .status-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    margin-top: 12px;
    min-height: 20px;
  }

  /* Results Section */
  #resultsSection { display: none; margin-top: 32px; }
  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    flex-wrap: gap;
    gap: 12px;
  }
  .results-header h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  @media(max-width:600px) { .stats-row { grid-template-columns: 1fr 1fr; } }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .stat .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: var(--accent);
    display: block;
  }
  .stat .label { font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

  table { width: 100%; border-collapse: collapse; }
  thead tr { border-bottom: 1px solid var(--border); }
  th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 12px;
    text-align: left;
  }
  td { padding: 12px; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.04); }
  tr:hover td { background: rgba(255,255,255,0.02); }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 100px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .badge-green { background: rgba(74,222,128,0.15); color: var(--accent); }
  .badge-yellow { background: rgba(251,191,36,0.15); color: var(--warning); }
  .badge-red { background: rgba(248,113,113,0.15); color: var(--error); }

  .conf-bar {
    width: 80px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
    overflow: hidden;
  }
  .conf-fill { height: 100%; border-radius: 3px; }

  .flag-tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(251,191,36,0.1);
    color: var(--warning);
    margin: 1px;
  }
  .no-flags { color: var(--muted); font-size: 11px; }

  .scroll-table { overflow-x: auto; }
  .target-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
  }
  .target-met { background: rgba(74,222,128,0.15); color: var(--accent); }
  .target-not { background: rgba(248,113,113,0.15); color: var(--error); }
</style>
</head>
<body>

<header>
  <div class="logo">EIT</div>
  <div>
    <h1>Audio Transcription System</h1>
    <p>Second / Additional Language Learner Speech → Text</p>
  </div>
</header>

<main>
  <div class="grid">
    <!-- Upload Card -->
    <div class="card">
      <h2>01 — Upload Audio</h2>
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <div class="icon">🎙️</div>
        <p>Drag & drop <strong>.wav</strong> or <strong>.mp3</strong> files here</p>
        <p style="margin-top:6px;font-size:12px;">or click to browse</p>
      </div>
      <input type="file" id="fileInput" multiple accept=".wav,.mp3,.m4a,.ogg">
      <div class="file-list" id="fileList"></div>
    </div>

    <!-- Settings Card -->
    <div class="card">
      <h2>02 — Settings</h2>

      <label>Transcription Model</label>
      <select id="modelSelect">
        <option value="demo">Demo Mode (no install needed)</option>
        <option value="tiny">Whisper Tiny (fastest)</option>
        <option value="base">Whisper Base</option>
        <option value="small">Whisper Small</option>
        <option value="medium">Whisper Medium (recommended)</option>
        <option value="large">Whisper Large (most accurate)</option>
      </select>

      <label>Target Language</label>
      <select id="langSelect">
        <option value="es">Spanish (es)</option>
        <option value="en">English (en)</option>
        <option value="fr">French (fr)</option>
        <option value="de">German (de)</option>
      </select>

      <div class="checkbox-row">
        <input type="checkbox" id="removeDisfluencies" checked>
        <label for="removeDisfluencies">Remove disfluencies (um, eh, uh...)</label>
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="applyCorrections" checked>
        <label for="applyCorrections">Apply learner error corrections</label>
      </div>

      <button class="btn btn-primary" id="transcribeBtn" onclick="startTranscription()" disabled>
        ▶ Transcribe Files
      </button>
      <div class="progress-bar" id="progressBar">
        <div class="fill" id="progressFill"></div>
      </div>
      <div class="status-bar" id="statusBar">No files selected.</div>
    </div>
  </div>

  <!-- Results -->
  <div id="resultsSection">
    <div class="results-header">
      <h2>03 — Results</h2>
      <div style="display:flex;gap:10px;align-items:center;">
        <span id="targetBadge" class="target-badge"></span>
        <button class="btn btn-secondary" onclick="downloadCSV()">⬇ Download CSV</button>
        <button class="btn btn-secondary" onclick="downloadJSON()">⬇ Download JSON</button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat">
        <span class="value" id="statFiles">—</span>
        <div class="label">Files Processed</div>
      </div>
      <div class="stat">
        <span class="value" id="statConf">—</span>
        <div class="label">Avg Confidence</div>
      </div>
      <div class="stat">
        <span class="value" id="statAgreement">—</span>
        <div class="label">Word Agreement</div>
      </div>
      <div class="stat">
        <span class="value" id="statFlagged">—</span>
        <div class="label">Flagged Files</div>
      </div>
    </div>

    <div class="card">
      <div class="scroll-table">
        <table>
          <thead>
            <tr>
              <th>Learner ID</th>
              <th>Sentence</th>
              <th>Transcription</th>
              <th>Confidence</th>
              <th>Duration</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody id="resultsTable"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>

<script>
let selectedFiles = [];
let resultsData = [];

// File selection
document.getElementById('fileInput').addEventListener('change', e => {
  handleFiles(Array.from(e.target.files));
});

const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(Array.from(e.dataTransfer.files));
});

function handleFiles(files) {
  selectedFiles = files.filter(f => f.name.match(/\.(wav|mp3|m4a|ogg)$/i));
  const list = document.getElementById('fileList');
  list.innerHTML = selectedFiles.map(f => `
    <div class="file-item">
      <div class="dot"></div>
      <span class="name">${f.name}</span>
      <span class="size">${(f.size/1024).toFixed(1)} KB</span>
    </div>
  `).join('');
  document.getElementById('transcribeBtn').disabled = selectedFiles.length === 0;
  document.getElementById('statusBar').textContent = 
    selectedFiles.length > 0 ? `${selectedFiles.length} file(s) ready.` : 'No files selected.';
}

async function startTranscription() {
  if (!selectedFiles.length) return;

  document.getElementById('transcribeBtn').disabled = true;
  document.getElementById('progressBar').style.display = 'block';
  const statusBar = document.getElementById('statusBar');
  const fill = document.getElementById('progressFill');
  const model = document.getElementById('modelSelect').value;
  const lang = document.getElementById('langSelect').value;
  const removeDis = document.getElementById('removeDisfluencies').checked;
  const applyCorr = document.getElementById('applyCorrections').checked;

  resultsData = [];
  let processed = 0;

  for (const file of selectedFiles) {
    statusBar.textContent = `Processing: ${file.name} (${processed+1}/${selectedFiles.length})`;
    const pct = Math.round((processed / selectedFiles.length) * 100);
    fill.style.width = pct + '%';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('language', lang);
    formData.append('remove_disfluencies', removeDis);
    formData.append('apply_corrections', applyCorr);

    try {
      const response = await fetch('/transcribe', { method: 'POST', body: formData });
      const result = await response.json();
      resultsData.push(result);
    } catch(err) {
      resultsData.push({ error: err.message, audio_file: file.name });
    }
    processed++;
  }

  fill.style.width = '100%';
  statusBar.textContent = `✓ Done! ${processed} file(s) transcribed.`;
  document.getElementById('transcribeBtn').disabled = false;
  showResults();
}

function showResults() {
  const section = document.getElementById('resultsSection');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth' });

  const valid = resultsData.filter(r => !r.error);
  const avgConf = valid.reduce((s,r) => s + r.confidence_score, 0) / (valid.length || 1);
  const flagged = valid.filter(r => r.flags && r.flags.length > 0).length;
  const agreements = valid.filter(r => r.word_agreement !== undefined).map(r => r.word_agreement);
  const avgAgreement = agreements.length > 0 
    ? (agreements.reduce((a,b) => a+b, 0) / agreements.length) 
    : null;

  document.getElementById('statFiles').textContent = valid.length;
  document.getElementById('statConf').textContent = (avgConf * 100).toFixed(0) + '%';
  document.getElementById('statAgreement').textContent = 
    avgAgreement !== null ? (avgAgreement * 100).toFixed(0) + '%' : 'N/A';
  document.getElementById('statFlagged').textContent = flagged;

  const badge = document.getElementById('targetBadge');
  if (avgAgreement !== null) {
    badge.className = 'target-badge ' + (avgAgreement >= 0.9 ? 'target-met' : 'target-not');
    badge.textContent = avgAgreement >= 0.9 ? '✓ 90% Target Met' : '✗ Below 90% Target';
  }

  const tbody = document.getElementById('resultsTable');
  tbody.innerHTML = valid.map(r => {
    const conf = r.confidence_score;
    const confColor = conf >= 0.8 ? '#4ade80' : conf >= 0.65 ? '#fbbf24' : '#f87171';
    const flags = r.flags && r.flags.length > 0 
      ? r.flags.map(f => `<span class="flag-tag">${f}</span>`).join(' ')
      : '<span class="no-flags">—</span>';
    const lid = r.learner_id || r.audio_file?.split('/').pop()?.split('_')[0] || '—';
    const sid = r.sentence_id?.replace(lid + '_', '') || '—';

    return `<tr>
      <td><code style="font-family:IBM Plex Mono,monospace;font-size:12px;">${lid}</code></td>
      <td style="color:#94a3b8;font-size:12px;">${sid}</td>
      <td>${r.corrected_transcription || '—'}</td>
      <td>
        <span class="conf-bar"><span class="conf-fill" style="width:${conf*100}%;background:${confColor}"></span></span>
        <code style="font-size:12px;">${(conf*100).toFixed(0)}%</code>
      </td>
      <td style="color:#94a3b8;">${r.duration_seconds?.toFixed(1) || '—'}s</td>
      <td>${flags}</td>
    </tr>`;
  }).join('');
}

function downloadCSV() {
  if (!resultsData.length) return;
  const headers = ['learner_id','sentence_id','audio_file','raw_transcription',
                   'corrected_transcription','confidence_score','duration_seconds',
                   'processing_time_ms','flags'];
  const rows = resultsData.map(r =>
    headers.map(h => {
      const v = h === 'flags' ? (r[h] || []).join('; ') : (r[h] ?? '');
      return `"${String(v).replace(/"/g,'""')}"`;
    }).join(',')
  );
  const csv = [headers.join(','), ...rows].join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'eit_transcriptions.csv';
  a.click();
}

function downloadJSON() {
  if (!resultsData.length) return;
  const blob = new Blob([JSON.stringify(resultsData, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'eit_transcriptions.json';
  a.click();
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    use_demo = request.form.get('model', 'demo') == 'demo'
    remove_dis = request.form.get('remove_disfluencies', 'true').lower() == 'true'
    apply_corr = request.form.get('apply_corrections', 'true').lower() == 'true'

    # Save uploaded file
    safe_name = re.sub(r'[^\w.\-]', '_', f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    f.save(save_path)

    # Run pipeline
    pipe = EITPipeline(
        output_dir=app.config['OUTPUT_FOLDER'],
        use_demo=use_demo,
        whisper_model=request.form.get('model', 'medium')
    )
    pipe.postprocessor.remove_disfluencies = remove_dis

    match = re.search(r'sentence_(\d+)', safe_name)
    sidx = int(match.group(1)) - 1 if match else None

    result = pipe.process_file(save_path, sentence_index=sidx)

    from dataclasses import asdict
    r = asdict(result)

    # Compute word agreement if possible
    refs = WhisperTranscriber.EIT_REFERENCE_SENTENCES
    if sidx is not None and 0 <= sidx < len(refs):
        r['word_agreement'] = pipe.postprocessor.compute_agreement(
            result.corrected_transcription, refs[sidx]
        )

    return jsonify(r)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'mode': 'demo'})


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  EIT Transcription Web App")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
