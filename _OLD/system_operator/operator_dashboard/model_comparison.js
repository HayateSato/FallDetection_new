/**
 * Model Comparison — Operator Dashboard sub-page
 *
 * Fetches aggregated replay data from GET /model/comparison and renders
 * interactive Plotly charts. All charts support zoom, pan, and hover tooltips
 * via Plotly's built-in modebar.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Same base URL as app.js — always routes through nginx.
const ML_SERVER = '/api/ml';

// Consistent colour palette per model (cycles if > 6 models)
const MODEL_COLORS = ['#4a9eff', '#ff6b6b', '#6bcb77', '#ffd93d', '#c77dff', '#ff9f43'];

// Shared Plotly layout base (dark theme matching operator dashboard)
const LAYOUT_BASE = {
  paper_bgcolor: '#1a1f2e',
  plot_bgcolor:  '#0f1117',
  font:   { color: '#ccc', size: 12, family: 'system-ui, sans-serif' },
  xaxis:  { gridcolor: '#2d3448', linecolor: '#2d3448', zerolinecolor: '#2d3448' },
  yaxis:  { gridcolor: '#2d3448', linecolor: '#2d3448', zerolinecolor: '#2d3448' },
  legend: { bgcolor: 'transparent', font: { color: '#ccc' } },
  margin: { t: 20, b: 55, l: 60, r: 20 },
};

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

// Decision-boundary reference line at confidence = 0.5
const THRESHOLD_SHAPE = {
  type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5,
  line: { color: '#ef5350', dash: 'dot', width: 1 },
};
const THRESHOLD_ANNOTATION = {
  xref: 'paper', yref: 'y', x: 1, y: 0.5,
  text: 'threshold 0.5', showarrow: false,
  font: { color: '#ef5350', size: 10 }, xanchor: 'right', yanchor: 'bottom',
};

// ---------------------------------------------------------------------------
// Main load
// ---------------------------------------------------------------------------

async function loadComparison() {
  const days = document.getElementById('since-days').value;
  setStatus('meta', 'Fetching data…');

  try {
    const res = await fetch(`${ML_SERVER}/model/comparison?since_days=${days}`);
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    const data = await res.json();

    setStatus('ok');
    renderSummaryCards(data.summary, data.by_model);
    renderFallRateChart(data.by_model);
    renderLatencyChart(data.by_model);
    renderBoxPlot(data.by_model);
    renderHistogram(data.by_model);
    renderScatter(data.timeseries, data.summary.models_tested);
    renderPercentileTable(data.by_model);
    renderRecordingTable(data.by_recording, data.summary.models_tested);
    renderSessionsTable(data.recent_sessions);

  } catch (err) {
    setStatus('error', err.message);
    showEmptyCharts();
  }
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function setStatus(state, message = '') {
  const badge  = document.getElementById('server-status');
  const status = document.getElementById('load-status');

  if (state === 'ok') {
    badge.textContent  = 'Connected';
    badge.className    = 'badge badge-green';
    status.textContent = '';
  } else if (state === 'error') {
    badge.textContent  = 'Error';
    badge.className    = 'badge badge-red';
    status.textContent = message;
  } else {
    badge.textContent  = 'Loading…';
    badge.className    = 'badge badge-grey';
    status.textContent = message;
  }
}

function showEmptyCharts() {
  ['chart-fall-rate','chart-latency','chart-boxplot','chart-histogram','chart-scatter'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<p class="empty-state">No data — run a replay first.</p>';
  });
  ['percentile-table-container','recording-table-container'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<p class="empty-state">No replay data in selected period.</p>';
  });
  document.getElementById('sessions-tbody').innerHTML =
    '<tr><td colspan="8" class="loading">No sessions.</td></tr>';
}

// ---------------------------------------------------------------------------
// Summary cards
// ---------------------------------------------------------------------------

function renderSummaryCards(summary, byModel) {
  document.getElementById('stat-windows').textContent    = (summary.total_windows || 0).toLocaleString();
  document.getElementById('stat-recordings').textContent = summary.total_recordings || 0;
  document.getElementById('stat-models').textContent     = (summary.models_tested || []).length;
  const totalFalls = (byModel || []).reduce((s, m) => s + m.falls_detected, 0);
  document.getElementById('stat-falls').textContent      = totalFalls.toLocaleString();
}

// ---------------------------------------------------------------------------
// Chart 1: Fall Rate by Model (horizontal bar)
// ---------------------------------------------------------------------------

function renderFallRateChart(byModel) {
  if (!byModel || byModel.length === 0) {
    document.getElementById('chart-fall-rate').innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const maxPct = Math.max(...byModel.map(m => m.fall_rate_pct), 5);

  const traces = [{
    type:        'bar',
    orientation: 'h',
    x:    byModel.map(m => m.fall_rate_pct),
    y:    byModel.map(m => m.model_version),
    text: byModel.map(m => `${m.fall_rate_pct}%  (${m.falls_detected}/${m.total_windows})`),
    textposition: 'outside',
    marker: { color: byModel.map((_, i) => MODEL_COLORS[i % MODEL_COLORS.length]) },
    hovertemplate: '<b>%{y}</b><br>Fall rate: %{x:.1f}%<extra></extra>',
  }];

  const layout = {
    ...LAYOUT_BASE,
    margin: { t: 10, b: 50, l: 70, r: 90 },
    xaxis: {
      ...LAYOUT_BASE.xaxis,
      title: 'Fall Rate (%)',
      range: [0, maxPct * 1.35],
    },
    yaxis: { ...LAYOUT_BASE.yaxis, automargin: true },
  };

  Plotly.newPlot('chart-fall-rate', traces, layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 2: Latency (grouped bar: avg + p95)
// ---------------------------------------------------------------------------

function renderLatencyChart(byModel) {
  if (!byModel || byModel.length === 0) {
    document.getElementById('chart-latency').innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const models = byModel.map(m => m.model_version);

  const traces = [
    {
      type: 'bar', name: 'Avg latency',
      x: models,
      y: byModel.map(m => m.latency?.avg_ms),
      marker: { color: '#4a9eff', opacity: 0.85 },
      hovertemplate: '%{x}: %{y:.0f} ms avg<extra></extra>',
    },
    {
      type: 'bar', name: 'P95 latency',
      x: models,
      y: byModel.map(m => m.latency?.p95_ms),
      marker: { color: '#ff9f43', opacity: 0.85 },
      hovertemplate: '%{x}: %{y:.0f} ms p95<extra></extra>',
    },
  ];

  const layout = {
    ...LAYOUT_BASE,
    barmode: 'group',
    xaxis: { ...LAYOUT_BASE.xaxis, title: 'Model Version' },
    yaxis: { ...LAYOUT_BASE.yaxis, title: 'Latency (ms)' },
    legend: { ...LAYOUT_BASE.legend, orientation: 'h', x: 0, y: 1.12 },
    margin: { t: 40, b: 55, l: 60, r: 20 },
  };

  Plotly.newPlot('chart-latency', traces, layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 3: Box Plot — falls vs non-falls per model
// ---------------------------------------------------------------------------

function renderBoxPlot(byModel) {
  if (!byModel || byModel.length === 0) {
    document.getElementById('chart-boxplot').innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const traces = [];
  byModel.forEach((m, i) => {
    const color = MODEL_COLORS[i % MODEL_COLORS.length];

    // Falls trace
    if (m.box_data.falls.length > 0) {
      traces.push({
        type:       'box',
        name:       `${m.model_version} — falls`,
        y:          m.box_data.falls,
        boxpoints:  m.box_data.falls.length <= 300 ? 'all' : 'outliers',
        jitter:     0.3,
        marker:     { color, size: 4, opacity: 0.8 },
        line:       { color },
        fillcolor:  color + '28',
        hovertemplate: `<b>${m.model_version} fall</b><br>conf: %{y:.3f}<extra></extra>`,
      });
    }

    // Non-falls trace
    if (m.box_data.no_falls.length > 0) {
      traces.push({
        type:       'box',
        name:       `${m.model_version} — no fall`,
        y:          m.box_data.no_falls,
        boxpoints:  m.box_data.no_falls.length <= 300 ? 'all' : 'outliers',
        jitter:     0.3,
        marker:     { color, size: 3, opacity: 0.4 },
        line:       { color, dash: 'dash', width: 1.5 },
        fillcolor:  'transparent',
        hovertemplate: `<b>${m.model_version} no-fall</b><br>conf: %{y:.3f}<extra></extra>`,
      });
    }
  });

  const layout = {
    ...LAYOUT_BASE,
    yaxis: { ...LAYOUT_BASE.yaxis, title: 'Confidence Score', range: [-0.02, 1.05] },
    xaxis: { ...LAYOUT_BASE.xaxis },
    legend: { ...LAYOUT_BASE.legend, orientation: 'h', x: 0, y: 1.08 },
    margin: { t: 40, b: 30, l: 60, r: 20 },
    shapes:      [THRESHOLD_SHAPE],
    annotations: [THRESHOLD_ANNOTATION],
  };

  Plotly.newPlot('chart-boxplot', traces, layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 4: Confidence Histogram (bar overlay)
// ---------------------------------------------------------------------------

function renderHistogram(byModel) {
  if (!byModel || byModel.length === 0) {
    document.getElementById('chart-histogram').innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const traces = byModel.map((m, i) => ({
    type:         'bar',
    name:         m.model_version,
    x:            m.confidence_buckets.map(b => b.range),
    y:            m.confidence_buckets.map(b => b.count),
    marker:       { color: MODEL_COLORS[i % MODEL_COLORS.length], opacity: 0.65 },
    hovertemplate:`<b>${m.model_version}</b> [%{x}]<br>%{y} windows<extra></extra>`,
  }));

  const layout = {
    ...LAYOUT_BASE,
    barmode: 'overlay',
    xaxis:  { ...LAYOUT_BASE.xaxis, title: 'Confidence Score Range', tickangle: -35 },
    yaxis:  { ...LAYOUT_BASE.yaxis, title: 'Window Count' },
    legend: { ...LAYOUT_BASE.legend, orientation: 'h', x: 0, y: 1.1 },
    margin: { t: 40, b: 80, l: 60, r: 20 },
  };

  Plotly.newPlot('chart-histogram', traces, layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 5: Confidence Scatter — all windows coloured by fall/no-fall
// ---------------------------------------------------------------------------

function renderScatter(timeseries, models) {
  if (!timeseries || timeseries.length === 0) {
    document.getElementById('chart-scatter').innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const traces = [];

  models.forEach((mv, mi) => {
    const mvRows  = timeseries.filter(r => r.model_version === mv);
    const noFalls = mvRows.filter(r => !r.fall_detected && r.confidence !== null);
    const falls   = mvRows.filter(r =>  r.fall_detected && r.confidence !== null);
    const color   = MODEL_COLORS[mi % MODEL_COLORS.length];

    if (noFalls.length > 0) {
      traces.push({
        type:   'scatter',
        mode:   'markers',
        name:   `${mv} — no fall`,
        x:      noFalls.map((_, idx) => idx),
        y:      noFalls.map(r => r.confidence),
        marker: { color, opacity: 0.3, size: 4 },
        customdata: noFalls.map(r => [r.recording || '—', r.timestamp || '']),
        hovertemplate: `<b>${mv}</b> no-fall<br>conf: %{y:.3f}<br>file: %{customdata[0]}<extra></extra>`,
      });
    }

    if (falls.length > 0) {
      traces.push({
        type:   'scatter',
        mode:   'markers',
        name:   `${mv} — FALL`,
        x:      falls.map((_, idx) => idx),
        y:      falls.map(r => r.confidence),
        marker: { color: '#ef5350', opacity: 0.85, size: 7, symbol: 'x' },
        customdata: falls.map(r => [r.recording || '—', r.timestamp || '']),
        hovertemplate: `<b>${mv}</b> FALL<br>conf: %{y:.3f}<br>file: %{customdata[0]}<extra></extra>`,
      });
    }
  });

  const layout = {
    ...LAYOUT_BASE,
    xaxis: { ...LAYOUT_BASE.xaxis, title: 'Window index (chronological order)' },
    yaxis: { ...LAYOUT_BASE.yaxis, title: 'Confidence Score', range: [-0.02, 1.05] },
    legend: { ...LAYOUT_BASE.legend, orientation: 'h', x: 0, y: 1.06 },
    margin: { t: 40, b: 60, l: 60, r: 20 },
    shapes:      [THRESHOLD_SHAPE],
    annotations: [THRESHOLD_ANNOTATION],
  };

  Plotly.newPlot('chart-scatter', traces, layout, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Confidence Percentile Table
// ---------------------------------------------------------------------------

function renderPercentileTable(byModel) {
  const container = document.getElementById('percentile-table-container');
  if (!byModel || byModel.length === 0) {
    container.innerHTML = '<p class="empty-state">No replay data in selected period.</p>';
    return;
  }

  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Windows</th>
          <th>Fall Rate</th>
          <th>Uncertainty %</th>
          <th>P10</th>
          <th>P25</th>
          <th>Median</th>
          <th>P75</th>
          <th>P90</th>
          <th>P95</th>
          <th>Mean</th>
          <th>Std Dev</th>
        </tr>
      </thead>
      <tbody>
  `;

  byModel.forEach((m, i) => {
    const p = m.percentiles || {};
    const fallClass = m.fall_rate_pct > 30 ? 'fall-high' : m.fall_rate_pct > 10 ? 'fall-mid' : 'fall-low';
    const color = MODEL_COLORS[i % MODEL_COLORS.length];

    html += `
      <tr>
        <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span><strong>${m.model_version}</strong></td>
        <td>${m.total_windows.toLocaleString()}</td>
        <td class="${fallClass}">${m.fall_rate_pct}%</td>
        <td title="Windows with confidence 0.4–0.6">${m.uncertainty_pct ?? '—'}%</td>
        <td>${p.p10   ?? '—'}</td>
        <td>${p.p25   ?? '—'}</td>
        <td><strong>${p.p50 ?? '—'}</strong></td>
        <td>${p.p75   ?? '—'}</td>
        <td>${p.p90   ?? '—'}</td>
        <td>${p.p95   ?? '—'}</td>
        <td>${p.mean  ?? '—'}</td>
        <td>${p.stddev ?? '—'}</td>
      </tr>
    `;
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Per-Recording × Model Table
// ---------------------------------------------------------------------------

function renderRecordingTable(byRecording, models) {
  const container = document.getElementById('recording-table-container');
  if (!byRecording || byRecording.length === 0) {
    container.innerHTML = '<p class="empty-state">No recording data yet. Run replays on multiple CSV files to compare.</p>';
    return;
  }

  // Header
  let html = '<table class="data-table"><thead><tr><th>Recording</th>';
  models.forEach(mv => {
    html += `<th colspan="2" style="text-align:center;">${mv}</th>`;
  });
  html += '</tr><tr><th></th>';
  models.forEach(() => {
    html += '<th>Fall Rate</th><th>Avg Conf</th>';
  });
  html += '</tr></thead><tbody>';

  byRecording.forEach(rec => {
    // Truncate long filenames
    const name = rec.recording.length > 40
      ? '…' + rec.recording.slice(-37)
      : rec.recording;
    html += `<tr><td title="${rec.recording}" style="font-size:0.8rem;">${name}</td>`;

    models.forEach(mv => {
      const m = rec.models[mv];
      if (!m) {
        html += '<td class="no-data">—</td><td class="no-data">—</td>';
        return;
      }
      const pct = m.fall_rate_pct;
      const cls = pct > 30 ? 'fall-high' : pct > 10 ? 'fall-mid' : 'fall-low';
      html += `<td class="${cls}">${pct}%<span class="meta" style="margin-left:4px;">(${m.falls_detected}/${m.total_windows})</span></td>`;
      html += `<td>${m.avg_confidence ?? '—'}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Recent Sessions Table
// ---------------------------------------------------------------------------

function renderSessionsTable(sessions) {
  const tbody = document.getElementById('sessions-tbody');
  if (!sessions || sessions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="loading">No replay sessions yet.</td></tr>';
    return;
  }

  tbody.innerHTML = sessions.map(s => {
    const ts       = s.timestamp ? new Date(s.timestamp).toLocaleString() : '—';
    const rec      = (s.recording || '—').split('/').pop();  // basename only
    const fallCls  = s.fall_rate_pct > 30 ? 'badge-red' : s.fall_rate_pct > 10 ? 'badge-grey' : 'badge-green';

    return `<tr>
      <td class="meta">${ts}</td>
      <td style="font-size:0.82rem;" title="${s.recording || ''}">${rec}</td>
      <td><span class="badge badge-grey">${s.model_version}</span></td>
      <td>${s.total_windows}</td>
      <td>${s.falls_detected}</td>
      <td><span class="badge ${fallCls}">${s.fall_rate_pct}%</span></td>
      <td>${s.step_seconds ?? '—'}</td>
      <td class="meta">${s.resampling_method ?? '—'}</td>
    </tr>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('since-days').addEventListener('change', loadComparison);
  loadComparison();
});
