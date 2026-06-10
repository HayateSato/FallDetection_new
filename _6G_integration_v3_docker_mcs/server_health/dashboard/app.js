// server_health frontend — talks only to its own /api/status.

const OVERALL_TEXT = {
  healthy:  'All systems operational',
  degraded: 'Some services degraded',
  down:     'One or more services down',
};
const OVERALL_ICON = { healthy: '●', degraded: '●', down: '●' };

// Log commands per service.
// MCS services run in Docker Compose (docker logs <container_name>).
// Caregiver services (fall_dashboard, mqtt_broker) are in K3s on Laptop 1 (kubectl logs).
const LOG_TARGETS = {
  inference_server: { runtime: 'docker',  cmd: 'docker logs fall_inference_server --tail=200 -f' },
  postgres:         { runtime: 'docker',  cmd: 'docker logs mcs_fall_postgres --tail=200 -f' },
  mlflow:           { runtime: 'docker',  cmd: 'docker logs fall_mlflow --tail=200 -f' },
  minio:            { runtime: 'docker',  cmd: 'docker logs fall_minio --tail=200 -f' },
  fall_dashboard:   { runtime: 'kubectl', cmd: 'kubectl logs -n fall-dashboard -l app=fall-dashboard --tail=200 -f' },
  mqtt_broker:      { runtime: 'kubectl', cmd: 'kubectl logs -n fall-dashboard -l app=mosquitto --tail=200 -f' },
};

function logsEntry(serviceName) {
  return LOG_TARGETS[serviceName] || null;
}

async function copyCmd(btn, cmd) {
  try {
    await navigator.clipboard.writeText(cmd);
    const original = btn.textContent;
    btn.textContent = 'Copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = original; btn.classList.remove('copied'); }, 1200);
  } catch (e) {
    btn.textContent = 'Copy failed';
  }
}

async function refresh() {
  const overallCard = document.querySelector('.overall-card');
  const services   = document.getElementById('services');

  // Show "checking" state without rebuilding the layout
  document.getElementById('overall-text').textContent = 'Checking…';
  document.getElementById('overall-sub').textContent  = 'Probing services';

  try {
    const r = await fetch('/api/status');
    const data = await r.json();

    const overall = data.overall || 'down';
    overallCard.className = 'card overall-card ' + overall;
    document.getElementById('overall-icon').textContent  = OVERALL_ICON[overall] || '●';
    document.getElementById('overall-text').textContent  = OVERALL_TEXT[overall] || overall;
    document.getElementById('overall-sub').textContent   =
      `Last checked ${new Date().toLocaleTimeString()}  •  ${data.services.length} services`;

    services.innerHTML = data.services.map(s => {
      const entry = logsEntry(s.name);
      const logsBlock = entry ? `
        <details class="logs-details">
          <summary>View logs (${entry.runtime})</summary>
          <div class="logs-cmd-row">
            <code class="logs-cmd">${escapeHtml(entry.cmd)}</code>
            <button class="btn-copy" data-cmd="${escapeHtml(entry.cmd)}">Copy</button>
          </div>
          <div class="logs-hint">Run in a terminal with ${entry.runtime === 'kubectl' ? 'cluster (Laptop 1)' : 'Docker host (Laptop 2)'} access. <code>-f</code> streams; drop it for a snapshot.</div>
        </details>
      ` : '';
      return `
        <div class="service-cell ${s.status}">
          <div class="service-name">
            <span class="service-status-dot ${s.status}"></span>
            ${escapeHtml(s.name)}
            <span style="margin-left:auto; font-size:12px; color:#94a3b8; font-weight:400;">
              ${escapeHtml(s.status)}
            </span>
          </div>
          <div class="service-details">${escapeHtml(s.details || '—')}</div>
          <div class="service-meta">
            <span>${escapeHtml(s.url || '')}</span>
            <span>${s.latency_ms}ms</span>
          </div>
          ${logsBlock}
        </div>
      `;
    }).join('');

    // Wire copy buttons after re-render
    services.querySelectorAll('.btn-copy').forEach(btn => {
      btn.addEventListener('click', () => copyCmd(btn, btn.dataset.cmd));
    });
  } catch (e) {
    overallCard.className = 'card overall-card down';
    document.getElementById('overall-text').textContent = 'Cannot reach server_health backend';
    document.getElementById('overall-sub').textContent  = e.message;
    services.innerHTML = `<div class="loading">Failed to reach /api/status: ${escapeHtml(e.message)}</div>`;
  }
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// Initial probe + auto-refresh every 30 seconds.
// Auto-refresh is appropriate here (read-only, low cost, the whole point is
// "live status"). The Refresh button is for "I want it now."
refresh();
setInterval(refresh, 30000);
