// server_health frontend — talks only to its own /api/status.

const OVERALL_TEXT = {
  healthy:  'All systems operational',
  degraded: 'Some services degraded',
  down:     'One or more services down',
};
const OVERALL_ICON = { healthy: '●', degraded: '●', down: '●' };

// Namespace + pod/deployment identifier used by `kubectl logs`.
// Postgres + MinIO are StatefulSets so we target the pod by index, not the deploy.
const NAMESPACE = 'mcs-fall-detection';
const LOG_TARGETS = {
  inference_server: 'deploy/inference-server',
  fall_dashboard:   'deploy/fall-dashboard',
  mqtt_broker:      'deploy/mqtt-broker',
  postgres:         'pod/postgres-0',
  mlflow:           'deploy/mlflow',
  minio:            'pod/minio-0',
};

function kubectlLogsCmd(serviceName) {
  const target = LOG_TARGETS[serviceName];
  if (!target) return null;
  return `kubectl logs ${target} -n ${NAMESPACE} --tail=200 -f`;
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
      const cmd = kubectlLogsCmd(s.name);
      const logsBlock = cmd ? `
        <details class="logs-details">
          <summary>View logs (kubectl)</summary>
          <div class="logs-cmd-row">
            <code class="logs-cmd">${escapeHtml(cmd)}</code>
            <button class="btn-copy" data-cmd="${escapeHtml(cmd)}">Copy</button>
          </div>
          <div class="logs-hint">Run in a terminal with cluster access. <code>-f</code> streams; drop it for a snapshot.</div>
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
