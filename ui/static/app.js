/* ==========================================================================
   AdaptiveFlow — Frontend Logic
   ========================================================================== */

const API = '';
let lastPrompt = '';
let lastModelUsed = '';

function formatModel(name) {
  return (name || '').replace('openrouter/', '').replace('ollama/', 'local/');
}

/* ── Toast Notifications ── */
function showToast(message) {
  const toast = document.getElementById('toastNotice');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('visible');
  setTimeout(() => toast.classList.remove('visible'), 3000);
}

/* ── Tab Navigation ── */
function switchTab(tabId) {
  document.querySelectorAll('.tab-link').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${tabId}`);
  });

  if (tabId === 'models') loadModelStats();
  if (tabId === 'profile') loadProfile();
}

document.querySelectorAll('.tab-link').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

/* ── Health Check ── */
async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      dot.className = 'status-dot online';
      text.textContent = 'online';
    } else {
      throw new Error();
    }
  } catch {
    dot.className = 'status-dot offline';
    text.textContent = 'offline';
  }
}
checkHealth();
setInterval(checkHealth, 15000);

/* ── Rate Limit Indicator ── */
async function refreshRateLimit(userId) {
  try {
    const res = await fetch(`${API}/ratelimit/${encodeURIComponent(userId)}`);
    if (!res.ok) return;
    const data = await res.json();

    const limit = data.limit_per_minute;
    const used = data.current_window_usage;
    const tier = data.tier;

    document.getElementById('activeTierText').textContent = tier;
    if (limit === null) {
      document.getElementById('rateLimitCount').textContent = `${used} / ∞`;
      document.getElementById('rateLimitMeter').style.width = '10%';
    } else {
      document.getElementById('rateLimitCount').textContent = `${used} / ${limit}`;
      const pct = Math.min(100, Math.round((used / limit) * 100));
      const meter = document.getElementById('rateLimitMeter');
      meter.style.width = `${pct}%`;
      meter.style.backgroundColor = pct >= 90 ? 'var(--colors-error)' : pct >= 60 ? 'var(--colors-warning)' : 'var(--colors-primary)';
    }
  } catch (e) {
    console.debug('Rate limit query failed', e);
  }
}

/* ── Chat Implementation ── */
async function sendPrompt() {
  const userId = document.getElementById('userId').value.trim() || 'user_001';
  const promptInput = document.getElementById('promptInput');
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  const mode = document.getElementById('routeModeSelect').value;
  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;

  appendChatMessage('user', prompt);
  promptInput.value = '';
  promptInput.style.height = 'auto';

  const typingId = appendTypingIndicator();

  try {
    let endpoint = '/route';
    let body = { user_id: userId, prompt };

    if (mode === 'parallel_fastest') {
      endpoint = '/route/parallel';
      body = { user_id: userId, prompt, strategy: 'fastest' };
    } else if (mode === 'parallel_all') {
      endpoint = '/route/parallel';
      body = { user_id: userId, prompt, strategy: 'all' };
    } else if (mode === 'synthesize') {
      endpoint = '/route/synthesize';
      body = { user_id: userId, prompt };
    }

    const res = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    removeTypingIndicator(typingId);

    if (!res.ok) {
      const err = await res.json();
      appendChatMessage('assistant', `⚠️ ${err.detail || 'Request rejected'}`, null);
      showToast(err.detail || 'Request error');
    } else {
      const data = await res.json();
      appendChatMessage('assistant', data.response, data, prompt);
      updateTurnTelemetry(data);
      lastPrompt = prompt;
      lastModelUsed = data.model_used;
      document.getElementById('feedbackSection').style.display = 'block';
    }
  } catch (e) {
    removeTypingIndicator(typingId);
    appendChatMessage('assistant', `⚠️ Network error: ${e.message}`, null);
  } finally {
    sendBtn.disabled = false;
    refreshRateLimit(userId);
  }
}

function appendChatMessage(role, text, meta = null, originalPrompt = '') {
  const feed = document.getElementById('chatMessages');
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = `msg-wrapper ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (meta && role === 'assistant') {
    const metaBar = document.createElement('div');
    metaBar.className = 'msg-meta-bar';

    const shortModel = formatModel(meta.model_used || 'synthesized');
    const costFormatted = meta.cost === 0 ? 'free' : `$${meta.cost.toFixed(5)}`;
    const latencyFormatted = `${meta.latency.toFixed(2)}s`;

    let complexityBadge = '';
    if (meta.complexity) {
      const comp = meta.complexity;
      const compClass = comp.is_simple ? 'badge-complexity-simple' : 'badge-complexity-complex';
      const label = comp.is_simple ? `⚡ simple (${comp.score.toFixed(2)})` : `🧠 complex (${comp.score.toFixed(2)})`;
      complexityBadge = `<span class="badge ${compClass}">${label}</span>`;
    }

    metaBar.innerHTML = `
      <span class="badge badge-model">${shortModel}</span>
      <span class="badge badge-task">${meta.task_type || 'general'}</span>
      ${complexityBadge}
      ${meta.cached ? '<span class="badge badge-cached">cached</span>' : ''}
      <span class="badge">${costFormatted}</span>
      <span class="badge">${latencyFormatted}</span>
      <button class="explain-link-btn" onclick="openExplainWithPrompt('${encodeURIComponent(originalPrompt)}')">Explain Route</button>
    `;
    wrap.appendChild(metaBar);
  }

  feed.appendChild(wrap);
  feed.scrollTop = feed.scrollHeight;
}

function appendTypingIndicator() {
  const feed = document.getElementById('chatMessages');
  const id = 'typing-' + Date.now();
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrapper assistant';
  wrap.id = id;
  wrap.innerHTML = `<div class="msg-bubble" style="color:var(--colors-mute); font-style:italic;">thinking & routing...</div>`;
  feed.appendChild(wrap);
  feed.scrollTop = feed.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function updateTurnTelemetry(data) {
  document.getElementById('metaModel').textContent = data.model_used || '—';
  document.getElementById('metaTask').textContent = data.task_type || '—';
  if (data.complexity) {
    document.getElementById('metaComplexity').textContent = `${data.complexity.verdict} (${data.complexity.score.toFixed(2)})`;
  } else {
    document.getElementById('metaComplexity').textContent = '—';
  }
  document.getElementById('metaCost').textContent = data.cost === 0 ? 'free ($0.0)' : `$${data.cost.toFixed(6)}`;
  document.getElementById('metaLatency').textContent = `${data.latency.toFixed(2)}s`;
  document.getElementById('metaCached').textContent = data.cached ? '✅ Hit' : '❌ Miss';
}

function useSamplePrompt(text) {
  document.getElementById('promptInput').value = text;
  sendPrompt();
}

/* ── Feedback ── */
async function submitFeedback(rating) {
  if (!lastPrompt || !lastModelUsed) return;
  const userId = document.getElementById('userId').value.trim() || 'user_001';
  try {
    await fetch(`${API}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        prompt: lastPrompt,
        model_used: lastModelUsed,
        rating: rating,
      }),
    });
    showToast(rating === 1 ? '👍 Positive feedback recorded' : '👎 Negative penalty recorded');
    document.getElementById('feedbackSection').style.display = 'none';
  } catch (e) {
    showToast('Failed to record feedback');
  }
}

/* ── Auto-Grow Textarea & Enter to Send ── */
const promptInput = document.getElementById('promptInput');
if (promptInput) {
  promptInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });
  promptInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });
}

/* ── Routing Explainability Studio ── */
function openExplainWithPrompt(encodedPrompt) {
  const prompt = decodeURIComponent(encodedPrompt);
  document.getElementById('explainPrompt').value = prompt;
  switchTab('explain');
  runExplain();
}

async function runExplain() {
  const prompt = document.getElementById('explainPrompt').value.trim();
  const userId = document.getElementById('explainUserId').value.trim() || 'user_001';
  if (!prompt) return;

  try {
    const res = await fetch(`${API}/route/explain?user_id=${encodeURIComponent(userId)}&prompt=${encodeURIComponent(prompt)}`);
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || 'Explain query failed');
      return;
    }
    const data = await res.json();

    // Show containers
    document.getElementById('explainMetricsGrid').style.display = 'grid';
    document.getElementById('explainDetailsSection').style.display = 'grid';

    // Populate Key Metrics
    document.getElementById('expTaskClass').textContent = data.predicted_task_class;
    document.getElementById('expTaskConfidence').textContent = `Confidence: ${(data.classification_confidence || 1.0).toFixed(2)}`;

    if (data.prompt_complexity) {
      const comp = data.prompt_complexity;
      document.getElementById('expComplexityScore').textContent = comp.score.toFixed(2);
      document.getElementById('expComplexityVerdict').textContent = comp.verdict;
    } else {
      document.getElementById('expComplexityScore').textContent = '—';
    }

    const topModel = data.top_3_candidate_models[0] ? data.top_3_candidate_models[0].model : 'None';
    document.getElementById('expSelectedModel').textContent = formatModel(topModel);
    document.getElementById('expEstimatedCost').textContent = `Est. Cost: $${data.estimated_cost.toFixed(6)}`;

    // Cache preview
    if (data.cache_prediction && data.cache_prediction.will_hit) {
      document.getElementById('expCacheStatus').textContent = 'HIT';
      document.getElementById('expCacheType').textContent = `Type: ${data.cache_prediction.cache_type}`;
    } else {
      document.getElementById('expCacheStatus').textContent = 'MISS';
      document.getElementById('expCacheType').textContent = 'Fresh model inference';
    }

    // Populate Candidates Table
    const tbody = document.getElementById('candidatesTableBody');
    tbody.innerHTML = data.top_3_candidate_models.map((cand, idx) => {
      const prior = cand.preference_prior > 0 ? `+${cand.preference_prior.toFixed(2)}` : cand.preference_prior.toFixed(2);
      const priorClass = cand.preference_prior > 0 ? 'style="color:var(--colors-emerald); font-weight:600;"' : cand.preference_prior < 0 ? 'style="color:var(--colors-error); font-weight:600;"' : '';
      const cbStatus = cand.circuit_breaker || 'CLOSED';
      const cbBadge = `<span class="cb-badge cb-${cbStatus.toLowerCase()}">${cbStatus}</span>`;

      return `
        <tr>
          <td><strong>#${idx + 1}</strong></td>
          <td style="font-family:var(--font-mono); font-weight:500;">${cand.model}</td>
          <td style="font-family:var(--font-mono);">${cand.task_q_value.toFixed(3)}</td>
          <td ${priorClass}>${prior}</td>
          <td>${cand.historical_avg_latency.toFixed(2)}s</td>
          <td>${cbBadge}</td>
        </tr>
      `;
    }).join('');

    // Populate Reasoning
    document.getElementById('expReasoningText').textContent = data.selection_reasoning;

    // Populate Complexity breakdown
    const compTable = document.getElementById('complexityDimensionsTable');
    if (data.prompt_complexity && data.prompt_complexity.dimensions) {
      const dims = data.prompt_complexity.dimensions;
      compTable.innerHTML = `
        <div class="telemetry-row"><span class="telemetry-label">Tokens Score (weight 0.25)</span><span class="telemetry-val">${dims.tokens_score.toFixed(3)}</span></div>
        <div class="telemetry-row"><span class="telemetry-label">Entropy & TTR (weight 0.35)</span><span class="telemetry-val">${dims.entropy_score.toFixed(3)}</span></div>
        <div class="telemetry-row"><span class="telemetry-label">Question Depth (weight 0.40)</span><span class="telemetry-val">${dims.depth_score.toFixed(3)}</span></div>
        <div class="telemetry-row"><span class="telemetry-label">Target Routing Arm</span><span class="telemetry-val">${data.prompt_complexity.is_simple ? 'Cheapest Model Bypass' : 'RL Bandit Exploration'}</span></div>
      `;
    }
  } catch (e) {
    showToast(`Explain error: ${e.message}`);
  }
}

/* ── Models Catalog & Warmup ── */
async function loadModelStats() {
  const container = document.getElementById('modelsGrid');
  container.innerHTML = '<div style="color:var(--colors-mute); padding:20px;">Fetching live models and circuit breakers...</div>';

  try {
    const res = await fetch(`${API}/models/stats`);
    if (!res.ok) throw new Error('Failed to fetch stats');
    const data = await res.json();

    container.innerHTML = data.map(m => {
      const short = formatModel(m.model);
      const cbStatus = m.circuit_breaker || 'CLOSED';
      const cbBadge = `<span class="cb-badge cb-${cbStatus.toLowerCase()}">${cbStatus}</span>`;

      // Fallback display
      const fallback = m.fallback ? formatModel(m.fallback) : 'None';

      return `
        <div class="model-card">
          <div class="model-card-header">
            <div>
              <div class="model-title">${short}</div>
              <div class="model-meta-sub">${m.model}</div>
            </div>
            ${cbBadge}
          </div>

          <div class="q-values-grid">
            <div class="q-slot"><span class="q-slot-key">Global Q</span><span class="q-slot-val">${m.q_value.toFixed(3)}</span></div>
            <div class="q-slot"><span class="q-slot-key">Selections</span><span class="q-slot-val">${m.selection_count}</span></div>
            <div class="q-slot"><span class="q-slot-key">Avg Latency</span><span class="q-slot-val">${m.avg_latency.toFixed(2)}s</span></div>
            <div class="q-slot"><span class="q-slot-key">Fallback</span><span class="q-slot-val" style="color:var(--colors-mute);">${fallback}</span></div>
          </div>

          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:auto; pt:8px;">
            <span style="font-size:11px; font-family:var(--font-mono); color:var(--colors-mute);">
              Cost: ${m.avg_cost === 0 ? 'Free' : '$' + m.avg_cost.toFixed(5)}
            </span>
            <button class="btn-sm-secondary" onclick="warmupModel('${m.model}')">Warm Up Benchmark</button>
          </div>
        </div>
      `;
    }).join('');
  } catch (e) {
    container.innerHTML = `<div style="color:var(--colors-error); padding:20px;">Failed to load models: ${e.message}</div>`;
  }
}

async function warmupModel(model) {
  const adminKey = document.getElementById('adminSecretKey') ? document.getElementById('adminSecretKey').value : 'admin-secret-key';
  showToast(`Running synthetic benchmarks for ${model}...`);
  try {
    const res = await fetch(`${API}/admin/models/warmup?model=${encodeURIComponent(model)}`, {
      method: 'POST',
      headers: { 'X-Admin-Key': adminKey },
    });
    if (!res.ok) throw new Error('Warmup unauthorized or failed');
    const data = await res.json();
    showToast(`✅ ${model} warmed up: ${data.benchmarks_run} benchmarks, new Q=${data.new_q_value.toFixed(3)}`);
    loadModelStats();
  } catch (e) {
    showToast(`Warmup error: ${e.message}`);
  }
}

/* ── Profiles & Routing Hints ── */
async function loadProfile() {
  const userId = document.getElementById('profileLookupId').value.trim();
  if (!userId) return;

  try {
    const res = await fetch(`${API}/profile/${encodeURIComponent(userId)}`);
    if (!res.ok) throw new Error('User not found');
    const data = await res.json();

    document.getElementById('profileDetails').style.display = 'block';
    document.getElementById('profTier').textContent = data.tier;
    document.getElementById('profTotalCost').textContent = `$${data.total_cost.toFixed(4)}`;
    document.getElementById('profRequestCount').textContent = data.request_count;
    document.getElementById('profBudget').textContent = data.budget_remaining === null ? '∞' : `$${data.budget_remaining.toFixed(2)}`;

    // Set preference hints
    if (data.preferences) {
      document.getElementById('prefSpecialization').value = data.preferences.prefer || '';
      document.getElementById('prefAvoid').value = (data.preferences.avoid || []).join(', ');
    }
  } catch (e) {
    showToast(e.message);
  }
}

async function savePreferences() {
  const userId = document.getElementById('profileLookupId').value.trim();
  if (!userId) {
    showToast('Please enter a User ID first');
    return;
  }

  const prefer = document.getElementById('prefSpecialization').value;
  const avoidRaw = document.getElementById('prefAvoid').value;
  const avoid = avoidRaw.split(',').map(s => s.trim().toLowerCase()).filter(Boolean);

  try {
    const res = await fetch(`${API}/profile/${encodeURIComponent(userId)}/preferences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prefer: prefer || null, avoid }),
    });
    if (!res.ok) throw new Error('Failed to save preferences');
    showToast(`✅ Saved routing hints for ${userId}`);
    loadProfile();
  } catch (e) {
    showToast(`Error: ${e.message}`);
  }
}

/* ── Compliance & Admin ── */
function getAdminKey() {
  return document.getElementById('adminSecretKey').value.trim();
}

async function loadSecurityAuditLogs() {
  const tbody = document.getElementById('auditTableBody');
  tbody.innerHTML = '<tr><td colspan="5" style="color:var(--colors-mute);">Fetching audit events from SQLite...</td></tr>';

  try {
    const res = await fetch(`${API}/admin/security/audit?limit=50`, {
      headers: { 'X-Admin-Key': getAdminKey() },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} Unauthorized`);
    const logs = await res.json();

    if (!logs || logs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--colors-mute);">No security threats logged yet. System is secure.</td></tr>';
      return;
    }

    tbody.innerHTML = logs.map(row => `
      <tr>
        <td style="font-family:var(--font-mono); font-size:11px;">${row.timestamp}</td>
        <td style="font-weight:500;">${row.user_id}</td>
        <td><span class="badge" style="color:var(--colors-error); background:#fff0f0; border-color:#fecaca;">${row.threat_type}</span></td>
        <td style="font-family:var(--font-mono); font-size:11px;">${row.matched_pattern}</td>
        <td style="font-size:12px; max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${row.prompt_snippet}">${row.prompt_snippet}</td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--colors-error);">${e.message}</td></tr>`;
  }
}

async function fetchAdminJson(endpoint, titleText) {
  const card = document.getElementById('adminOutputCard');
  const title = document.getElementById('adminOutputTitle');
  const pre = document.getElementById('adminOutputText');
  card.style.display = 'block';
  title.textContent = titleText;
  pre.textContent = 'Loading...';

  try {
    const res = await fetch(`${API}${endpoint}`, { headers: { 'X-Admin-Key': getAdminKey() } });
    if (!res.ok) throw new Error(`HTTP ${res.status} Unauthorized`);
    const data = await res.json();
    pre.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    pre.textContent = e.message;
  }
}

async function loadAllUserCosts() {
  return fetchAdminJson('/admin/costs', 'Accrued Costs per User (GET /admin/costs)');
}

async function loadRLWeights() {
  return fetchAdminJson('/admin/rl/stats', 'Global RL State & Weights (GET /admin/rl/stats)');
}

async function adminSetTier() {
  const userId = document.getElementById('adminTierUserId').value.trim();
  const tier = document.getElementById('adminTierSelect').value;
  const budget = document.getElementById('adminTierBudget').value;
  if (!userId) {
    showToast('Target User ID required');
    return;
  }

  let url = `${API}/admin/users/${encodeURIComponent(userId)}/tier?tier=${tier}`;
  if (budget) url += `&budget=${budget}`;

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'X-Admin-Key': getAdminKey() },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    showToast(`✅ ${data.user_id} tier updated to ${data.tier}`);
  } catch (e) {
    showToast(`Error updating tier: ${e.message}`);
  }
}

async function adminFlagUser() {
  const userId = document.getElementById('adminFlagUserId').value.trim();
  if (!userId) {
    showToast('User ID required');
    return;
  }

  try {
    const res = await fetch(`${API}/admin/users/${encodeURIComponent(userId)}/flag`, {
      method: 'POST',
      headers: { 'X-Admin-Key': getAdminKey() },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast(`⚠️ User ${userId} flagged as high risk (sandboxed to local Ollama)`);
  } catch (e) {
    showToast(`Error flagging user: ${e.message}`);
  }
}
