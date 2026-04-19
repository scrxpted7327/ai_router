/**
 * AI Router control UI — session cookies + optional Bearer for docs only.
 */
const $ = (sel, root = document) => root.querySelector(sel);

const PROVIDER_CATALOG = [
  { id: "anthropic",    name: "Anthropic",     hint: "sk-ant-…",     url: "https://console.anthropic.com/settings/keys" },
  { id: "openai",       name: "OpenAI",        hint: "sk-…",         url: "https://platform.openai.com/api-keys" },
  { id: "gemini",       name: "Google Gemini", hint: "AIza…",        url: "https://aistudio.google.com/apikey" },
  { id: "mistral",      name: "Mistral",       hint: "…",            url: "https://console.mistral.ai/api-keys" },
  { id: "deepseek",     name: "DeepSeek",      hint: "sk-…",         url: "https://platform.deepseek.com/api_keys" },
  { id: "xai",          name: "xAI (Grok)",    hint: "xai-…",        url: "https://console.x.ai" },
  { id: "groq",         name: "Groq",          hint: "gsk_…",        url: "https://console.groq.com/keys" },
  { id: "cerebras",     name: "Cerebras",      hint: "csk-…",        url: "https://cloud.cerebras.ai/platform" },
  { id: "openrouter",   name: "OpenRouter",    hint: "sk-or-…",      url: "https://openrouter.ai/settings/keys" },
  { id: "together",     name: "Together AI",   hint: "…",            url: "https://api.together.xyz/settings/api-keys" },
  { id: "perplexity",   name: "Perplexity",    hint: "pplx-…",       url: "https://www.perplexity.ai/settings/api" },
  { id: "fireworks",    name: "Fireworks",     hint: "fw-…",         url: "https://fireworks.ai/account/api-keys" },
  { id: "cohere",       name: "Cohere",        hint: "…",            url: "https://dashboard.cohere.com/api-keys" },
];

const state = {
  user: null,
  tab: "overview",
  health: null,
  models: null,
  conversations: null,
  selectedConv: null,
  piStatus: null,
  tokenMeta: null,
  lastToken: "",
  modelControls: null,
  providerTokens: [],       // [{provider_id, token_prefix, updated_at}]
  adminUsers: null,
  adminSelectedUser: null,
  adminUserTokens: [],
  terminal: {
    term: null,
    socket: null,
    fitTimer: null,
    onDataBound: false,
    resizeBound: false,
  },
};

const api = (path, opts = {}) =>
  fetch(path, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });

async function fetchMe() {
  const r = await api("/auth/me");
  if (r.status === 401) {
    state.user = null;
    return null;
  }
  if (!r.ok) throw new Error(`GET /auth/me ${r.status}`);
  state.user = await r.json();
  return state.user;
}

async function fetchHealth() {
  const r = await fetch("/health");
  state.health = r.ok ? await r.json() : { status: "error", models: "-" };
}

async function fetchModels() {
  const r = await fetch("/v1/models");
  state.models = r.ok ? await r.json() : null;
}

async function fetchConversations() {
  const r = await api("/auth/conversations");
  if (r.status === 401) {
    state.conversations = null;
    return;
  }
  if (r.status === 403) {
    state.conversations = [];
    return;
  }
  if (!r.ok) throw new Error(`GET /auth/conversations ${r.status}`);
  state.conversations = await r.json();
}

async function fetchConversation(id) {
  const r = await api(`/auth/conversations/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error("Failed to load conversation");
  state.selectedConv = await r.json();
}

async function fetchPiStatus() {
  if (!state.user?.is_admin) {
    state.piStatus = null;
    return;
  }
  const r = await api("/auth/pi/status");
  if (r.status === 403) {
    state.piStatus = null;
    return;
  }
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    state.piStatus = {
      providers: [],
      error: data.detail || `GET /auth/pi/status ${r.status}`,
    };
    return;
  }
  state.piStatus = await r.json();
}

async function refreshPiTokens() {
  const r = await api("/auth/pi/refresh-tokens", { method: "POST" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.detail || "Failed to refresh tokens");
  }
  state.piStatus = data;
  renderPiStatus();
}

async function fetchTokenMeta() {
  if (!state.user) {
    state.tokenMeta = null;
    return;
  }
  const r = await api("/auth/tokens");
  if (!r.ok) {
    state.tokenMeta = {
      has_token: false,
      token_count: 0,
      error: `GET /auth/tokens ${r.status}`,
    };
    return;
  }
  state.tokenMeta = await r.json();
}

async function regenerateToken() {
  const r = await api("/auth/tokens/regenerate", { method: "POST" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.detail || "Failed to regenerate token");
  }
  state.lastToken = data.token || "";
  await fetchTokenMeta();
}

async function copyLastToken() {
  if (!state.lastToken) {
    throw new Error("No token available yet. Click Regenerate first.");
  }
  await navigator.clipboard.writeText(state.lastToken);
}

// ── Provider token helpers ────────────────────────────────────────────────────

async function fetchProviderTokens() {
  if (!state.user) { state.providerTokens = []; return; }
  const r = await api("/auth/provider-tokens");
  state.providerTokens = r.ok ? await r.json() : [];
}

async function upsertProviderToken(providerId, token) {
  const r = await api(`/auth/provider-tokens/${encodeURIComponent(providerId)}`, {
    method: "PUT",
    body: JSON.stringify({ token }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Failed to save token");
  await fetchProviderTokens();
}

async function deleteProviderToken(providerId) {
  const r = await api(`/auth/provider-tokens/${encodeURIComponent(providerId)}`, {
    method: "DELETE",
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Failed to delete token");
  await fetchProviderTokens();
}

function renderProviderTokens(gridId, tokens, forUserId) {
  const grid = document.getElementById(gridId);
  if (!grid) return;

  const byProvider = Object.fromEntries(tokens.map((t) => [t.provider_id, t]));

  grid.innerHTML = PROVIDER_CATALOG.map((p) => {
    const existing = byProvider[p.id];
    const isSet = !!existing;
    return `<div class="provider-card ${isSet ? "provider-card--set" : ""}" data-provider-card="${p.id}" data-for-user="${forUserId || ""}">
      <div class="provider-card-head">
        <span class="provider-card-name">${escapeHtml(p.name)}</span>
        ${isSet ? `<span class="pill ok" style="font-size:0.68rem">set</span>` : `<span class="pill" style="font-size:0.68rem">not set</span>`}
      </div>
      ${isSet ? `<div class="provider-card-prefix mono">${escapeHtml(existing.token_prefix || "")}</div>` : ""}
      <div class="provider-card-form">
        <input
          type="password"
          class="provider-token-input"
          placeholder="${escapeHtml(p.hint)}"
          autocomplete="off"
          data-provider="${p.id}"
        />
        <div class="provider-card-actions">
          <button type="button" class="btn btn-primary btn-xs" data-action="save" data-provider="${p.id}">
            ${isSet ? "Update" : "Save"}
          </button>
          ${isSet ? `<button type="button" class="btn btn-danger btn-xs" data-action="delete" data-provider="${p.id}">Remove</button>` : ""}
          <a href="${p.url}" target="_blank" rel="noopener" class="btn btn-xs">Get key ↗</a>
        </div>
      </div>
    </div>`;
  }).join("");
}

async function fetchAdminUsers() {
  if (!state.user?.is_admin) { state.adminUsers = null; return; }
  const r = await api("/dashboard/users");
  state.adminUsers = r.ok ? (await r.json()).users : null;
}

async function fetchAdminUserTokens(userId) {
  const r = await api(`/dashboard/users/${encodeURIComponent(userId)}/provider-tokens`);
  state.adminUserTokens = r.ok ? (await r.json()).tokens : [];
}

async function adminUpsertToken(userId, providerId, token) {
  const r = await api(`/dashboard/users/${encodeURIComponent(userId)}/provider-tokens/${encodeURIComponent(providerId)}`, {
    method: "PUT",
    body: JSON.stringify({ token }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Failed to save token");
  await fetchAdminUserTokens(userId);
}

async function adminDeleteToken(userId, providerId) {
  const r = await api(`/dashboard/users/${encodeURIComponent(userId)}/provider-tokens/${encodeURIComponent(providerId)}`, {
    method: "DELETE",
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Failed to delete token");
  await fetchAdminUserTokens(userId);
}

function renderAdminUsers() {
  const list = document.getElementById("admin-users-list");
  const detail = document.getElementById("admin-user-detail");
  if (!list) return;

  if (!state.adminUsers?.length) {
    list.innerHTML = '<div class="empty">No users found.</div>';
    return;
  }

  list.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Email</th><th>Status</th><th>Keys</th><th></th></tr></thead>
    <tbody>
      ${state.adminUsers.map((u) => `
        <tr>
          <td>${escapeHtml(u.email)}</td>
          <td>${u.is_admin ? '<span class="pill ok">admin</span>' : u.is_whitelisted ? '<span class="pill ok">whitelisted</span>' : '<span class="pill warn">pending</span>'}</td>
          <td class="mono">${u.provider_token_count}</td>
          <td><button class="btn btn-xs" data-action="manage-user" data-user-id="${escapeHtml(u.id)}" data-user-email="${escapeHtml(u.email)}">Manage keys</button></td>
        </tr>`).join("")}
    </tbody>
  </table></div>`;

  if (state.adminSelectedUser) {
    detail.style.display = "";
    document.getElementById("admin-user-detail-title").textContent =
      `Keys for ${state.adminSelectedUser.email}`;
    renderProviderTokens("admin-user-tokens-grid", state.adminUserTokens, state.adminSelectedUser.id);
  } else {
    detail.style.display = "none";
  }
}

async function fetchModelControls() {
  if (!state.user?.is_admin) {
    state.modelControls = null;
    return;
  }
  const r = await api("/dashboard/model-controls");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.detail || `GET /dashboard/model-controls ${r.status}`);
  }
  state.modelControls = data.models || [];
}

async function saveModelControls() {
  const rows = Array.from(document.querySelectorAll("[data-policy-row]"));
  const models = rows.map((row) => ({
    id: row.dataset.modelId,
    enabled: !!row.querySelector("[data-field='enabled']")?.checked,
    classification: row.querySelector("[data-field='classification']")?.value || "",
    effort: row.querySelector("[data-field='effort']")?.value || "medium",
  }));
  const r = await api("/dashboard/model-controls", {
    method: "POST",
    body: JSON.stringify({ models }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.detail || "Failed to save model policy");
  }
}

function showAuthGate() {
  $("#view-gate").classList.remove("hidden");
  $("#view-app").classList.add("hidden");
}

function showApp() {
  $("#view-gate").classList.add("hidden");
  $("#view-app").classList.remove("hidden");
}

function renderUserPill() {
  const u = state.user;
  const el = $("#user-pill");
  if (!u) {
    el.textContent = "-";
    el.className = "pill";
    return;
  }
  if (u.is_admin) {
    el.textContent = "admin";
    el.className = "pill ok";
    return;
  }
  el.textContent = u.is_whitelisted ? "whitelisted" : "pending whitelist";
  el.className = "pill " + (u.is_whitelisted ? "ok" : "warn");
}

function renderOverview() {
  const h = state.health || {};
  $("#stat-status").textContent = h.status === "ok" ? "online" : "check";
  $("#stat-models").textContent =
    typeof h.models === "number" ? String(h.models) : "-";
  $("#stat-user").textContent = state.user ? state.user.email : "-";
}

function renderModels() {
  const body = $("#models-body");
  body.innerHTML = "";
  const data = state.models?.data;
  if (!data?.length) {
    body.innerHTML =
      '<tr><td colspan="2" class="empty">No models returned.</td></tr>';
    return;
  }
  for (const m of data) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="mono">${escapeHtml(m.id)}</td><td>${escapeHtml(
      m.owned_by || "-"
    )}</td>`;
    body.appendChild(tr);
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderConversations() {
  const list = $("#conv-list");
  const detail = $("#conv-detail");
  list.innerHTML = "";
  detail.innerHTML = "";

  if (!state.user) {
    list.innerHTML =
      '<div class="empty">Sign in to view saved conversations.</div>';
    return;
  }
  if (!state.user.is_whitelisted) {
    list.innerHTML =
      '<div class="empty">Your account is not whitelisted yet - /auth/conversations is blocked.</div>';
    return;
  }
  const rows = state.conversations;
  if (!rows || rows.length === 0) {
    list.innerHTML =
      '<div class="empty">No conversations stored yet (they appear as you use the gateway with memory).</div>';
    return;
  }

  for (const c of rows) {
    const div = document.createElement("div");
    div.className = "conv-item";
    div.innerHTML = `<strong>${escapeHtml(
      c.title || "Untitled"
    )}</strong><span>${escapeHtml(c.model || "")} · ${escapeHtml(
      String(c.updated_at || "")
    )}</span>`;
    div.addEventListener("click", () => selectConversation(c.id));
    list.appendChild(div);
  }

  if (state.selectedConv) {
    detail.innerHTML = `<h3 style="margin:0 0 .5rem;font-size:.9rem">${escapeHtml(
      state.selectedConv.title || "Conversation"
    )}</h3><div class="detail-json">${escapeHtml(
      JSON.stringify(state.selectedConv, null, 2)
    )}</div>`;
  }
}

function renderPiStatus() {
  const box = $("#auth-status");
  if (!box) return;

  if (!state.user?.is_admin) {
    box.innerHTML = '<div class="empty">Admin access required.</div>';
    return;
  }

  if (!state.piStatus?.providers?.length) {
    if (state.piStatus?.error) {
      box.innerHTML = `<div class="empty">${escapeHtml(String(state.piStatus.error))}</div>`;
      return;
    }
    box.innerHTML = '<div class="empty">No provider status available.</div>';
    return;
  }

  box.innerHTML = state.piStatus.providers
    .map((p) => {
      const ready = p.has_token && !p.expired;
      const status = ready ? "ready" : "needs login";
      return `<div class="auth-row">
        <div><strong>${escapeHtml(p.name)}</strong></div>
        <div class="mono">${escapeHtml(p.env)}</div>
        <div class="state ${ready ? "ok" : "bad"}">${status}</div>
        <div class="mono">expires: ${escapeHtml(p.expires || "-")}</div>
      </div>`;
    })
    .join("");
}

function renderTokenMeta() {
  const status = $("#token-status");
  const value = $("#token-value");
  if (!status || !value) return;

  if (!state.user) {
    status.innerHTML = '<div class="empty">Sign in required.</div>';
    value.classList.add("hidden");
    return;
  }

  const meta = state.tokenMeta;
  if (!meta) {
    status.innerHTML = '<div class="empty">Token metadata unavailable.</div>';
  } else if (meta.error) {
    status.innerHTML = `<div class="empty">${escapeHtml(String(meta.error))}</div>`;
  } else {
    status.innerHTML = `<div class="token-box"><div>Tokens provisioned</div><div class="mono">${escapeHtml(
      String(meta.token_count || 0)
    )}</div></div>`;
  }

  if (state.lastToken) {
    value.textContent = state.lastToken;
    value.classList.remove("hidden");
  } else {
    value.classList.add("hidden");
  }
}

function groupModelControls(models) {
  const groups = new Map();
  for (const model of models || []) {
    const provider = model.provider || "unknown";
    if (!groups.has(provider)) groups.set(provider, []);
    groups.get(provider).push(model);
  }
  return Array.from(groups.entries()).sort((a, b) => a[0].localeCompare(b[0]));
}

function syncProviderToggle(container) {
  if (!container) return;
  const toggle = container.querySelector("[data-provider-toggle]");
  const boxes = Array.from(
    container.querySelectorAll("[data-policy-row] [data-field='enabled']")
  );
  if (!toggle || !boxes.length) return;

  const enabledCount = boxes.filter((box) => box.checked).length;
  toggle.indeterminate = enabledCount > 0 && enabledCount < boxes.length;
  toggle.checked = enabledCount === boxes.length;
}

function syncAllProviderToggles() {
  document.querySelectorAll(".policy-provider").forEach((container) => {
    syncProviderToggle(container);
  });
}

function applyProviderEnabled(container, checked) {
  if (!container) return;
  container
    .querySelectorAll("[data-policy-row] [data-field='enabled']")
    .forEach((box) => {
      box.checked = checked;
    });
  syncProviderToggle(container);
}

function applyProviderEffort(container, effort) {
  if (!container) return;
  container
    .querySelectorAll("[data-policy-row] [data-field='effort']")
    .forEach((sel) => {
      sel.value = effort;
    });
}

function renderModelPolicy() {
  const host = $("#model-policy-table");
  if (!host) return;

  if (!state.user?.is_admin) {
    host.innerHTML = '<div class="empty">Admin access required.</div>';
    return;
  }
  if (!state.modelControls?.length) {
    host.innerHTML = '<div class="empty">No model controls loaded.</div>';
    return;
  }

  const classOptions = [
    "",
    "heavy_reasoning",
    "code_generation",
    "nuanced_coding",
    "multimodal",
    "fast_simple",
  ];
  const effortOptions = ["default", "low", "medium", "high", "xhigh"];

  const groups = groupModelControls(state.modelControls);
  const sections = groups
    .map(([provider, models]) => {
      const rows = models
        .map((m) => {
          const classSelect = classOptions
            .map(
              (opt) =>
                `<option value="${opt}" ${m.classification === opt ? "selected" : ""}>${
                  opt || "(auto)"
                }</option>`
            )
            .join("");
          const effortSelect = effortOptions
            .map(
              (opt) =>
                `<option value="${opt}" ${m.effort === opt ? "selected" : ""}>${opt}</option>`
            )
            .join("");
          return `<tr data-policy-row data-model-id="${escapeHtml(m.id)}">
            <td class="mono">${escapeHtml(m.id)}</td>
            <td><input data-field="enabled" type="checkbox" ${m.enabled ? "checked" : ""}></td>
            <td><select data-field="classification">${classSelect}</select></td>
            <td><select data-field="effort">${effortSelect}</select></td>
          </tr>`;
        })
        .join("");

      return `<details class="policy-provider" data-provider="${escapeHtml(provider)}" open>
        <summary>
          <div class="provider-head">
            <span class="provider-title">${escapeHtml(provider)}</span>
            <span class="provider-meta">${models.length} model${models.length === 1 ? "" : "s"}</span>
          </div>
          <div class="provider-actions" onclick="event.stopPropagation()">
            <button type="button" class="btn" data-provider-action="enable">Enable all</button>
            <button type="button" class="btn" data-provider-action="disable">Disable all</button>
            <button type="button" class="btn" data-provider-action="effort-default">Effort: default</button>
            <button type="button" class="btn" data-provider-action="effort-low">Effort: low</button>
            <button type="button" class="btn" data-provider-action="effort-medium">Effort: medium</button>
            <button type="button" class="btn" data-provider-action="effort-high">Effort: high</button>
            <button type="button" class="btn" data-provider-action="effort-xhigh">Effort: xhigh</button>
          </div>
          <label class="provider-toggle" onclick="event.stopPropagation()">
            <input type="checkbox" data-provider-toggle="${escapeHtml(provider)}">
            <span>Enable all</span>
          </label>
        </summary>
        <div class="table-wrap">
          <table class="policy-table">
            <thead>
              <tr><th>Model</th><th>Enabled</th><th>Classification</th><th>Effort</th></tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </details>`;
    })
    .join("");

  host.innerHTML = `<div class="policy-groups">${sections}</div>`;
  syncAllProviderToggles();
}

function selectConversation(id) {
  fetchConversation(id).then(() => {
    renderConversations();
  });
}

function updateAdminNav() {
  const show = Boolean(state.user?.is_admin);
  document.querySelectorAll(".admin-only").forEach((el) => {
    el.classList.toggle("hidden", !show);
  });
}

function ensureTerminal() {
  if (state.terminal.term) return state.terminal.term;
  const host = $("#terminal-shell");
  if (!host || !window.Terminal) return null;
  const term = new window.Terminal({
    cursorBlink: true,
    fontFamily: '"JetBrains Mono", "Cascadia Code", monospace',
    fontSize: 13,
    theme: {
      background: "#07080d",
      foreground: "#e8eaef",
      cursor: "#2dd4bf",
      black: "#121520",
      red: "#f87171",
      green: "#2dd4bf",
      yellow: "#fbbf24",
      blue: "#60a5fa",
      magenta: "#a78bfa",
      cyan: "#22d3ee",
      white: "#f3f4f6",
      brightBlack: "#64748b",
      brightRed: "#fca5a5",
      brightGreen: "#5eead4",
      brightYellow: "#fde68a",
      brightBlue: "#93c5fd",
      brightMagenta: "#c4b5fd",
      brightCyan: "#67e8f9",
      brightWhite: "#ffffff",
    },
  });
  term.open(host);
  term.writeln("Interactive pi shell. Press Connect to start.");
  state.terminal.term = term;
  return term;
}

function showTerminalStatus(message, kind = "ok") {
  const el = $("#terminal-status");
  if (!el) return;
  el.className = `msg ${kind === "error" ? "error" : "ok"}`;
  el.textContent = message;
  el.classList.remove("hidden");
}

function closeTerminalSocket() {
  const socket = state.terminal.socket;
  state.terminal.socket = null;
  if (socket && socket.readyState <= 1) {
    socket.close();
  }
}

function connectTerminal() {
  if (!state.user?.is_admin) {
    showTerminalStatus("Admin access required", "error");
    return;
  }

  const term = ensureTerminal();
  if (!term) {
    showTerminalStatus("Terminal client failed to initialize", "error");
    return;
  }

  closeTerminalSocket();
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${scheme}://${window.location.host}/terminal`;
  console.log("[Terminal] Connecting to:", wsUrl);
  term.writeln(`[debug] Connecting to ${wsUrl}...\r\n`);

  const ws = new WebSocket(wsUrl);
  state.terminal.socket = ws;

  ws.addEventListener("open", () => {
    console.log("[Terminal] WebSocket opened");

    // Send initial terminal dimensions
    const cols = term.cols || 80;
    const rows = term.rows || 24;
    ws.send(JSON.stringify({ type: "resize", cols, rows }));
    console.log("[Terminal] Sent initial dimensions:", cols, "x", rows);

    term.focus();
    showTerminalStatus("Connected", "ok");
  });

  ws.addEventListener("message", (event) => {
    console.log("[Terminal] Message received:", event.data);
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.error("[Terminal] JSON parse error:", e);
      term.writeln(`\r\n[debug] Raw message: ${event.data}\r\n`);
      return;
    }
    console.log("[Terminal] Parsed message:", msg);
    if (msg.type === "output") {
      const output = msg.data || "";
      console.log("[Terminal] Writing output to terminal:", JSON.stringify(output), "length:", output.length);
      term.write(output);
      return;
    }
    if (msg.type === "error") {
      term.writeln(`\r\n[error] ${msg.data || "Unknown error"}\r\n`);
      showTerminalStatus(msg.data || "Terminal error", "error");
      return;
    }
    if (msg.type === "exit") {
      term.writeln(`\r\n[exit ${msg.code}]\r\n`);
    }
  });

  ws.addEventListener("close", (event) => {
    console.log("[Terminal] WebSocket closed:", event.code, event.reason);
    if (state.terminal.socket === ws) state.terminal.socket = null;
    term.writeln(`\r\n[disconnected: ${event.code}]\r\n`);
  });

  ws.addEventListener("error", (event) => {
    console.error("[Terminal] WebSocket error:", event);
    showTerminalStatus("WebSocket failed", "error");
    term.writeln("\r\n[WebSocket error - check browser console]\r\n");
  });

  // Bind input handler (always rebind to capture new socket)
  term.onData((data) => {
    console.log("[Terminal] onData received:", JSON.stringify(data), "length:", data.length, "charCode:", data.charCodeAt(0));
    const current = state.terminal.socket;
    if (!current) {
      console.log("[Terminal] No socket available");
      return;
    }
    if (current.readyState !== WebSocket.OPEN) {
      console.log("[Terminal] Socket not open, state:", current.readyState);
      return;
    }

    // No local echo - PTY handles all echoing and cursor positioning
    const message = JSON.stringify({ type: "input", data });
    console.log("[Terminal] Sending input:", message);
    current.send(message);
  });

  // Handle terminal resize events
  term.onResize(({ cols, rows }) => {
    console.log("[Terminal] Terminal resized:", cols, "x", rows);
    const current = state.terminal.socket;
    if (current && current.readyState === WebSocket.OPEN) {
      current.send(JSON.stringify({ type: "resize", cols, rows }));
      console.log("[Terminal] Sent resize to server");
    }
  });

  console.log("[Terminal] Input and resize handlers bound (PTY handles echo)");
}

async function setTab(name) {
  state.tab = name;
  document.querySelectorAll(".nav button[data-tab]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll("[data-panel]").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.panel !== name);
  });
  $("#panel-title").textContent =
    {
      overview: "Overview",
      models: "Models",
      memory: "Memory",
      connect: "Connect",
      tokens: "API Keys",
      users: "Users",
      auth: "Auth",
      terminal: "Terminal",
      "model-policy": "Model Policy",
    }[name] || "";

  if (name === "models") renderModels();
  if (name === "memory") {
    await fetchConversations();
    renderConversations();
  }
  if (name === "connect") renderConnect();
  if (name === "tokens") {
    await fetchProviderTokens();
    renderProviderTokens("provider-tokens-grid", state.providerTokens, null);
  }
  if (name === "users") {
    await fetchAdminUsers();
    renderAdminUsers();
  }
  if (name === "auth") {
    await fetchPiStatus();
    renderPiStatus();
  }
  if (name === "terminal") {
    ensureTerminal();
  }
  if (name === "model-policy") {
    await fetchModelControls();
    renderModelPolicy();
  }
}

function renderConnect() {
  const origin = window.location.origin;
  $("#connect-base").textContent = origin + "/v1";
  $("#connect-curl").textContent = `curl -sS ${origin}/health
curl -sS ${origin}/v1/models \\
  -H "Authorization: Bearer air_YOUR_TOKEN"`;
  const ant = $("#anthropic-connect");
  if (ant) {
    ant.textContent = `ANTHROPIC_BASE_URL=${origin}/anthropic
ANTHROPIC_API_KEY=air_YOUR_TOKEN

curl -sS ${origin}/anthropic/v1/models \\
  -H "x-api-key: air_YOUR_TOKEN"`;
  }
  renderTokenMeta();
}

async function onLogin(e) {
  e.preventDefault();
  const email = $("#login-email").value.trim();
  const password = $("#login-password").value;
  const msg = $("#gate-msg");
  msg.classList.add("hidden");
  const r = await api("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    msg.textContent = data.detail || "Login failed";
    msg.className = "msg error";
    msg.classList.remove("hidden");
    return;
  }
  await refreshAll();
}

async function onRegister(e) {
  e.preventDefault();
  const email = $("#reg-email").value.trim();
  const password = $("#reg-password").value;
  const msg = $("#gate-msg");
  msg.classList.add("hidden");
  const r = await api("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    msg.textContent =
      typeof data.detail === "string"
        ? data.detail
        : "Registration failed";
    msg.className = "msg error";
    msg.classList.remove("hidden");
    return;
  }
  msg.textContent =
    "Account created. Ask an operator to whitelist you, then log in.";
  msg.className = "msg ok";
  msg.classList.remove("hidden");
}

async function onLogout() {
  closeTerminalSocket();
  await api("/auth/logout", { method: "POST" });
  state.user = null;
  state.conversations = null;
  state.selectedConv = null;
  state.piStatus = null;
  state.tokenMeta = null;
  state.lastToken = "";
  state.modelControls = null;
  await refreshAll();
}

async function refreshAll() {
  await fetchHealth();
  await fetchModels();
  await fetchMe();
  if (state.user) {
    await fetchConversations();
    await fetchTokenMeta();
  }
  if (state.user?.is_whitelisted && state.tab === "memory" && state.selectedConv) {
    const id = state.selectedConv.id;
    await fetchConversation(id);
  }
  if (state.user?.is_admin && state.tab === "auth") {
    await fetchPiStatus();
  }
  if (state.user?.is_admin && state.tab === "model-policy") {
    await fetchModelControls();
  }

  if (!state.user) {
    showAuthGate();
    renderOverview();
    updateAdminNav();
    return;
  }
  showApp();
  $("#dash-email").textContent = state.user.email;
  renderUserPill();
  renderOverview();
  renderModels();
  renderConversations();
  renderConnect();
  renderTokenMeta();
  renderPiStatus();
  renderModelPolicy();
  updateAdminNav();
  if (state.tab === "tokens") {
    await fetchProviderTokens();
    renderProviderTokens("provider-tokens-grid", state.providerTokens, null);
  }
  if (state.tab === "users" && state.user?.is_admin) {
    await fetchAdminUsers();
    renderAdminUsers();
  }
}

function wire() {
  $("#form-login").addEventListener("submit", onLogin);
  $("#form-register").addEventListener("submit", onRegister);
  $("#btn-logout").addEventListener("click", onLogout);
  $("#btn-auth-refresh").addEventListener("click", async () => {
    try {
      await refreshPiTokens();
      showTerminalStatus("Token refresh complete", "ok");
    } catch (err) {
      showTerminalStatus(String(err.message || err), "error");
    }
  });
  $("#btn-term-connect").addEventListener("click", connectTerminal);
  $("#btn-term-clear").addEventListener("click", () => {
    const term = ensureTerminal();
    if (term) term.clear();
  });
  $("#btn-token-regenerate")?.addEventListener("click", async () => {
    try {
      await regenerateToken();
      renderTokenMeta();
      showTerminalStatus("Token regenerated", "ok");
    } catch (err) {
      showTerminalStatus(String(err.message || err), "error");
    }
  });
  $("#btn-token-copy")?.addEventListener("click", async () => {
    try {
      await copyLastToken();
      showTerminalStatus("Token copied to clipboard", "ok");
    } catch (err) {
      showTerminalStatus(String(err.message || err), "error");
    }
  });
  $("#btn-policy-save")?.addEventListener("click", async () => {
    try {
      await saveModelControls();
      await fetchModelControls();
      renderModelPolicy();
      showTerminalStatus("Model policy saved", "ok");
    } catch (err) {
      showTerminalStatus(String(err.message || err), "error");
    }
  });

  $("#model-policy-table")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;

    if (target.matches("[data-provider-toggle]")) {
      const providerBox = target.closest(".policy-provider");
      if (!providerBox) return;
      applyProviderEnabled(providerBox, target.checked);
      return;
    }

    if (target.matches("[data-policy-row] [data-field='enabled']")) {
      syncProviderToggle(target.closest(".policy-provider"));
    }
  });

  $("#model-policy-table")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const actionBtn = target.closest("[data-provider-action]");
    if (!actionBtn) return;
    const providerBox = actionBtn.closest(".policy-provider");
    if (!providerBox) return;
    const action = actionBtn.getAttribute("data-provider-action");
    if (action === "enable") {
      applyProviderEnabled(providerBox, true);
      return;
    }
    if (action === "disable") {
      applyProviderEnabled(providerBox, false);
      return;
    }
    if (action === "effort-default") {
      applyProviderEffort(providerBox, "default");
      return;
    }
    if (action === "effort-low") {
      applyProviderEffort(providerBox, "low");
      return;
    }
    if (action === "effort-medium") {
      applyProviderEffort(providerBox, "medium");
      return;
    }
    if (action === "effort-high") {
      applyProviderEffort(providerBox, "high");
      return;
    }
    if (action === "effort-xhigh") {
      applyProviderEffort(providerBox, "xhigh");
    }
  });

  document.querySelectorAll(".nav button[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => void setTab(btn.dataset.tab));
  });

  // ── Provider token cards (own tokens) ──────────────────────────────────────
  function tokenMsg(text, kind) {
    const el = document.getElementById("tokens-msg");
    if (!el) return;
    el.textContent = text;
    el.className = `msg ${kind}`;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 4000);
  }

  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const providerId = btn.dataset.provider;
    const forUser = btn.dataset.forUser || btn.closest("[data-provider-card]")?.dataset.forUser || "";

    if (action === "save") {
      const card = btn.closest("[data-provider-card]");
      const input = card?.querySelector(".provider-token-input");
      const val = input?.value.trim();
      if (!val) { tokenMsg("Enter a key first", "error"); return; }
      try {
        if (forUser) {
          await adminUpsertToken(forUser, providerId, val);
          if (state.adminSelectedUser?.id === forUser) {
            renderProviderTokens("admin-user-tokens-grid", state.adminUserTokens, forUser);
          }
        } else {
          await upsertProviderToken(providerId, val);
          renderProviderTokens("provider-tokens-grid", state.providerTokens, null);
          tokenMsg(`${providerId} key saved`, "ok");
        }
        if (input) input.value = "";
      } catch (err) { tokenMsg(String(err.message || err), "error"); }
      return;
    }

    if (action === "delete") {
      if (!confirm(`Remove ${providerId} key?`)) return;
      try {
        if (forUser) {
          await adminDeleteToken(forUser, providerId);
          if (state.adminSelectedUser?.id === forUser) {
            renderProviderTokens("admin-user-tokens-grid", state.adminUserTokens, forUser);
          }
        } else {
          await deleteProviderToken(providerId);
          renderProviderTokens("provider-tokens-grid", state.providerTokens, null);
          tokenMsg(`${providerId} key removed`, "ok");
        }
      } catch (err) { tokenMsg(String(err.message || err), "error"); }
      return;
    }

    if (action === "manage-user") {
      state.adminSelectedUser = { id: btn.dataset.userId, email: btn.dataset.userEmail };
      await fetchAdminUserTokens(btn.dataset.userId);
      renderAdminUsers();
      return;
    }
  });

  document.getElementById("btn-admin-user-back")?.addEventListener("click", () => {
    state.adminSelectedUser = null;
    state.adminUserTokens = [];
    renderAdminUsers();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  wire();
  await refreshAll();
  const boot = document.getElementById("boot");
  if (boot) boot.remove();
  await setTab("overview");
});
