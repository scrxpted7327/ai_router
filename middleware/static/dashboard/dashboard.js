/**
 * AI Router control UI — session cookies + optional Bearer for docs only.
 */
const $ = (sel, root = document) => root.querySelector(sel);

const state = {
  user: null,
  tab: "overview",
  health: null,
  models: null,
  conversations: null,
  selectedConv: null,
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
  state.health = r.ok ? await r.json() : { status: "error", models: "—" };
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
    el.textContent = "—";
    el.className = "pill";
    return;
  }
  el.textContent = u.is_whitelisted ? "whitelisted" : "pending whitelist";
  el.className = "pill " + (u.is_whitelisted ? "ok" : "warn");
}

function renderOverview() {
  const h = state.health || {};
  $("#stat-status").textContent = h.status === "ok" ? "online" : "check";
  $("#stat-models").textContent =
    typeof h.models === "number" ? String(h.models) : "—";
  $("#stat-user").textContent = state.user ? state.user.email : "—";
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
      m.owned_by || "—"
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
      '<div class="empty">Your account is not whitelisted yet — /auth/conversations is blocked.</div>';
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

function selectConversation(id) {
  fetchConversation(id).then(() => {
    renderConversations();
  });
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
    }[name] || "";

  if (name === "models") renderModels();
  if (name === "memory") {
    await fetchConversations();
    renderConversations();
  }
  if (name === "connect") renderConnect();
}

function renderConnect() {
  const origin = window.location.origin;
  $("#connect-base").textContent = origin + "/v1";
  $("#connect-curl").textContent = `curl -sS ${origin}/health
curl -sS ${origin}/v1/models \\
  -H "Authorization: Bearer air_YOUR_TOKEN"`;
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
  await api("/auth/logout", { method: "POST" });
  state.user = null;
  state.conversations = null;
  state.selectedConv = null;
  await refreshAll();
}

async function refreshAll() {
  await fetchHealth();
  await fetchModels();
  await fetchMe();
  if (state.user) {
    await fetchConversations();
  }
  if (state.user?.is_whitelisted && state.tab === "memory" && state.selectedConv) {
    const id = state.selectedConv.id;
    await fetchConversation(id);
  }

  if (!state.user) {
    showAuthGate();
    renderOverview();
    return;
  }
  showApp();
  $("#dash-email").textContent = state.user.email;
  renderUserPill();
  renderOverview();
  renderModels();
  renderConversations();
  renderConnect();
}

function wire() {
  $("#form-login").addEventListener("submit", onLogin);
  $("#form-register").addEventListener("submit", onRegister);
  $("#btn-logout").addEventListener("click", onLogout);

  document.querySelectorAll(".nav button[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => void setTab(btn.dataset.tab));
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  wire();
  await refreshAll();
  const boot = document.getElementById("boot");
  if (boot) boot.remove();
  await setTab("overview");
});
