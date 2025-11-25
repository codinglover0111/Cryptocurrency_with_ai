/**
 * Admin Panel JavaScript
 */

// State
const adminState = {
  authenticated: false,
  user: null,
  models: {},
  agentConfig: {},
  scheduler: {},
  apiKeysStatus: {},
  currentSection: "dashboard",
  currentBybitEnv: "demo",
  availableSymbols: [],
  selectedSymbols: [],
  defaultSymbols: [],
};

// Utilities
async function fetchJSON(path, init = {}) {
  const res = await fetch(path, {
    credentials: "include",
    ...init,
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

async function postJSON(path, payload, init = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    credentials: "include",
    ...init,
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  const str = String(value ?? "");
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", { maximumFractionDigits: decimals });
}

function formatUSD(value) {
  if (value === null || value === undefined) return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(num);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num >= 0 ? "+" : ""}${num.toFixed(1)}%`;
}

function formatTime(isoString) {
  if (!isoString) return "-";
  try {
    const date = new Date(isoString);
    return date.toLocaleString("ko-KR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return isoString;
  }
}

// Auth
async function checkAuth() {
  try {
    const data = await fetchJSON("/auth/me");
    adminState.authenticated = !!data.authenticated;
    adminState.user = data.user || null;
  } catch {
    adminState.authenticated = false;
    adminState.user = null;
  }
  updateAuthUI();
}

function updateAuthUI() {
  const authSection = el("auth-section");
  const userInfo = el("admin-user-info");
  const sections = document.querySelectorAll(
    ".admin-section:not(#auth-section)"
  );

  if (adminState.authenticated && adminState.user?.role === "admin") {
    if (authSection) authSection.hidden = true;
    sections.forEach(
      (s) => (s.hidden = !s.id.includes(adminState.currentSection))
    );
    if (userInfo) {
      userInfo.innerHTML = `
        <span class="user-avatar">👤</span>
        <span class="user-name">관리자</span>
      `;
    }
    loadAdminData();
  } else {
    if (authSection) authSection.hidden = false;
    sections.forEach((s) => (s.hidden = true));
    if (userInfo) {
      userInfo.innerHTML = `
        <span class="user-avatar">👤</span>
        <span class="user-name">로그인 필요</span>
      `;
    }
  }
}

async function handleLogin(e) {
  e.preventDefault();
  const username = el("admin-username")?.value?.trim();
  const password = el("admin-password")?.value;
  const statusEl = el("admin-auth-status");

  if (!username || !password) return;

  try {
    if (statusEl) statusEl.textContent = "로그인 중...";
    const res = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "include",
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "로그인 실패");
    }

    el("admin-password").value = "";
    await checkAuth();
    if (statusEl) statusEl.textContent = "";
  } catch (err) {
    console.error(err);
    if (statusEl) {
      statusEl.textContent =
        err.message || "로그인 실패: 아이디 또는 비밀번호를 확인하세요";
    }
  }
}

async function handleLogout() {
  try {
    await postJSON("/auth/logout", {});
  } catch {
    // ignore
  }
  adminState.authenticated = false;
  adminState.user = null;
  updateAuthUI();
}

// Navigation
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item[data-section]");
  navItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const section = item.dataset.section;
      if (!section) return;

      // Update active state
      navItems.forEach((n) => n.classList.remove("active"));
      item.classList.add("active");

      // Show section
      adminState.currentSection = section;
      showSection(section);
    });
  });
}

function showSection(sectionId) {
  const sections = document.querySelectorAll(
    ".admin-section:not(#auth-section)"
  );
  sections.forEach((s) => {
    s.hidden = !s.id.includes(sectionId);
  });

  // Update header
  const titles = {
    dashboard: {
      title: "대시보드",
      subtitle: "시스템 상태를 한눈에 확인하세요",
    },
    agents: {
      title: "에이전트 설정",
      subtitle: "AI 에이전트별 LLM 모델을 구성하세요",
    },
    scheduler: {
      title: "스케줄러",
      subtitle: "자동매매 주기와 옵션을 설정하세요",
    },
    apikeys: {
      title: "API 키 설정",
      subtitle: "거래소 및 LLM Provider API 키를 관리하세요",
    },
    system: {
      title: "시스템",
      subtitle: "시스템 정보와 긴급 조치를 관리하세요",
    },
    logs: {
      title: "시스템 로그",
      subtitle: "실시간 로그를 확인하세요",
    },
    risk: {
      title: "리스크 설정",
      subtitle: "레버리지, 손실 한도, 포지션 할당을 설정하세요",
    },
    symbols: {
      title: "거래 심볼",
      subtitle: "자동매매에 사용할 거래 심볼을 설정하세요",
    },
  };

  const info = titles[sectionId] || { title: sectionId, subtitle: "" };
  const titleEl = el("page-title");
  const subtitleEl = el("page-subtitle");
  if (titleEl) titleEl.textContent = info.title;
  if (subtitleEl) subtitleEl.textContent = info.subtitle;

  // Load blocked IPs when system section is shown
  if (sectionId === "system") {
    loadBlockedIPs();
  }

  // Load logs when logs section is shown
  if (sectionId === "logs") {
    loadLogs();
  }

  // Load risk config when risk section is shown
  if (sectionId === "risk") {
    loadRiskConfig();
  }

  // Load symbols when symbols section is shown
  if (sectionId === "symbols") {
    loadTradingSymbols();
  }
}

// Load Admin Data
async function loadAdminData() {
  try {
    const [models, agents, scheduler, apiKeysStatus] = await Promise.all([
      fetchJSON("/admin/models").catch(() => ({ providers: {} })),
      fetchJSON("/admin/agent-config").catch(() => ({})),
      fetchJSON("/admin/scheduler").catch(() => ({})),
      fetchJSON("/admin/api-keys/status").catch(() => ({ status: {} })),
    ]);

    adminState.models = models.providers || {};
    adminState.agentConfig = agents || {};
    adminState.scheduler = scheduler || {};
    adminState.apiKeysStatus = apiKeysStatus.status || {};

    populateAgentForms();
    populateSchedulerForm();
    renderAvailableModels();
    loadDashboardData();
    updateApiKeysUI();
  } catch (err) {
    console.warn("관리자 데이터 로드 실패", err);
  }
}

// Dashboard
async function loadDashboardData() {
  try {
    const [status, stats] = await Promise.all([
      fetchJSON("/status").catch(() => ({})),
      fetchJSON("/stats").catch(() => ({})),
    ]);

    // Balance
    const balance = status.balance || {};
    const balanceEl = el("stat-balance");
    if (balanceEl) {
      balanceEl.textContent = formatUSD(balance.total);
    }

    // Stats
    const profitEl = el("stat-profit");
    const tradesEl = el("stat-trades");
    const winrateEl = el("stat-winrate");

    if (profitEl) {
      const pnl = stats.realized_pnl || 0;
      profitEl.textContent = formatUSD(pnl);
      profitEl.className = `stat-value ${
        pnl >= 0 ? "pnl-positive" : "pnl-negative"
      }`;
    }
    if (tradesEl) tradesEl.textContent = stats.trades || 0;
    if (winrateEl)
      winrateEl.textContent = formatPercent((stats.win_rate || 0) * 100);

    // Positions
    renderPositions(status.positionsSummary || []);

    // Recent Activity
    loadRecentActivity();
  } catch (err) {
    console.warn("대시보드 데이터 로드 실패", err);
  }
}

function renderPositions(positions) {
  const container = el("admin-positions");
  if (!container) return;

  if (!positions.length) {
    container.innerHTML = '<p class="muted">활성 포지션 없음</p>';
    return;
  }

  const rows = positions
    .map((p) => {
      const pnlClass = (p.pnl || 0) >= 0 ? "pnl-positive" : "pnl-negative";
      const side = (p.side || "").toLowerCase();
      const sideLabel =
        side === "long" || side === "buy"
          ? "롱"
          : side === "short" || side === "sell"
          ? "숏"
          : p.side;
      return `
        <tr>
          <td>${escapeHtml(p.symbol || "-")}</td>
          <td>${sideLabel}</td>
          <td class="text-right">${formatNumber(p.entryPrice, 4)}</td>
          <td class="text-right">${formatNumber(p.size, 4)}</td>
          <td class="text-right ${pnlClass}">${formatUSD(p.pnl)}</td>
        </tr>
      `;
    })
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>심볼</th>
          <th>사이드</th>
          <th class="text-right">진입가</th>
          <th class="text-right">수량</th>
          <th class="text-right">PNL</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadRecentActivity() {
  const container = el("recent-activity");
  if (!container) return;

  try {
    const data = await fetchJSON(
      "/api/journals_filtered?limit=10&types=decision,action,error&today_only=0"
    );
    const items = data.items || [];

    if (!items.length) {
      container.innerHTML = '<p class="muted">최근 활동 없음</p>';
      return;
    }

    const html = items
      .map((item) => {
        const type = (item.entry_type || "").toLowerCase();
        const typeClass =
          type === "error"
            ? "error"
            : type === "action"
            ? "action"
            : "decision";
        const typeLabel =
          type === "decision"
            ? "판단"
            : type === "action"
            ? "액션"
            : type === "error"
            ? "오류"
            : type;
        const symbol = item.symbol ? ` · ${item.symbol}` : "";
        let content = item.content || "";
        if (content.length > 100) content = content.substring(0, 100) + "...";

        return `
          <div class="activity-item activity-item--${typeClass}">
            <span class="activity-time">${formatTime(item.ts)}</span>
            <div class="activity-content">
              <span class="activity-type">${typeLabel}${symbol}</span>
              <p>${escapeHtml(content)}</p>
            </div>
          </div>
        `;
      })
      .join("");

    container.innerHTML = html;
  } catch {
    container.innerHTML = '<p class="muted">활동 로드 실패</p>';
  }
}

// Agent Config
function populateAgentForms() {
  const agents = [
    "indicator_agent",
    "pattern_agent",
    "trend_agent",
    "decision_agent",
  ];
  const providers = Object.keys(adminState.models);

  agents.forEach((agentKey) => {
    const card = document.querySelector(`[data-agent="${agentKey}"]`);
    if (!card) return;

    const config = adminState.agentConfig[agentKey] || {};
    const providerSelect = card.querySelector('[data-field="provider"]');
    const modelInput = card.querySelector('[data-field="model"]');
    const tempInput = card.querySelector('[data-field="temperature"]');

    // Populate provider options
    if (providerSelect && providers.length) {
      providerSelect.innerHTML = providers
        .map(
          (p) =>
            `<option value="${p}" ${
              p === config.provider ? "selected" : ""
            }>${p}</option>`
        )
        .join("");
    }

    if (modelInput) modelInput.value = config.model || "";
    if (tempInput) tempInput.value = config.temperature ?? 0.1;
  });
}

function gatherAgentConfig() {
  const agents = [
    "indicator_agent",
    "pattern_agent",
    "trend_agent",
    "decision_agent",
  ];
  const payload = {};

  agents.forEach((agentKey) => {
    const card = document.querySelector(`[data-agent="${agentKey}"]`);
    if (!card) return;

    const provider = card.querySelector('[data-field="provider"]')?.value || "";
    const model = card.querySelector('[data-field="model"]')?.value || "";
    const temperature = parseFloat(
      card.querySelector('[data-field="temperature"]')?.value || "0.1"
    );

    payload[agentKey] = {
      provider,
      model,
      temperature: Number.isFinite(temperature) ? temperature : 0.1,
    };
  });

  return payload;
}

async function saveAgentConfig() {
  const hint = el("agent-config-hint");
  try {
    if (hint) hint.textContent = "저장 중...";
    const payload = gatherAgentConfig();
    await postJSON("/admin/agent-config", payload);
    if (hint) hint.textContent = "✓ 저장 완료";
    setTimeout(() => {
      if (hint && hint.textContent === "✓ 저장 완료") hint.textContent = "";
    }, 3000);
    await loadAdminData();
  } catch (err) {
    console.error(err);
    if (hint) hint.textContent = "✗ 저장 실패";
  }
}

function resetAgentConfig() {
  populateAgentForms();
  const hint = el("agent-config-hint");
  if (hint) hint.textContent = "초기화됨";
  setTimeout(() => {
    if (hint && hint.textContent === "초기화됨") hint.textContent = "";
  }, 2000);
}

// Scheduler
function populateSchedulerForm() {
  const scheduler = adminState.scheduler || {};
  const autoInput = el("automation-minutes");
  const lossInput = el("loss-review-minutes");
  const coldFlag = el("cold-start-flag");

  if (autoInput) autoInput.value = scheduler.automation_minutes ?? 15;
  if (lossInput) lossInput.value = scheduler.loss_review_minutes ?? 60;
  if (coldFlag) coldFlag.checked = !!scheduler.cold_start;

  // 스케줄 상태 업데이트
  updateSchedulerStatus(scheduler);
}

function updateSchedulerStatus(scheduler) {
  const statusBadge = el("automation-status");
  const nextRunEl = el("next-run");
  const lastRunEl = el("last-run");

  // 실행 상태
  if (statusBadge) {
    const isRunning = scheduler.is_running;
    statusBadge.textContent = isRunning ? "활성" : "비활성";
    statusBadge.className = isRunning
      ? "status-badge status-badge--active"
      : "status-badge status-badge--inactive";
  }

  // 마지막 실행 시간
  if (lastRunEl) {
    lastRunEl.textContent = formatTime(scheduler.last_automation_run) || "-";
  }

  // 다음 실행 시간
  if (nextRunEl) {
    const nextRun = scheduler.next_automation_run;
    if (nextRun) {
      const nextDate = new Date(nextRun);
      const now = new Date();
      if (nextDate > now) {
        // 남은 시간 계산
        const diffMs = nextDate - now;
        const diffMins = Math.floor(diffMs / 60000);
        const diffSecs = Math.floor((diffMs % 60000) / 1000);
        if (diffMins > 0) {
          nextRunEl.textContent = `${diffMins}분 ${diffSecs}초 후`;
        } else {
          nextRunEl.textContent = `${diffSecs}초 후`;
        }
      } else {
        nextRunEl.textContent = "곧 실행";
      }
    } else if (scheduler.is_running) {
      // 실행 중이지만 아직 첫 실행 전
      nextRunEl.textContent = "대기 중...";
    } else {
      nextRunEl.textContent = "-";
    }
  }
}

async function saveScheduler() {
  const hint = el("scheduler-hint");
  const autoInput = el("automation-minutes");
  const lossInput = el("loss-review-minutes");
  const coldFlag = el("cold-start-flag");

  try {
    if (hint) hint.textContent = "저장 중...";
    await postJSON("/admin/scheduler", {
      automation_minutes: Number(autoInput?.value || 15),
      loss_review_minutes: Number(lossInput?.value || 60),
      cold_start: !!coldFlag?.checked,
    });
    if (hint) hint.textContent = "✓ 저장 완료";
    setTimeout(() => {
      if (hint && hint.textContent === "✓ 저장 완료") hint.textContent = "";
    }, 3000);
    await loadAdminData();
  } catch (err) {
    console.error(err);
    if (hint) hint.textContent = "✗ 저장 실패";
  }
}

function resetScheduler() {
  populateSchedulerForm();
  const hint = el("scheduler-hint");
  if (hint) hint.textContent = "초기화됨";
  setTimeout(() => {
    if (hint && hint.textContent === "초기화됨") hint.textContent = "";
  }, 2000);
}

// System
function renderAvailableModels() {
  const container = el("available-models");
  if (!container) return;

  const models = adminState.models;
  if (!Object.keys(models).length) {
    container.innerHTML = '<p class="muted">모델 정보 없음</p>';
    return;
  }

  const html = Object.entries(models)
    .map(
      ([provider, modelList]) => `
      <div class="model-group">
        <h4>${escapeHtml(provider)}</h4>
        <ul>
          ${(modelList || []).map((m) => `<li>${escapeHtml(m)}</li>`).join("")}
        </ul>
      </div>
    `
    )
    .join("");

  container.innerHTML = html;
}

async function closeAllPositions() {
  if (
    !confirm(
      "정말 모든 포지션을 청산하시겠습니까?\n이 작업은 되돌릴 수 없습니다."
    )
  ) {
    return;
  }

  try {
    await postJSON("/close_all", {});
    alert("전체 포지션 청산 요청이 전송되었습니다.");
    loadDashboardData();
  } catch (err) {
    alert("청산 실패: " + err.message);
  }
}

async function syncStats() {
  try {
    await postJSON("/stats/reconcile", {});
    alert("통계 동기화가 완료되었습니다.");
    loadDashboardData();
  } catch (err) {
    alert("동기화 실패: " + err.message);
  }
}

// Blocked IPs Management
async function loadBlockedIPs() {
  const container = el("blocked-ips-table");
  if (!container) return;

  try {
    const data = await fetchJSON("/auth/blocked-ips");
    const items = data.items || [];

    if (!items.length) {
      container.innerHTML = '<p class="muted">차단된 IP가 없습니다</p>';
      return;
    }

    const rows = items
      .map((ip) => {
        const blockedAt = ip.blocked_at
          ? new Date(ip.blocked_at).toLocaleString("ko-KR")
          : "-";
        return `
          <tr>
            <td><code>${escapeHtml(ip.ip_address)}</code></td>
            <td>${escapeHtml(ip.reason || "-")}</td>
            <td>${blockedAt}</td>
            <td>
              <button class="btn btn--ghost btn--sm" onclick="unblockIP('${escapeHtml(
                ip.ip_address
              )}')">
                차단 해제
              </button>
            </td>
          </tr>
        `;
      })
      .join("");

    container.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>IP 주소</th>
            <th>사유</th>
            <th>차단 시각</th>
            <th>조치</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (err) {
    console.error("Blocked IPs load error:", err);
    container.innerHTML = '<p class="muted">차단 목록을 불러올 수 없습니다</p>';
  }
}

async function unblockIP(ipAddress) {
  if (!confirm(`${ipAddress} 의 차단을 해제하시겠습니까?`)) {
    return;
  }

  try {
    await postJSON("/auth/unblock-ip", { ip_address: ipAddress });
    alert(`${ipAddress} 차단이 해제되었습니다.`);
    loadBlockedIPs();
  } catch (err) {
    alert("차단 해제 실패: " + err.message);
  }
}

// API Keys Management
function updateApiKeysUI() {
  const status = adminState.apiKeysStatus;

  // Update Bybit status
  const bybitStatus = status.bybit;
  if (bybitStatus) {
    const env = adminState.currentBybitEnv;
    const envStatus = bybitStatus.environments?.[env];
    const bybitBadge = el("bybit-status");
    if (bybitBadge && envStatus) {
      const allConfigured =
        envStatus.api_key?.configured && envStatus.api_secret?.configured;
      bybitBadge.textContent = allConfigured ? "설정됨" : "미설정";
      bybitBadge.className = allConfigured
        ? "status-badge status-badge--active"
        : "status-badge status-badge--inactive";
    }
  }

  // Update LLM provider statuses
  const llmProviders = ["openai", "gemini", "openrouter", "anthropic"];
  llmProviders.forEach((provider) => {
    const providerStatus = status[provider];
    const badge = el(`${provider}-status`);
    if (badge && providerStatus) {
      const configured =
        providerStatus.environments?.default?.api_key?.configured;
      badge.textContent = configured ? "설정됨" : "미설정";
      badge.className = configured
        ? "status-badge status-badge--active"
        : "status-badge status-badge--inactive";
    }
  });
}

function setupApiKeysTabs() {
  const tabs = document.querySelectorAll(".env-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      adminState.currentBybitEnv = tab.dataset.env;
      updateApiKeysUI();
    });
  });
}

function setupVisibilityToggles() {
  document.querySelectorAll(".toggle-visibility").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = btn.parentElement.querySelector("input");
      if (input) {
        input.type = input.type === "password" ? "text" : "password";
        btn.textContent = input.type === "password" ? "👁" : "🙈";
      }
    });
  });
}

async function saveApiKeys() {
  const hint = el("apikeys-hint");
  const keysToSave = [];

  // Collect Bybit keys
  const bybitPanel = document.querySelector('[data-provider="bybit"]');
  if (bybitPanel) {
    const apiKeyInput = bybitPanel.querySelector('[data-field="api_key"]');
    const apiSecretInput = bybitPanel.querySelector(
      '[data-field="api_secret"]'
    );
    const env = adminState.currentBybitEnv;

    if (apiKeyInput?.value) {
      keysToSave.push({
        provider: "bybit",
        key_type: "api_key",
        value: apiKeyInput.value,
        environment: env,
      });
    }
    if (apiSecretInput?.value) {
      keysToSave.push({
        provider: "bybit",
        key_type: "api_secret",
        value: apiSecretInput.value,
        environment: env,
      });
    }
  }

  // Collect LLM keys
  const llmProviders = ["openai", "gemini", "openrouter", "anthropic"];
  llmProviders.forEach((provider) => {
    const row = document.querySelector(
      `.llm-provider-row[data-provider="${provider}"]`
    );
    if (row) {
      const input = row.querySelector('[data-field="api_key"]');
      if (input?.value) {
        keysToSave.push({
          provider: provider,
          key_type: "api_key",
          value: input.value,
          environment: "default",
        });
      }
    }
  });

  if (!keysToSave.length) {
    if (hint) hint.textContent = "저장할 키가 없습니다";
    return;
  }

  try {
    if (hint) hint.textContent = "저장 중...";
    const result = await postJSON("/admin/api-keys/bulk", { keys: keysToSave });

    const allOk = result.results?.every((r) => r.ok);
    if (allOk) {
      if (hint) hint.textContent = "✓ 저장 완료";
      // Clear inputs after successful save
      document.querySelectorAll(".apikey-panel input").forEach((input) => {
        input.value = "";
      });
      // Reload status
      const apiKeysStatus = await fetchJSON("/admin/api-keys/status").catch(
        () => ({ status: {} })
      );
      adminState.apiKeysStatus = apiKeysStatus.status || {};
      updateApiKeysUI();
    } else {
      const failedCount = result.results?.filter((r) => !r.ok).length || 0;
      if (hint) hint.textContent = `⚠ ${failedCount}개 키 저장 실패`;
    }

    setTimeout(() => {
      if (hint && hint.textContent.startsWith("✓")) hint.textContent = "";
    }, 3000);
  } catch (err) {
    console.error(err);
    if (hint) hint.textContent = "✗ 저장 실패";
  }
}

async function refreshApiKeys() {
  try {
    const apiKeysStatus = await fetchJSON("/admin/api-keys/status");
    adminState.apiKeysStatus = apiKeysStatus.status || {};
    updateApiKeysUI();
    const hint = el("apikeys-hint");
    if (hint) {
      hint.textContent = "✓ 새로고침 완료";
      setTimeout(() => {
        if (hint.textContent === "✓ 새로고침 완료") hint.textContent = "";
      }, 2000);
    }
  } catch (err) {
    console.error(err);
  }
}

// Event Listeners
function setupEventListeners() {
  // Login form
  const loginForm = el("admin-login-form");
  if (loginForm) {
    loginForm.addEventListener("submit", handleLogin);
  }

  // Logout
  const logoutBtn = el("admin-logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", handleLogout);
  }

  // Agent config
  const saveAgentBtn = el("save-agent-config");
  const resetAgentBtn = el("reset-agent-config");
  if (saveAgentBtn) saveAgentBtn.addEventListener("click", saveAgentConfig);
  if (resetAgentBtn) resetAgentBtn.addEventListener("click", resetAgentConfig);

  // Scheduler
  const saveSchedulerBtn = el("save-scheduler");
  const resetSchedulerBtn = el("reset-scheduler");
  if (saveSchedulerBtn)
    saveSchedulerBtn.addEventListener("click", saveScheduler);
  if (resetSchedulerBtn)
    resetSchedulerBtn.addEventListener("click", resetScheduler);

  // System actions
  const closeAllBtn = el("close-all-positions");
  const syncStatsBtn = el("sync-stats");
  if (closeAllBtn) closeAllBtn.addEventListener("click", closeAllPositions);
  if (syncStatsBtn) syncStatsBtn.addEventListener("click", syncStats);

  // Refresh positions
  const refreshPosBtn = el("refresh-positions");
  if (refreshPosBtn) {
    refreshPosBtn.addEventListener("click", loadDashboardData);
  }

  // Blocked IPs
  const refreshBlockedBtn = el("refresh-blocked-ips");
  if (refreshBlockedBtn) {
    refreshBlockedBtn.addEventListener("click", loadBlockedIPs);
  }

  // API Keys
  const saveApiKeysBtn = el("save-apikeys");
  const refreshApiKeysBtn = el("refresh-apikeys");
  if (saveApiKeysBtn) saveApiKeysBtn.addEventListener("click", saveApiKeys);
  if (refreshApiKeysBtn)
    refreshApiKeysBtn.addEventListener("click", refreshApiKeys);

  // Scheduler Pause/Resume
  const pauseSchedulerBtn = el("pause-scheduler");
  const resumeSchedulerBtn = el("resume-scheduler");
  if (pauseSchedulerBtn)
    pauseSchedulerBtn.addEventListener("click", pauseScheduler);
  if (resumeSchedulerBtn)
    resumeSchedulerBtn.addEventListener("click", resumeScheduler);

  // Logs
  const refreshLogsBtn = el("refresh-logs");
  const logLevelFilter = el("log-level-filter");
  if (refreshLogsBtn) refreshLogsBtn.addEventListener("click", loadLogs);
  if (logLevelFilter) logLevelFilter.addEventListener("change", loadLogs);

  // Risk Config
  const saveRiskBtn = el("save-risk-config");
  const resetRiskBtn = el("reset-risk-config");
  if (saveRiskBtn) saveRiskBtn.addEventListener("click", saveRiskConfig);
  if (resetRiskBtn) resetRiskBtn.addEventListener("click", resetRiskConfig);

  // Trading Symbols
  const saveSymbolsBtn = el("save-symbols");
  const resetSymbolsBtn = el("reset-symbols");
  if (saveSymbolsBtn)
    saveSymbolsBtn.addEventListener("click", saveTradingSymbols);
  if (resetSymbolsBtn)
    resetSymbolsBtn.addEventListener("click", resetTradingSymbols);
}

// Initialize
window.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupEventListeners();
  setupApiKeysTabs();
  setupVisibilityToggles();
  checkAuth();

  // Auto refresh dashboard every 30 seconds
  setInterval(() => {
    if (adminState.authenticated && adminState.currentSection === "dashboard") {
      loadDashboardData();
    }
  }, 30000);

  // Auto refresh scheduler status every 10 seconds
  setInterval(() => {
    if (adminState.authenticated && adminState.currentSection === "scheduler") {
      refreshSchedulerStatus();
    }
  }, 10000);

  // Start log auto refresh
  startLogAutoRefresh();
});

async function refreshSchedulerStatus() {
  try {
    const scheduler = await fetchJSON("/admin/scheduler");
    adminState.scheduler = scheduler || {};
    updateSchedulerStatus(adminState.scheduler);
    updateSchedulerPausedUI(adminState.scheduler.paused);
  } catch (err) {
    console.warn("스케줄러 상태 새로고침 실패", err);
  }
}

// ===== Scheduler Pause/Resume =====
function updateSchedulerPausedUI(paused) {
  const pausedBadge = el("scheduler-paused-status");
  const pauseBtn = el("pause-scheduler");
  const resumeBtn = el("resume-scheduler");

  if (pausedBadge) {
    pausedBadge.textContent = paused ? "예" : "아니오";
    pausedBadge.className = paused
      ? "status-badge status-badge--warning"
      : "status-badge status-badge--inactive";
  }

  if (pauseBtn && resumeBtn) {
    pauseBtn.style.display = paused ? "none" : "inline-flex";
    resumeBtn.style.display = paused ? "inline-flex" : "none";
  }
}

async function pauseScheduler() {
  try {
    await postJSON("/admin/scheduler/pause", {});
    adminState.scheduler.paused = true;
    updateSchedulerPausedUI(true);
    alert("스케줄러가 일시 중단되었습니다.");
  } catch (err) {
    alert("스케줄러 중단 실패: " + err.message);
  }
}

async function resumeScheduler() {
  try {
    await postJSON("/admin/scheduler/resume", {});
    adminState.scheduler.paused = false;
    updateSchedulerPausedUI(false);
    alert("스케줄러가 재개되었습니다.");
  } catch (err) {
    alert("스케줄러 재개 실패: " + err.message);
  }
}

// ===== Logs Viewer =====
let logAutoRefreshInterval = null;

async function loadLogs() {
  const container = el("log-viewer");
  if (!container) return;

  const levelFilter = el("log-level-filter")?.value || "";
  const autoScroll = el("log-auto-scroll")?.checked ?? true;

  try {
    const url = levelFilter
      ? `/admin/logs?lines=500&level=${levelFilter}`
      : "/admin/logs?lines=500";
    const data = await fetchJSON(url);
    const logs = data.logs || [];

    if (!logs.length) {
      container.innerHTML = '<p class="muted">로그가 없습니다</p>';
      return;
    }

    const html = logs
      .map((log) => {
        const levelClass = `log-line--${(log.level || "info").toLowerCase()}`;
        return `<div class="log-line ${levelClass}">${escapeHtml(
          log.text
        )}</div>`;
      })
      .join("");

    container.innerHTML = html;

    if (autoScroll) {
      container.scrollTop = container.scrollHeight;
    }
  } catch (err) {
    console.error("로그 로드 실패:", err);
    container.innerHTML = '<p class="muted">로그를 불러올 수 없습니다</p>';
  }
}

function startLogAutoRefresh() {
  if (logAutoRefreshInterval) {
    clearInterval(logAutoRefreshInterval);
  }
  logAutoRefreshInterval = setInterval(() => {
    if (adminState.authenticated && adminState.currentSection === "logs") {
      loadLogs();
    }
  }, 10000); // 10초마다 새로고침
}

// ===== Risk Config =====
async function loadRiskConfig() {
  try {
    const data = await fetchJSON("/admin/risk-config");
    const config = data || {};

    const leverageInput = el("default-leverage");
    const maxLossInput = el("max-loss-percent");
    const allocationInput = el("position-allocation-percent");

    if (leverageInput) leverageInput.value = config.default_leverage ?? 5;
    if (maxLossInput) maxLossInput.value = config.max_loss_percent ?? 40;
    if (allocationInput)
      allocationInput.value = config.position_allocation_percent ?? 20;
  } catch (err) {
    console.warn("리스크 설정 로드 실패:", err);
  }
}

async function saveRiskConfig() {
  const hint = el("risk-config-hint");
  const leverageInput = el("default-leverage");
  const maxLossInput = el("max-loss-percent");
  const allocationInput = el("position-allocation-percent");

  const payload = {
    default_leverage: Number(leverageInput?.value || 5),
    max_loss_percent: Number(maxLossInput?.value || 40),
    position_allocation_percent: Number(allocationInput?.value || 20),
  };

  try {
    if (hint) hint.textContent = "저장 중...";
    await postJSON("/admin/risk-config", payload);
    if (hint) hint.textContent = "✓ 저장 완료";
    setTimeout(() => {
      if (hint && hint.textContent === "✓ 저장 완료") hint.textContent = "";
    }, 3000);
  } catch (err) {
    console.error(err);
    if (hint) hint.textContent = "✗ 저장 실패";
  }
}

function resetRiskConfig() {
  const leverageInput = el("default-leverage");
  const maxLossInput = el("max-loss-percent");
  const allocationInput = el("position-allocation-percent");

  if (leverageInput) leverageInput.value = 5;
  if (maxLossInput) maxLossInput.value = 40;
  if (allocationInput) allocationInput.value = 20;

  const hint = el("risk-config-hint");
  if (hint) hint.textContent = "기본값으로 초기화됨";
  setTimeout(() => {
    if (hint && hint.textContent === "기본값으로 초기화됨")
      hint.textContent = "";
  }, 2000);
}

// ===== Theme Toggle =====
function handleThemeToggle() {
  // theme.js의 toggleTheme 호출
  if (typeof window.toggleTheme === "function") {
    window.toggleTheme();
  } else if (typeof window.theme?.toggle === "function") {
    window.theme.toggle();
  } else {
    // 폴백: 직접 테마 전환
    const html = document.documentElement;
    const currentTheme = html.getAttribute("data-theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  }
  // 아이콘 업데이트
  updateThemeIcon();
}

function updateThemeIcon() {
  const html = document.documentElement;
  const currentTheme = html.getAttribute("data-theme") || "dark";
  const moonIcon = document.querySelector(".moon-icon");
  const sunIcon = document.querySelector(".sun-icon");

  if (moonIcon && sunIcon) {
    if (currentTheme === "dark") {
      // 다크모드: 달 아이콘 표시 (클릭하면 라이트로)
      moonIcon.style.display = "block";
      sunIcon.style.display = "none";
    } else {
      // 라이트모드: 해 아이콘 표시 (클릭하면 다크로)
      moonIcon.style.display = "none";
      sunIcon.style.display = "block";
    }
  }
}

// 페이지 로드 시 테마 아이콘 초기화
window.addEventListener("DOMContentLoaded", () => {
  // 테마 아이콘 초기화
  setTimeout(updateThemeIcon, 100);
});

// ===== Trading Symbols Management =====

async function loadTradingSymbols() {
  try {
    // 사용 가능한 심볼과 현재 설정된 심볼을 동시에 로드
    const [availableRes, currentRes] = await Promise.all([
      fetchJSON("/admin/trading-symbols/available"),
      fetchJSON("/admin/trading-symbols"),
    ]);

    adminState.availableSymbols = availableRes.symbols || [];
    adminState.selectedSymbols = currentRes.symbols || [];
    adminState.defaultSymbols = currentRes.defaults || [];

    renderSymbols();
    updateSymbolCount();
    setupSymbolSearch();
  } catch (err) {
    console.error("거래 심볼 로드 실패:", err);
    const container = el("available-symbols");
    if (container) {
      container.innerHTML = '<p class="muted">심볼을 불러올 수 없습니다</p>';
    }
  }
}

function renderSymbols(filterText = "") {
  const selectedContainer = el("selected-symbols");
  const availableContainer = el("available-symbols");

  if (!selectedContainer || !availableContainer) return;

  const filter = filterText.toLowerCase().trim();
  const selectedSet = new Set(adminState.selectedSymbols);

  // 선택된 심볼 렌더링
  if (adminState.selectedSymbols.length === 0) {
    selectedContainer.innerHTML = '<p class="muted">선택된 심볼이 없습니다</p>';
  } else {
    const selectedHtml = adminState.selectedSymbols
      .filter((s) => !filter || s.toLowerCase().includes(filter))
      .map((symbol) => {
        const isBTC = symbol === "BTCUSDT";
        return `
          <button
            class="symbol-chip symbol-chip--selected ${
              isBTC ? "symbol-chip--btc" : ""
            }"
            data-symbol="${escapeHtml(symbol)}"
            onclick="toggleSymbol('${escapeHtml(symbol)}')"
          >
            ${isBTC ? "⭐ " : ""}${escapeHtml(symbol)}
            <span class="symbol-remove">×</span>
          </button>
        `;
      })
      .join("");

    selectedContainer.innerHTML =
      selectedHtml || '<p class="muted">검색 결과가 없습니다</p>';
  }

  // 사용 가능한 심볼 렌더링 (선택되지 않은 것만)
  const availableSymbols = adminState.availableSymbols.filter(
    (s) => !selectedSet.has(s) && (!filter || s.toLowerCase().includes(filter))
  );

  if (availableSymbols.length === 0) {
    availableContainer.innerHTML = filter
      ? '<p class="muted">검색 결과가 없습니다</p>'
      : '<p class="muted">모든 심볼이 선택되었습니다</p>';
  } else {
    const availableHtml = availableSymbols
      .map((symbol) => {
        const isBTC = symbol === "BTCUSDT";
        return `
          <button
            class="symbol-chip ${isBTC ? "symbol-chip--btc-available" : ""}"
            data-symbol="${escapeHtml(symbol)}"
            onclick="toggleSymbol('${escapeHtml(symbol)}')"
          >
            ${isBTC ? "⭐ " : ""}${escapeHtml(symbol)}
          </button>
        `;
      })
      .join("");

    availableContainer.innerHTML = availableHtml;
  }
}

function toggleSymbol(symbol) {
  const index = adminState.selectedSymbols.indexOf(symbol);
  if (index > -1) {
    // 선택 해제
    adminState.selectedSymbols.splice(index, 1);
  } else {
    // 선택 추가
    adminState.selectedSymbols.push(symbol);
  }

  // BTCUSDT가 포함되어 있으면 맨 앞으로 이동
  const btcIndex = adminState.selectedSymbols.indexOf("BTCUSDT");
  if (btcIndex > 0) {
    adminState.selectedSymbols.splice(btcIndex, 1);
    adminState.selectedSymbols.unshift("BTCUSDT");
  }

  const searchInput = el("symbol-search");
  const filterText = searchInput?.value || "";
  renderSymbols(filterText);
  updateSymbolCount();
}

function updateSymbolCount() {
  const countEl = el("selected-symbol-count");
  if (countEl) {
    countEl.textContent = `${adminState.selectedSymbols.length}개 선택됨`;
  }
}

function setupSymbolSearch() {
  const searchInput = el("symbol-search");
  if (!searchInput) return;

  // 이벤트 리스너 중복 방지
  searchInput.removeEventListener("input", handleSymbolSearch);
  searchInput.addEventListener("input", handleSymbolSearch);
}

function handleSymbolSearch(e) {
  const filterText = e.target.value;
  renderSymbols(filterText);
}

async function saveTradingSymbols() {
  const hint = el("symbols-hint");

  if (adminState.selectedSymbols.length === 0) {
    if (hint) hint.textContent = "⚠ 최소 1개 이상의 심볼을 선택하세요";
    return;
  }

  try {
    if (hint) hint.textContent = "저장 중...";

    const result = await postJSON("/admin/trading-symbols", {
      symbols: adminState.selectedSymbols,
    });

    if (result.ok) {
      if (hint) {
        hint.textContent = result.warnings
          ? `✓ 저장 완료 (${result.warnings})`
          : "✓ 저장 완료";
      }
      setTimeout(() => {
        if (hint && hint.textContent.startsWith("✓")) hint.textContent = "";
      }, 3000);
    } else {
      if (hint) hint.textContent = "✗ 저장 실패";
    }
  } catch (err) {
    console.error("심볼 저장 실패:", err);
    if (hint) hint.textContent = "✗ 저장 실패";
  }
}

function resetTradingSymbols() {
  adminState.selectedSymbols = [...adminState.defaultSymbols];
  const searchInput = el("symbol-search");
  if (searchInput) searchInput.value = "";
  renderSymbols();
  updateSymbolCount();

  const hint = el("symbols-hint");
  if (hint) hint.textContent = "기본값으로 초기화됨";
  setTimeout(() => {
    if (hint && hint.textContent === "기본값으로 초기화됨")
      hint.textContent = "";
  }, 2000);
}
