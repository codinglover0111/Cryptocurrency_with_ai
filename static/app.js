async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return await res.json();
}

function el(id) {
  return document.getElementById(id);
}

function renderBalance(data) {
  const b = data.balance || {};
  el("balance").innerHTML = `
    <div>통화: ${b.currency || "USDT"}</div>
    <div>총액: ${b.total ?? "-"}</div>
    <div>가용: ${b.free ?? "-"}</div>
    <div>사용중: ${b.used ?? "-"}</div>
  `;
}

function renderPositions(data) {
  const list = (data.positions || [])
    .map((p) => {
      const sym = p.symbol || (p.info && p.info.symbol) || "-";
      const side = p.side || (p.info && p.info.side) || "-";
      const entry = p.entryPrice || (p.info && p.info.avgPrice) || "-";
      const size = p.contracts || p.amount || p.size || "-";
      return `<tr><td>${sym}</td><td>${side}</td><td>${entry}</td><td>${size}</td></tr>`;
    })
    .join("");
  el("positions").innerHTML = `
    <table>
      <thead><tr><th>심볼</th><th>사이드</th><th>진입가</th><th>수량</th></tr></thead>
      <tbody>${
        list || '<tr><td colspan="4" class="muted">없음</td></tr>'
      }</tbody>
    </table>`;
}

function renderOrders(data) {
  const list = (data.openOrders || [])
    .map((o) => {
      return `<tr><td>${o.symbol}</td><td>${o.side}</td><td>${o.type}</td><td>${
        o.price ?? "-"
      }</td><td>${o.amount ?? "-"}</td></tr>`;
    })
    .join("");
  el("orders").innerHTML = `
    <table>
      <thead><tr><th>심볼</th><th>사이드</th><th>타입</th><th>가격</th><th>수량</th></tr></thead>
      <tbody>${
        list || '<tr><td colspan="5" class="muted">없음</td></tr>'
      }</tbody>
    </table>`;
}

function renderStatsRange(data) {
  const fmtUSD = (n) =>
    n === null || n === undefined || Number.isNaN(n)
      ? "-"
      : new Intl.NumberFormat("en-US", {
          style: "currency",
          currency: "USD",
          maximumFractionDigits: 2,
        }).format(Number(n));
  const s = data.summary || {};
  const bySym = (data.by_symbol || [])
    .map(
      (r) =>
        `<tr><td>${r.symbol}</td><td>${r.trades}</td><td>${fmtUSD(
          r.realized_pnl
        )}</td></tr>`
    )
    .join("");
  const series = (data.series || [])
    .map(
      (r) =>
        `<tr><td>${r.t}</td><td>${r.trades}</td><td>${fmtUSD(
          r.realized_pnl
        )}</td></tr>`
    )
    .join("");
  el("stats").innerHTML = `
    <div class="kpis">
      <div class="kpi">
        <div class="label">실현손익</div>
        <div class="value">${fmtUSD(s.realized_pnl)}</div>
      </div>
      <div class="kpi">
        <div class="label">거래 수</div>
        <div class="value">${s.trades ?? 0}</div>
      </div>
      <div class="kpi">
        <div class="label">승률</div>
        <div class="value">${((s.win_rate || 0) * 100).toFixed(1)}%</div>
      </div>
      <div class="kpi">
        <div class="label">평균 손익</div>
        <div class="value">${fmtUSD(s.avg_pnl)}</div>
      </div>
    </div>
    <div class="grid-2">
      <div>
        <h3 class="subtle-heading">심볼별</h3>
        <table><thead><tr><th>심볼</th><th>거래수</th><th>실현손익</th></tr></thead><tbody>
          ${bySym || '<tr><td colspan="3" class="muted">없음</td></tr>'}
        </tbody></table>
      </div>
      <div>
        <h3 class="subtle-heading">시계열</h3>
        <table><thead><tr><th>기간</th><th>거래수</th><th>실현손익</th></tr></thead><tbody>
          ${series || '<tr><td colspan="3" class="muted">없음</td></tr>'}
        </tbody></table>
      </div>
    </div>
  `;
}

function getTzParam() {
  try {
    const url = new URL(window.location.href);
    return url.searchParams.get("tz");
  } catch (_) {
    return null;
  }
}

function normalizeUtcTimestamp(raw) {
  if (raw == null) return raw;
  let value = String(raw).trim();
  if (!value) return value;
  if (!value.includes("T") && value.includes(" ")) {
    value = value.replace(" ", "T");
  }
  if (!/[zZ]|[+-]\d{2}:?\d{2}$/.test(value)) {
    value = value + "Z";
  }
  return value;
}

function formatTimeWithTZ(tsISO, opts = {}) {
  try {
    const tz = getTzParam();
    const normalized = normalizeUtcTimestamp(tsISO);
    const base = new Date(normalized);
    if (Number.isNaN(base.getTime())) return String(tsISO);
    const baseOpts = Object.assign(
      {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: "UTC",
      },
      opts
    );

    let displayDate = base;
    let label = "UTC";
    if (tz && tz.startsWith("UTC")) {
      const m = tz.match(/^UTC([+-]\d{1,2})$/);
      if (m) {
        const offsetH = parseInt(m[1], 10);
        displayDate = new Date(base.getTime() + offsetH * 3600000);
        label = tz;
      }
    }

    const formatted = displayDate.toLocaleString("ko-KR", baseOpts);
    return `${formatted} ${label}`;
  } catch (_) {
    return String(tsISO);
  }
}

async function refreshAll() {
  try {
    const [status, stats, syms] = await Promise.all([
      fetchJSON("/status"),
      fetchJSON(buildStatsRangeUrl()),
      fetchJSON("/symbols"),
    ]);
    renderBalance(status);
    renderPositions(status);
    renderOrders(status);
    renderStatsRange(stats);
    const dl = el("symbols");
    if (dl && syms && Array.isArray(syms.symbols)) {
      dl.innerHTML = syms.symbols
        .map((s) => `<option value="${s.contract}">${s.code}</option>`)
        .join("");
    }
  } catch (e) {
    console.error(e);
  }
}

function buildStatsRangeUrl() {
  const since = el("st-since").value.trim();
  const until = el("st-until").value.trim();
  const symbol = el("st-symbol").value.trim();
  const group = el("st-group").value;
  const q = new URLSearchParams();
  if (since) q.set("since", since);
  if (until) q.set("until", until);
  if (symbol) q.set("symbol", symbol);
  if (group) q.set("group", group);
  return "/stats_range" + (q.toString() ? "?" + q.toString() : "");
}

async function manualRefreshStats() {
  try {
    const url = buildStatsRangeUrl();
    const data = await fetchJSON(url);
    renderStatsRange(data);
  } catch (e) {
    console.error(e);
  }
}

async function applyLeverage() {
  const symbol = el("symbol").value.trim();
  const leverage = Number(el("leverage").value);
  const body = { symbol, leverage, margin_mode: "cross" };
  try {
    const res = await fetch("/leverage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await res.json();
    el("action-result").textContent = "레버리지 적용 완료";
    refreshAll();
  } catch (e) {
    el("action-result").textContent = "오류: " + e.message;
  }
}

async function closeAll() {
  try {
    const res = await fetch("/close_all", { method: "POST" });
    const j = await res.json();
    el("action-result").textContent = "청산 시도 완료";
    refreshAll();
  } catch (e) {
    el("action-result").textContent = "오류: " + e.message;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  ensureJournalModalDOM();
  setupJournalDetailHandler();
  const refreshBtn = document.getElementById("jr-refresh");
  if (refreshBtn) refreshBtn.addEventListener("click", refreshJournals);
  const stRefresh = document.getElementById("st-refresh");
  if (stRefresh) stRefresh.addEventListener("click", manualRefreshStats);
  const stThisMonth = document.getElementById("st-this-month");
  if (stThisMonth)
    stThisMonth.addEventListener("click", () => {
      const d = new Date();
      const iso2 = (n) => String(n).padStart(2, "0");
      const y = d.getUTCFullYear();
      const m = iso2(d.getUTCMonth() + 1);
      const first = `${y}-${m}-01`;
      const next = new Date(
        Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1)
      );
      const until = `${next.getUTCFullYear()}-${iso2(
        next.getUTCMonth() + 1
      )}-01`;
      document.getElementById("st-since").value = first;
      document.getElementById("st-until").value = until;
      document.getElementById("st-group").value = "day";
      manualRefreshStats();
    });
  const stLast7d = document.getElementById("st-last-7d");
  if (stLast7d)
    stLast7d.addEventListener("click", () => {
      const now = new Date();
      const last7 = new Date(now.getTime() - 6 * 24 * 3600 * 1000);
      const iso = (x) =>
        `${x.getUTCFullYear()}-${String(x.getUTCMonth() + 1).padStart(
          2,
          "0"
        )}-${String(x.getUTCDate()).padStart(2, "0")}`;
      document.getElementById("st-since").value = iso(last7);
      document.getElementById("st-until").value = iso(
        new Date(now.getTime() + 24 * 3600 * 1000)
      );
      document.getElementById("st-group").value = "day";
      manualRefreshStats();
    });
  const typeFilters = document.getElementById("jr-type-filters");
  if (typeFilters && !typeFilters.__jrTypeAttached) {
    typeFilters.__jrTypeAttached = true;
    // 자동 적용 대신 조회 버튼으로만 실행
    // typeFilters.addEventListener("change", refreshJournals);
  }
  refreshAll();
  // 초기 자동 조회 1회는 유지
  refreshJournals();
  setInterval(refreshAll, 10000);
});

async function submitJournal() {
  const symbol = el("j-symbol").value.trim() || undefined;
  const entry_type = el("j-type").value;
  const content = el("j-content").value.trim();
  const reason = el("j-reason").value.trim() || undefined;
  try {
    const res = await fetch("/api/journals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, entry_type, content, reason }),
    });
    const j = await res.json();
    el("j-result").textContent = "기록 완료";
    el("j-content").value = "";
    el("j-reason").value = "";
  } catch (e) {
    el("j-result").textContent = "오류: " + e.message;
  }
}

async function refreshJournals() {
  try {
    const limit = Number(el("jr-limit")?.value || 20);
    const typeCheckboxes = getJournalTypeCheckboxes();
    const selectedTypes = typeCheckboxes
      .filter((cb) => cb.checked)
      .map((cb) => cb.value);
    const sort = document.getElementById("jr-sort")?.value || "desc";
    const ascFlag = sort === "asc" ? "1" : "0";
    const q = new URLSearchParams({
      limit: String(limit),
      today_only: "1",
      ascending: ascFlag,
    });
    if (
      selectedTypes.length > 0 &&
      selectedTypes.length < typeCheckboxes.length
    ) {
      q.set("types", selectedTypes.join(","));
    }
    // 안전 공개용 필터 API 사용 (types는 서버에서 필터링)
    const j = await fetchJSON(`/api/journals_filtered?${q.toString()}`);
    const items = j.items || [];
    const rows = items
      .map((it, idx) => {
        const tsStr = formatTimeWithTZ(it.ts);
        const title = `${(it.entry_type || "").toUpperCase()}${
          it.symbol ? " · " + it.symbol : ""
        }`;
        return `
          <tr>
            <td style="width: 120px" class="muted">${tsStr}</td>
            <td>${title}</td>
            <td style="width: 100px; text-align:right">
              <button class="btn secondary" data-action="jr-detail" data-item="${encodeURIComponent(
                JSON.stringify(it)
              )}">상세</button>
            </td>
          </tr>`;
      })
      .join("");
    el("journals").innerHTML = `
      <table>
        <thead><tr><th style="width:120px">시간</th><th>항목</th><th style="width:100px"></th></tr></thead>
        <tbody>${
          rows || '<tr><td colspan="3" class="muted">기록 없음</td></tr>'
        }</tbody>
      </table>`;
  } catch (e) {
    console.error(e);
  }
}

function escapeHtml(value) {
  const str = String(value ?? "");
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function ensureJournalModalDOM() {
  if (document.getElementById("jr-modal-backdrop")) return;
  const wrap = document.createElement("div");
  wrap.id = "jr-modal-backdrop";
  wrap.className = "modal-backdrop";
  wrap.hidden = true;
  wrap.innerHTML = `
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="jr-modal-title">
      <div class="modal-header">
        <div id="jr-modal-title">저널 상세</div>
        <button class="btn secondary" type="button" data-action="modal-close">닫기</button>
      </div>
      <div class="modal-body" id="jr-modal-body"></div>
    </div>
  `;
  wrap.addEventListener("click", (e) => {
    if (e.target === wrap || e.target.closest('[data-action="modal-close"]')) {
      closeJournalModal();
    }
  });
  document.body.appendChild(wrap);
  if (!window.__jrEscHandlerAttached) {
    window.__jrEscHandlerAttached = true;
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeJournalModal();
    });
  }
}

function getJournalTypeCheckboxes() {
  const container = document.getElementById("jr-type-filters");
  if (!container) return [];
  return Array.from(container.querySelectorAll('input[type="checkbox"]'));
}

function openJournalModal(html) {
  ensureJournalModalDOM();
  const backdrop = document.getElementById("jr-modal-backdrop");
  const body = document.getElementById("jr-modal-body");
  if (!backdrop || !body) return;
  body.innerHTML = html;
  backdrop.hidden = false;
}

function closeJournalModal() {
  const backdrop = document.getElementById("jr-modal-backdrop");
  if (backdrop) backdrop.hidden = true;
}

function showJournalModal(it) {
  try {
    const tsStr = formatTimeWithTZ(it.ts, {});
    const body = `
      <div><strong>시간</strong>: ${escapeHtml(tsStr)}</div>
      <div><strong>유형</strong>: ${escapeHtml(it.entry_type)}</div>
      <div><strong>심볼</strong>: ${escapeHtml(it.symbol || "-")}</div>
      ${
        it.reason
          ? `<div><strong>사유</strong>: ${escapeHtml(it.reason)}</div>`
          : ""
      }
      <div style="margin-top:8px"><strong>내용</strong></div>
      <pre>${escapeHtml(it.content || "-")}</pre>
      <div style="margin-top:8px"><strong>메타</strong></div>
      <pre>${escapeHtml(JSON.stringify(it.meta || {}, null, 2))}</pre>
    `;
    openJournalModal(body);
  } catch (e) {
    alert("보기 중 오류: " + e.message);
  }
}

function setupJournalDetailHandler() {
  const container = el("journals");
  if (!container || container.__jrClickAttached) return;
  container.__jrClickAttached = true;
  container.addEventListener("click", (e) => {
    const btn = e.target.closest('[data-action="jr-detail"]');
    if (!btn) return;
    try {
      const raw = decodeURIComponent(btn.dataset.item || "{}");
      const it = JSON.parse(raw);
      showJournalModal(it);
    } catch (err) {
      alert("보기 중 오류: " + err.message);
    }
  });
}
