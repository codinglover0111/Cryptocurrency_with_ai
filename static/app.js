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
  const summary = Array.isArray(data.positionsSummary)
    ? data.positionsSummary
    : null;

  const fmtSide = (side) => {
    const raw = String(side || "").toLowerCase();
    if (raw === "long" || raw === "buy") return "롱";
    if (raw === "short" || raw === "sell") return "숏";
    return side || "-";
  };

  const usdFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });

  const fmtUSD = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return "-";
    const formatted = usdFormatter.format(num);
    return num > 0 && !formatted.startsWith("+") ? `+${formatted}` : formatted;
  };

  const fmtPct = (value) => {
    if (value === null || value === undefined) return "-";
    const num = Number(value);
    if (!Number.isFinite(num)) return "-";
    const digits = Math.abs(num) >= 100 ? 1 : 2;
    const body = num.toFixed(digits);
    return `${num > 0 ? "+" : ""}${body}%`;
  };

  const fmtNumber = (value) => {
    const brief = formatNumberBrief(value);
    if (brief != null) return brief;
    const num = Number(value);
    if (Number.isFinite(num)) return num.toString();
    return value ?? "-";
  };

  if (summary) {
    const rows = summary
      .map((item) => {
        const sym = item.symbol || "-";
        const side = fmtSide(item.side);
        const entry = fmtNumber(item.entryPrice);
        const size = fmtNumber(item.size);
        const pnlNum = Number(item.pnl);
        const pnlClass = Number.isFinite(pnlNum)
          ? pnlNum > 0
            ? "pnl-positive"
            : pnlNum < 0
            ? "pnl-negative"
            : ""
          : "";
        const pnlText = fmtUSD(item.pnl);
        const pctNum = Number(item.pnlPct);
        const pctClass = Number.isFinite(pctNum)
          ? pctNum > 0
            ? "pnl-positive"
            : pctNum < 0
            ? "pnl-negative"
            : ""
          : "";
        const pctText = fmtPct(item.pnlPct);
        return `<tr>
          <td>${sym}</td>
          <td>${side}</td>
          <td class="text-right">${entry}</td>
          <td class="text-right">${size}</td>
          <td class="text-right ${pnlClass}">${pnlText}</td>
          <td class="text-right ${pctClass}">${pctText}</td>
        </tr>`;
      })
      .join("");

    el("positions").innerHTML = `
      <table>
        <thead>
          <tr>
            <th>심볼</th>
            <th>사이드</th>
            <th>진입가</th>
            <th>수량</th>
            <th>PNL (USDT)</th>
            <th>수익률</th>
          </tr>
        </thead>
        <tbody>${
          rows || '<tr><td colspan="6" class="muted">없음</td></tr>'
        }</tbody>
      </table>`;
    return;
  }

  const list = (data.positions || [])
    .map((p) => {
      const info = p.info && typeof p.info === "object" ? p.info : {};
      const sym = p.symbol || info.symbol || "-";
      const sideRaw = p.side || info.side || "-";
      const side = fmtSide(sideRaw);
      const entryRaw = firstAvailableNumber([
        p.entryPrice,
        info.entryPrice,
        info.avgPrice,
        info.averagePrice,
      ]);
      const sizeRaw = firstAvailableNumber([
        p.contracts,
        p.amount,
        p.size,
        info.contracts,
        info.amount,
        info.size,
        info.positionAmt,
      ]);
      const lastRaw = firstAvailableNumber([
        p.lastPrice,
        p.markPrice,
        info.lastPrice,
        info.markPrice,
        info.price,
      ]);
      let pnlRaw = firstAvailableNumber([
        p.pnl,
        p.unrealizedPnl,
        info.unrealisedPnl,
        info.unrealizedPnl,
      ]);
      const sideLower = String(sideRaw || "").toLowerCase();
      if (
        pnlRaw == null &&
        entryRaw != null &&
        sizeRaw != null &&
        lastRaw != null &&
        (sideLower === "long" ||
          sideLower === "buy" ||
          sideLower === "short" ||
          sideLower === "sell")
      ) {
        if (sideLower === "long" || sideLower === "buy") {
          pnlRaw = (lastRaw - entryRaw) * sizeRaw;
        } else {
          pnlRaw = (entryRaw - lastRaw) * sizeRaw;
        }
      }
      let pnlPctRaw = firstAvailableNumber([
        p.pnlPct,
        p.percentage,
        p.roe,
        info.pnlPct,
        info.unrealisedPnlPcnt,
        info.roe,
        info.percentage,
      ]);
      if (pnlPctRaw == null && pnlRaw != null) {
        const initMargin = firstAvailableNumber([
          p.initialMargin,
          p.margin,
          info.positionIM,
          info.positionInitialMargin,
          info.positionMargin,
        ]);
        if (initMargin != null && initMargin !== 0) {
          pnlPctRaw = (pnlRaw / initMargin) * 100;
        } else if (
          entryRaw != null &&
          lastRaw != null &&
          (sideLower === "long" ||
            sideLower === "buy" ||
            sideLower === "short" ||
            sideLower === "sell")
        ) {
          if (sideLower === "long" || sideLower === "buy") {
            pnlPctRaw = ((lastRaw - entryRaw) / entryRaw) * 100;
          } else {
            pnlPctRaw = ((entryRaw - lastRaw) / entryRaw) * 100;
          }
        }
      }

      const entry = entryRaw != null ? fmtNumber(entryRaw) : "-";
      const size = sizeRaw != null ? fmtNumber(sizeRaw) : "-";
      const pnlNum = pnlRaw == null ? NaN : Number(pnlRaw);
      const pnlClass = Number.isFinite(pnlNum)
        ? pnlNum > 0
          ? "pnl-positive"
          : pnlNum < 0
          ? "pnl-negative"
          : ""
        : "";
      const pnlText = fmtUSD(pnlRaw);
      const pctNum = pnlPctRaw == null ? NaN : Number(pnlPctRaw);
      const pctClass = Number.isFinite(pctNum)
        ? pctNum > 0
          ? "pnl-positive"
          : pctNum < 0
          ? "pnl-negative"
          : ""
        : "";
      const pctText = fmtPct(pnlPctRaw);

      return `<tr>
        <td>${sym}</td>
        <td>${side}</td>
        <td class="text-right">${entry}</td>
        <td class="text-right">${size}</td>
        <td class="text-right ${pnlClass}">${pnlText}</td>
        <td class="text-right ${pctClass}">${pctText}</td>
      </tr>`;
    })
    .join("");
  el("positions").innerHTML = `
    <table>
      <thead><tr><th>심볼</th><th>사이드</th><th>진입가</th><th>수량</th><th>PNL (USDT)</th><th>수익률</th></tr></thead>
      <tbody>${
        list || '<tr><td colspan="6" class="muted">없음</td></tr>'
      }</tbody>
    </table>`;
}

const BALANCE_REFRESH_COOLDOWN_MS = 5000;
let lastManualStatusRefreshTime = 0;

async function refreshStatusOnly() {
  const status = await fetchJSON("/status");
  renderBalance(status);
  renderPositions(status);
  return status;
}

async function manualRefreshStatus() {
  const btn = el("balance-refresh");
  const hint = el("balance-refresh-hint");
  const now = Date.now();
  const elapsed = now - lastManualStatusRefreshTime;
  const withinCooldown = elapsed < BALANCE_REFRESH_COOLDOWN_MS;
  let fetchedSuccessfully = false;

  if (btn) {
    btn.disabled = true;
  }

  if (hint) {
    if (withinCooldown) {
      const waitMs = BALANCE_REFRESH_COOLDOWN_MS - elapsed;
      const waitSec = Math.max(1, Math.ceil(waitMs / 1000));
      hint.textContent = `최근에 새로고침했습니다 · ${waitSec}초 후 새로운 데이터 적용`;
    } else {
      hint.textContent = "새로고침 중...";
    }
  }

  try {
    await refreshStatusOnly();
    fetchedSuccessfully = true;
    if (!withinCooldown && hint) {
      hint.textContent = "업데이트 완료";
      setTimeout(() => {
        if (hint.textContent === "업데이트 완료") {
          hint.textContent = "";
        }
      }, 2000);
    }
  } catch (e) {
    console.error(e);
    if (hint) {
      hint.textContent = "새로고침 실패";
      setTimeout(() => {
        if (hint.textContent === "새로고침 실패") {
          hint.textContent = "";
        }
      }, 3000);
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.blur();
    }
    if (withinCooldown && hint) {
      setTimeout(() => {
        if (
          hint.textContent &&
          hint.textContent.includes("최근에 새로고침했습니다")
        ) {
          hint.textContent = "";
        }
      }, 2000);
    }
  }

  if (!withinCooldown && fetchedSuccessfully) {
    lastManualStatusRefreshTime = now;
  }
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

function formatDurationShort(seconds) {
  const num = Number(seconds);
  if (!Number.isFinite(num)) return "-";
  if (num <= 0) return "곧";
  const total = Math.max(0, Math.floor(num));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const parts = [];
  if (hours) parts.push(`${hours}시간`);
  if (minutes) parts.push(`${minutes}분`);
  if (!hours && !minutes) parts.push(`${secs}초`);
  return parts.join(" ") || "곧";
}

function renderPendingReviews(payload) {
  const container = el("pending-reviews");
  if (!container) return;
  const items = payload && Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    container.innerHTML = '<div class="muted">대기 중인 리뷰가 없습니다.</div>';
    return;
  }

  const rows = items
    .map((item) => {
      const symbol = item.symbol || "-";
      const sideRaw = item.side || "";
      const side = statusLabel(String(sideRaw).toLowerCase()) || sideRaw || "-";
      const pnlNum = Number(item.pnl);
      const pnlText = formatNumberBrief(item.pnl) ?? item.pnl ?? "-";
      const pnlClass = Number.isFinite(pnlNum)
        ? pnlNum > 0
          ? "pnl-positive"
          : pnlNum < 0
          ? "pnl-negative"
          : ""
        : "";
      const closedAt =
        item && item.closed_ts ? formatTimeWithTZ(item.closed_ts) : "-";
      const readyAt =
        item && item.ready_at ? formatTimeWithTZ(item.ready_at) : "-";
      const stateRaw = String(item.state || "waiting");
      const isWaiting = stateRaw === "waiting";
      const stateLabel = isWaiting ? "대기" : "검토 필요";
      const remaining = isWaiting
        ? formatDurationShort(item.wait_seconds)
        : "즉시 가능";

      return `<tr>
        <td>${symbol}</td>
        <td>${side}</td>
        <td class="text-right ${pnlClass}">${pnlText}</td>
        <td>${closedAt}</td>
        <td>${readyAt}</td>
        <td>${stateLabel}<div class="muted">${remaining}</div></td>
      </tr>`;
    })
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>심볼</th>
          <th>사이드</th>
          <th class="text-right">PNL</th>
          <th>청산 시각</th>
          <th>리뷰 가능</th>
          <th>상태</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function getTzParam() {
  try {
    const url = new URL(window.location.href);
    return url.searchParams.get("tz");
  } catch (_) {
    return null;
  }
}

function getSelectedTZ() {
  // 우선순위: 셀렉트 UI 값 > URL 쿼리의 tz
  const sel = document.getElementById("tz-select");
  const v = sel && sel.value ? String(sel.value) : null;
  if (v && v.startsWith("UTC")) return v;
  const q = getTzParam();
  if (q && q.startsWith("UTC")) return q;
  return "UTC+9";
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
    const tz = getSelectedTZ();
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

function maybeParseJSON(value) {
  if (typeof value !== "string") return null;
  try {
    return JSON.parse(value);
  } catch (_) {
    return null;
  }
}

function normalizeDecisionStatus(value) {
  if (value == null) return null;
  const raw = String(value).trim().toLowerCase();
  if (!raw) return null;
  if (["long", "buy"].includes(raw)) return "long";
  if (["short", "sell"].includes(raw)) return "short";
  if (["hold", "watch"].includes(raw)) return "hold";
  if (["stop"].includes(raw)) return "stop";
  if (["skip"].includes(raw)) return "skip";
  return raw;
}

function firstAvailableNumber(values) {
  for (const v of values) {
    if (v === null || v === undefined || v === "") continue;
    const num = Number(v);
    if (Number.isFinite(num)) return num;
  }
  return null;
}

function extractDecisionInfo(item) {
  const info = {
    status: null,
    entry: null,
    tp: null,
    sl: null,
  };
  if (!item) return info;

  const meta = item.meta && typeof item.meta === "object" ? item.meta : {};
  const decisionMeta =
    meta.decision && typeof meta.decision === "object" ? meta.decision : null;
  const contentObj = maybeParseJSON(item.content);

  const statusCandidates = [
    item.decision_status,
    meta.status,
    meta.side,
    decisionMeta && (decisionMeta.status || decisionMeta.Status),
    decisionMeta && (decisionMeta.ai_status || decisionMeta.side),
    contentObj && contentObj.status,
  ];
  for (const cand of statusCandidates) {
    const norm = normalizeDecisionStatus(cand);
    if (norm) {
      info.status = norm;
      break;
    }
  }

  const entryCandidates = [
    item.decision_entry,
    meta.entry_price,
    meta.entry,
    decisionMeta &&
      (decisionMeta.entry || decisionMeta.entry_price || decisionMeta.price),
    contentObj &&
      (contentObj.entry || contentObj.entry_price || contentObj.price),
  ];
  info.entry = firstAvailableNumber(entryCandidates);

  const tpCandidates = [
    item.decision_tp,
    meta.tp,
    decisionMeta && decisionMeta.tp,
    contentObj && contentObj.tp,
  ];
  info.tp = firstAvailableNumber(tpCandidates);

  const slCandidates = [
    item.decision_sl,
    meta.sl,
    decisionMeta && decisionMeta.sl,
    contentObj && contentObj.sl,
  ];
  info.sl = firstAvailableNumber(slCandidates);

  return info;
}

function formatNumberBrief(value) {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  const abs = Math.abs(num);
  const options = {
    maximumFractionDigits: abs >= 100 ? 2 : abs >= 10 ? 4 : 6,
  };
  return num.toLocaleString("en-US", options);
}

const JR_ALLOWED_PAGE_SIZES = [10, 30, 50, 100];
const JR_DEFAULT_PAGE_SIZE = 30;

function getJournalState() {
  if (!window.__jrPaging) {
    window.__jrPaging = { page: 1, pageSize: JR_DEFAULT_PAGE_SIZE, total: 0 };
  }
  return window.__jrPaging;
}

function setJournalPage(page) {
  const state = getJournalState();
  const next = Number(page);
  state.page = Number.isFinite(next) && next > 0 ? Math.floor(next) : 1;
}

function setJournalPageSize(size) {
  const state = getJournalState();
  const next = Number(size);
  state.pageSize = JR_ALLOWED_PAGE_SIZES.includes(next)
    ? next
    : JR_DEFAULT_PAGE_SIZE;
  state.page = 1;
}

function resetJournalFilters() {
  const tzSelect = document.getElementById("tz-select");
  if (tzSelect) {
    tzSelect.value = "UTC+9";
  }

  const jrRange = document.getElementById("jr-range");
  if (jrRange) {
    jrRange.value = "recent15";
  }

  const jrSort = document.getElementById("jr-sort");
  if (jrSort) {
    jrSort.value = "desc";
  }

  const jrLimit = document.getElementById("jr-limit");
  if (jrLimit) {
    jrLimit.value = String(JR_DEFAULT_PAGE_SIZE);
  }

  const jrSymbol = document.getElementById("jr-symbol");
  if (jrSymbol) {
    jrSymbol.value = "";
  }

  getJournalTypeCheckboxes().forEach((cb) => {
    cb.checked = true;
  });

  getJournalStatusCheckboxes().forEach((cb) => {
    cb.checked = true;
  });

  setJournalPageSize(JR_DEFAULT_PAGE_SIZE);
  setJournalPage(1);
  refreshJournals();
}

function buildJournalPageList(current, totalPages, maxLength = 7) {
  if (totalPages <= maxLength) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }
  const pages = [1];
  let start = Math.max(2, current - 2);
  let end = Math.min(totalPages - 1, current + 2);

  if (start <= 2) {
    start = 2;
    end = Math.min(totalPages - 1, start + 4);
  }

  if (end >= totalPages - 1) {
    end = totalPages - 1;
    start = Math.max(2, end - 4);
  }

  if (start > 2) {
    pages.push("...");
  }
  for (let i = start; i <= end; i += 1) {
    pages.push(i);
  }
  if (end < totalPages - 1) {
    pages.push("...");
  }

  pages.push(totalPages);
  return pages;
}

function renderJournalPagination(meta) {
  const container = el("jr-pagination");
  if (!container) return;

  const total = Number(meta.total || 0);
  const pageSize = Number(meta.pageSize || JR_DEFAULT_PAGE_SIZE);
  const rawPage = Number(meta.page || 1);

  if (
    !Number.isFinite(total) ||
    total <= 0 ||
    !Number.isFinite(pageSize) ||
    pageSize <= 0
  ) {
    container.innerHTML = "";
    container.hidden = true;
    return;
  }

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (totalPages <= 1) {
    container.innerHTML = "";
    container.hidden = true;
    return;
  }

  const safePage = Math.min(Math.max(1, rawPage), totalPages);
  const pages = buildJournalPageList(safePage, totalPages);
  const buttons = [];

  if (safePage > 1) {
    buttons.push(
      `<button type="button" class="pager-btn" data-page="${
        safePage - 1
      }">이전</button>`
    );
  }

  pages.forEach((p) => {
    if (p === "...") {
      buttons.push('<span class="pager-ellipsis">…</span>');
      return;
    }
    const isActive = p === safePage;
    const cls = isActive ? "pager-btn pager-btn--active" : "pager-btn";
    buttons.push(
      `<button type="button" class="${cls}" data-page="${p}">${p}</button>`
    );
  });

  if (safePage < totalPages) {
    buttons.push(
      `<button type="button" class="pager-btn" data-page="${
        safePage + 1
      }">다음</button>`
    );
  }

  container.innerHTML = buttons.join("");
  container.hidden = false;
}

function statusLabel(status) {
  switch (status) {
    case "long":
      return "롱";
    case "short":
      return "숏";
    case "hold":
      return "홀드";
    case "stop":
      return "정지";
    case "skip":
      return "스킵";
    default:
      return status ? status.toUpperCase() : "";
  }
}

async function refreshAll() {
  try {
    const pendingPromise = fetchJSON("/api/reviews/pending").catch((err) => {
      console.warn("/api/reviews/pending 실패", err);
      return { items: [] };
    });

    const [status, stats, syms, pending] = await Promise.all([
      fetchJSON("/status"),
      fetchJSON(buildStatsRangeUrl()),
      fetchJSON("/symbols"),
      pendingPromise,
    ]);
    renderBalance(status);
    renderPositions(status);
    renderStatsRange(stats);
    renderPendingReviews(pending);
    const statsSymbolSelect = el("st-symbol");
    if (statsSymbolSelect && syms && Array.isArray(syms.symbols)) {
      const previousValue = statsSymbolSelect.value;
      const options = ['<option value="">전체</option>'];
      syms.symbols.forEach((s) => {
        const contract = s.contract || s.code || "";
        const label = s.code || contract || "-";
        options.push(
          `<option value="${contract}" data-code="${s.code || ""}" data-spot="${
            s.spot || ""
          }">${label}</option>`
        );
      });
      statsSymbolSelect.innerHTML = options.join("");
      if (previousValue) {
        const hasPrevious = syms.symbols.some(
          (s) => s.contract === previousValue || s.code === previousValue
        );
        statsSymbolSelect.value = hasPrevious ? previousValue : "";
      }
    }
    const journalSymbolSelect = el("jr-symbol");
    if (journalSymbolSelect && syms && Array.isArray(syms.symbols)) {
      const previousValue = journalSymbolSelect.value;
      const options = ['<option value="">전체</option>'];
      syms.symbols.forEach((s) => {
        const contract = s.contract || s.code || "";
        const label = s.code || contract || "-";
        options.push(
          `<option value="${contract}" data-code="${
            s.code || ""
          }">${label}</option>`
        );
      });
      journalSymbolSelect.innerHTML = options.join("");
      if (previousValue) {
        const hasPrevious = syms.symbols.some(
          (s) => s.contract === previousValue || s.code === previousValue
        );
        journalSymbolSelect.value = hasPrevious ? previousValue : "";
      }
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
  const btn = document.getElementById("st-refresh");
  const hint = document.getElementById("st-refresh-hint");
  const updateHint = (message, timeoutMs) => {
    if (!hint) return;
    hint.textContent = message || "";
    if (timeoutMs && timeoutMs > 0) {
      const captured = message;
      setTimeout(() => {
        if (hint.textContent === captured) {
          hint.textContent = "";
        }
      }, timeoutMs);
    }
  };

  if (btn) {
    btn.disabled = true;
  }

  const symbolSelect = document.getElementById("st-symbol");
  const symbolValue = symbolSelect ? symbolSelect.value.trim() : "";
  const reconcilePayload = symbolValue ? { symbol: symbolValue } : {};

  try {
    updateHint("주문 상태 동기화 중...");
    const res = await fetch("/stats/reconcile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reconcilePayload),
    });
    if (!res.ok) {
      throw new Error(`/stats/reconcile ${res.status}`);
    }
    try {
      await res.json();
    } catch (_) {
      // ignore json parse errors (optional)
    }
    updateHint("동기화 완료", 2500);
  } catch (err) {
    console.warn("/stats/reconcile 실패", err);
    updateHint("동기화 실패", 4000);
  }

  try {
    try {
      await fetchJSON("/stats");
    } catch (err) {
      console.warn("/stats 호출 실패", err);
    }
    const url = buildStatsRangeUrl();
    const data = await fetchJSON(url);
    renderStatsRange(data);
  } catch (e) {
    console.error(e);
  } finally {
    if (btn) {
      btn.disabled = false;
    }
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
  const balanceRefreshBtn = document.getElementById("balance-refresh");
  if (balanceRefreshBtn)
    balanceRefreshBtn.addEventListener("click", () => {
      manualRefreshStatus();
    });
  const refreshBtn = document.getElementById("jr-refresh");
  if (refreshBtn)
    refreshBtn.addEventListener("click", () => {
      setJournalPage(1);
      refreshJournals();
    });
  const resetBtn = document.getElementById("jr-reset");
  if (resetBtn)
    resetBtn.addEventListener("click", () => {
      resetJournalFilters();
    });
  const tzSel = document.getElementById("tz-select");
  if (tzSel)
    tzSel.addEventListener("change", () => {
      setJournalPage(1);
      refreshJournals();
    });
  const jrSymbol = document.getElementById("jr-symbol");
  if (jrSymbol)
    jrSymbol.addEventListener("change", () => {
      setJournalPage(1);
      refreshJournals();
    });
  const jrRange = document.getElementById("jr-range");
  if (jrRange)
    jrRange.addEventListener("change", () => {
      setJournalPage(1);
      refreshJournals();
    });
  const jrLimit = document.getElementById("jr-limit");
  if (jrLimit) {
    const state = getJournalState();
    if (!JR_ALLOWED_PAGE_SIZES.includes(Number(jrLimit.value))) {
      jrLimit.value = String(state.pageSize || JR_DEFAULT_PAGE_SIZE);
    }
    jrLimit.addEventListener("change", () => {
      setJournalPageSize(Number(jrLimit.value));
      refreshJournals();
    });
  }
  const stRefresh = document.getElementById("st-refresh");
  if (stRefresh) stRefresh.addEventListener("click", manualRefreshStats);
  const stSymbolSelect = document.getElementById("st-symbol");
  if (stSymbolSelect)
    stSymbolSelect.addEventListener("change", () => {
      manualRefreshStats();
    });
  const stToday = document.getElementById("st-today");
  if (stToday)
    stToday.addEventListener("click", () => {
      const now = new Date();
      const iso = (x) =>
        `${x.getUTCFullYear()}-${String(x.getUTCMonth() + 1).padStart(
          2,
          "0"
        )}-${String(x.getUTCDate()).padStart(2, "0")}`;
      const today = iso(now);
      const tomorrow = new Date(now.getTime() + 24 * 3600 * 1000);
      document.getElementById("st-since").value = today;
      document.getElementById("st-until").value = iso(tomorrow);
      document.getElementById("st-group").value = "day";
      manualRefreshStats();
    });
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
  const pagination = document.getElementById("jr-pagination");
  if (pagination && !pagination.__jrPagerAttached) {
    pagination.__jrPagerAttached = true;
    pagination.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-page]");
      if (!btn) return;
      const nextPage = Number(btn.dataset.page);
      if (!Number.isFinite(nextPage) || nextPage < 1) return;
      setJournalPage(nextPage);
      refreshJournals();
    });
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
  const state = getJournalState();
  try {
    const limitSelect = document.getElementById("jr-limit");
    if (limitSelect) {
      const selectedSize = Number(limitSelect.value);
      if (JR_ALLOWED_PAGE_SIZES.includes(selectedSize)) {
        if (state.pageSize !== selectedSize) {
          state.pageSize = selectedSize;
          state.page = 1;
        }
      } else {
        limitSelect.value = String(state.pageSize || JR_DEFAULT_PAGE_SIZE);
      }
    }

    const typeCheckboxes = getJournalTypeCheckboxes();
    const selectedTypes = typeCheckboxes
      .filter((cb) => cb.checked)
      .map((cb) => cb.value)
      .filter((value) => value && value !== "thought");
    if (typeCheckboxes.length > 0 && selectedTypes.length === 0) {
      state.total = 0;
      state.page = 1;
      el("journals").innerHTML = `
      <table>
        <thead><tr><th style="width:220px">시간</th><th>항목</th><th style="width:100px"></th></tr></thead>
        <tbody><tr><td colspan="3" class="muted">기록 없음</td></tr></tbody>
      </table>`;
      renderJournalPagination({
        page: state.page,
        pageSize: state.pageSize,
        total: state.total,
      });
      return;
    }
    const statusCheckboxes = getJournalStatusCheckboxes();
    const selectedStatuses = statusCheckboxes
      .filter((cb) => cb.checked)
      .map((cb) => cb.value);
    const sort = document.getElementById("jr-sort")?.value || "desc";
    const ascFlag = sort === "asc" ? "1" : "0";
    const rangeValue = document.getElementById("jr-range")?.value || "today";

    const pageSize = state.pageSize || JR_DEFAULT_PAGE_SIZE;
    const currentPage = state.page || 1;

    const q = new URLSearchParams({
      limit: String(pageSize),
      page: String(currentPage),
      today_only: rangeValue === "today" ? "1" : "0",
      ascending: ascFlag,
    });
    if (rangeValue === "recent15") {
      q.set("recent_minutes", "15");
    }
    if (selectedTypes.length > 0) {
      q.set("types", selectedTypes.join(","));
    }
    const symbolValue = document.getElementById("jr-symbol")?.value?.trim();
    if (symbolValue) {
      q.set("symbol", symbolValue);
    }
    if (statusCheckboxes.length) {
      if (selectedStatuses.length === 0) {
        q.set("decision_statuses", "__none__");
      } else if (selectedStatuses.length < statusCheckboxes.length) {
        q.set("decision_statuses", selectedStatuses.join(","));
      }
    }

    const j = await fetchJSON(`/api/journals_filtered?${q.toString()}`);
    const items = j.items || [];

    const pageFromResponse = Number(j.page);
    if (Number.isFinite(pageFromResponse) && pageFromResponse > 0) {
      state.page = Math.floor(pageFromResponse);
    }
    const pageSizeFromResponse = Number(j.page_size);
    if (JR_ALLOWED_PAGE_SIZES.includes(pageSizeFromResponse)) {
      state.pageSize = pageSizeFromResponse;
      if (limitSelect && limitSelect.value !== String(pageSizeFromResponse)) {
        limitSelect.value = String(pageSizeFromResponse);
      }
    }
    const totalFromResponse = Number(j.total);
    if (Number.isFinite(totalFromResponse) && totalFromResponse >= 0) {
      state.total = totalFromResponse;
    } else {
      state.total = items.length;
    }

    const totalPages =
      state.pageSize > 0 ? Math.ceil(state.total / state.pageSize) : 0;
    if (
      items.length === 0 &&
      state.total > 0 &&
      totalPages > 0 &&
      state.page > totalPages &&
      !refreshJournals.__adjusting
    ) {
      refreshJournals.__adjusting = true;
      try {
        setJournalPage(totalPages);
        await refreshJournals();
      } finally {
        refreshJournals.__adjusting = false;
      }
      return;
    }

    const statusFilterEmpty =
      statusCheckboxes.length > 0 && selectedStatuses.length === 0;
    const statusFilterSet =
      statusCheckboxes.length > 0 &&
      selectedStatuses.length > 0 &&
      selectedStatuses.length < statusCheckboxes.length
        ? new Set(selectedStatuses)
        : null;

    const rows = [];
    for (const it of items) {
      const entryTypeRaw = it.entry_type || "";
      const entryTypeLower = entryTypeRaw.toLowerCase();
      if (entryTypeLower === "thought") {
        continue;
      }
      const decision = extractDecisionInfo(it);

      if (statusFilterEmpty) {
        continue;
      }
      if (statusFilterSet) {
        if (entryTypeLower !== "decision") {
          continue;
        }
        if (!decision.status || !statusFilterSet.has(decision.status)) {
          continue;
        }
      }

      const tsStr = formatTimeWithTZ(it.ts);
      const entryType = entryTypeRaw.toUpperCase();
      const symbol = it.symbol ? " · " + it.symbol : "";
      let extra = "";
      if (entryTypeLower === "decision") {
        const label = statusLabel(decision.status);
        if (label) {
          extra += ` · ${label}`;
        }
        if (decision.status === "long" || decision.status === "short") {
          const parts = [];
          const entryFmt = formatNumberBrief(decision.entry);
          if (entryFmt) parts.push(`진입 ${entryFmt}`);
          const tpFmt = formatNumberBrief(decision.tp);
          if (tpFmt) parts.push(`TP ${tpFmt}`);
          const slFmt = formatNumberBrief(decision.sl);
          if (slFmt) parts.push(`SL ${slFmt}`);
          if (parts.length) {
            extra += ` · ${parts.join(" / ")}`;
          }
        }
      }
      const title = `${entryType}${symbol}${extra}`;
      rows.push(`
          <tr>
            <td style="width: 220px" class="muted" title="${escapeHtml(
              tsStr
            )}">${tsStr}</td>
            <td>${title}</td>
            <td style="width: 100px; text-align:right">
              <button class="btn secondary" data-action="jr-detail" data-item="${encodeURIComponent(
                JSON.stringify(it)
              )}">상세</button>
            </td>
          </tr>`);
    }
    const rowsHtml = rows.join("");
    el("journals").innerHTML = `
      <table>
        <thead><tr><th style="width:220px">시간</th><th>항목</th><th style="width:100px"></th></tr></thead>
        <tbody>${
          rowsHtml || '<tr><td colspan="3" class="muted">기록 없음</td></tr>'
        }</tbody>
      </table>`;

    renderJournalPagination({
      page: state.page,
      pageSize: state.pageSize,
      total: state.total,
    });
  } catch (e) {
    console.error(e);
  }
}

refreshJournals.__adjusting = false;

function escapeHtml(value) {
  const str = String(value ?? "");
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeHref(value) {
  if (value == null) return null;
  try {
    let href = String(value).trim();
    if (!href) return null;
    const lower = href.toLowerCase();
    if (lower.startsWith("javascript:") || lower.startsWith("data:")) {
      return null;
    }
    if (href.startsWith("#")) return href;
    if (href.startsWith("/")) return href;
    const parsed = new URL(href, window.location.origin);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch (_) {
    return null;
  }
  return null;
}

function applyInlineMarkdown(line) {
  if (!line) return "";
  const placeholderMatch =
    line.trim() && /^@@CODE_BLOCK_\d+@@$/.test(line.trim());
  if (placeholderMatch) return line.trim();

  let safe = escapeHtml(line);

  safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
    const href = sanitizeHref(url);
    if (!href) return label;
    return `<a href="${escapeHtml(
      href
    )}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  safe = safe.replace(/`([^`]+)`/g, (match, code) => `<code>${code}</code>`);

  safe = safe.replace(
    /\*\*([\s\S]+?)\*\*/g,
    (match, text) => `<strong>${text}</strong>`
  );
  safe = safe.replace(
    /__([\s\S]+?)__/g,
    (match, text) => `<strong>${text}</strong>`
  );

  safe = safe.replace(
    /(^|[^*])\*([^*\n]+)\*(?!\*)/g,
    (match, prefix, text) => `${prefix}<em>${text}</em>`
  );
  safe = safe.replace(
    /(^|[^_])_([^_\n]+)_(?!_)/g,
    (match, prefix, text) => `${prefix}<em>${text}</em>`
  );

  return safe;
}

function renderMarkdownToHtml(raw) {
  if (raw === null || raw === undefined) {
    return '<div class="markdown-body"><p class="muted">내용 없음</p></div>';
  }

  let text = String(raw);
  if (!text.trim()) {
    return '<div class="markdown-body"><p class="muted">내용 없음</p></div>';
  }

  text = text
    .replace(/\r\n/g, "\n")
    .replace(/\t/g, "    ")
    .replace(/\\n/g, "\n");

  const codeBlocks = [];
  text = text.replace(/```([\s\S]*?)```/g, (_, code) => {
    const index = codeBlocks.length;
    const cleaned = code.replace(/^\s*[\r\n]?/, "").replace(/[\r\n\s]*$/, "");
    codeBlocks.push(`<pre><code>${escapeHtml(cleaned)}</code></pre>`);
    return `@@CODE_BLOCK_${index}@@`;
  });

  const lines = text.split("\n");
  const blocks = [];
  let listBuffer = [];
  let paragraphBuffer = [];
  let lineIndex = 0;
  let listBufferSince = null;
  let paragraphBufferSince = null;

  const flushParagraph = () => {
    if (!paragraphBuffer.length) return;
    const paragraph = paragraphBuffer.join("<br />");
    blocks.push(`<p>${paragraph}</p>`);
    paragraphBuffer = [];
    paragraphBufferSince = null;
  };

  const flushList = () => {
    if (!listBuffer.length) return;
    blocks.push(`<ul>${listBuffer.join("")}</ul>`);
    listBuffer = [];
    listBufferSince = null;
  };

  const flushPendingInOrder = () => {
    if (!listBuffer.length && !paragraphBuffer.length) return;
    const pending = [];
    if (listBuffer.length) {
      pending.push({ type: "list", since: listBufferSince ?? lineIndex });
    }
    if (paragraphBuffer.length) {
      pending.push({
        type: "paragraph",
        since: paragraphBufferSince ?? lineIndex,
      });
    }
    pending
      .sort((a, b) => a.since - b.since)
      .forEach((entry) => {
        if (entry.type === "list") {
          flushList();
        } else {
          flushParagraph();
        }
      });
  };

  lines.forEach((line) => {
    lineIndex += 1;
    const trimmed = line.trim();

    if (trimmed && /^@@CODE_BLOCK_\d+@@$/.test(trimmed)) {
      flushPendingInOrder();
      blocks.push(trimmed);
      return;
    }

    if (!trimmed) {
      flushPendingInOrder();
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushPendingInOrder();
      const level = headingMatch[1].length;
      const content = headingMatch[2];
      blocks.push(`<h${level}>${applyInlineMarkdown(content)}</h${level}>`);
      return;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      flushParagraph();
      const itemText = trimmed.replace(/^[-*+]\s+/, "");
      listBuffer.push(`<li>${applyInlineMarkdown(itemText)}</li>`);
      if (listBufferSince === null) {
        listBufferSince = lineIndex;
      }
      return;
    }

    if (/^>\s?/.test(trimmed)) {
      flushPendingInOrder();
      const quoteText = trimmed.replace(/^>\s?/, "");
      blocks.push(`<blockquote>${applyInlineMarkdown(quoteText)}</blockquote>`);
      return;
    }

    if (/^( {4}|\t)/.test(line)) {
      flushPendingInOrder();
      blocks.push(
        `<pre><code>${escapeHtml(line.replace(/^( {4}|\t)/, ""))}</code></pre>`
      );
      return;
    }

    paragraphBuffer.push(applyInlineMarkdown(line));
    if (paragraphBufferSince === null) {
      paragraphBufferSince = lineIndex;
    }
  });

  flushPendingInOrder();

  if (!blocks.length) {
    return '<div class="markdown-body"><p class="muted">내용 없음</p></div>';
  }

  let html = blocks.join("");
  codeBlocks.forEach((blockHtml, index) => {
    const placeholder = `@@CODE_BLOCK_${index}@@`;
    html = html.replace(placeholder, blockHtml);
  });

  return `<div class="markdown-body">${html}</div>`;
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

function getJournalStatusCheckboxes() {
  const container = document.getElementById("jr-status-filters");
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
    const symbol = it.symbol || "";
    const decision = extractDecisionInfo(it);
    const reasonSection = it.reason
      ? `
      <div style="margin-top:8px"><strong>사유</strong></div>
      ${renderMarkdownToHtml(it.reason)}
    `
      : "";
    const contentSection = `
      <div style="margin-top:8px"><strong>내용</strong></div>
      ${renderMarkdownToHtml(it.content)}
    `;
    const body = `
      <div><strong>시간</strong>: ${escapeHtml(tsStr)}</div>
      <div><strong>유형</strong>: ${escapeHtml(it.entry_type)}</div>
      <div><strong>심볼</strong>: ${escapeHtml(symbol || "-")}</div>
      <div id="jr-trade-info" class="muted" style="margin:6px 0 8px">거래 정보 조회 중...</div>
      ${reasonSection}
      ${contentSection}
      <div style=\"margin-top:8px\"><strong>메타</strong></div>
      <pre>${escapeHtml(JSON.stringify(it.meta || {}, null, 2))}</pre>
    `;
    openJournalModal(body);

    const tradeInfoEl = document.getElementById("jr-trade-info");

    const setTradeInfo = (e, t, s) => {
      if (!tradeInfoEl) return;
      const fmt = (v) => (v === undefined || v === null ? "-" : String(v));
      tradeInfoEl.classList.remove("muted");
      tradeInfoEl.innerHTML = `<strong>진입가</strong>: ${escapeHtml(
        fmt(e)
      )} · <strong>TP</strong>: ${escapeHtml(
        fmt(t)
      )} · <strong>SL</strong>: ${escapeHtml(fmt(s))}`;
    };

    if (decision.entry != null || decision.tp != null || decision.sl != null) {
      setTradeInfo(
        formatNumberBrief(decision.entry) || decision.entry,
        formatNumberBrief(decision.tp) || decision.tp,
        formatNumberBrief(decision.sl) || decision.sl
      );
      return;
    }
    if (tradeInfoEl) {
      tradeInfoEl.textContent = "거래 정보 없음";
    }
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
