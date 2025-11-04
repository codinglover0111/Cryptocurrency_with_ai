function qs(name) {
  const url = new URL(window.location.href);
  return url.searchParams.get(name);
}

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return await res.json();
}

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  const d = digits;
  return Number(n).toFixed(d);
}

function render(items) {
  const board = document.getElementById("board");
  if (!board) return;
  if (!items || items.length === 0) {
    board.innerHTML = '<div class="card empty-state">포지션 없음</div>';
    fitToViewportHeight();
    return;
  }
  board.innerHTML = items
    .map((it) => {
      const pnlClass = it.pnl >= 0 ? "pnl-pos" : "pnl-neg";
      return `
        <div class="card">
          <div class="title">${it.symbol} · ${String(
        it.side || ""
      ).toUpperCase()}${it.leverage ? ` · ${it.leverage}x` : ""}</div>
          <div class="row"><span style="flex:1;min-width:0">진입가</span><span style="flex:0 0 auto;text-align:right">${fmt(
            it.entryPrice
          )}</span></div>
          <div class="row"><span style="flex:1;min-width:0">현재가</span><span style="flex:0 0 auto;text-align:right">${fmt(
            it.lastPrice
          )}</span></div>
          <div class="row"><span style="flex:1;min-width:0">수량</span><span style="flex:0 0 auto;text-align:right">${fmt(
            it.size,
            3
          )}</span></div>
          <div class="row"><span style="flex:1;min-width:0">익절가</span><span style="flex:0 0 auto;text-align:right">${fmt(
            it.tp
          )}</span></div>
          <div class="row"><span style="flex:1;min-width:0">손절가</span><span style="flex:0 0 auto;text-align:right">${fmt(
            it.sl
          )}</span></div>
          <div class="row"><span style="flex:1;min-width:0">손익</span><span style="flex:0 0 auto;text-align:right" class="${pnlClass}">${fmt(
        it.pnl,
        4
      )} (${it.pnlPct != null ? it.pnlPct.toFixed(2) : "-"}%)</span></div>
        </div>
      `;
    })
    .join("");
  fitToViewportHeight();
}

async function refresh() {
  const symbol = qs("symbol");
  const q = new URLSearchParams();
  if (symbol) q.set("symbol", symbol);
  const optMark = document.getElementById("opt_force_mark");
  const optEx = document.getElementById("opt_force_ex");
  const optRoe = document.getElementById("opt_force_roe");
  if (optMark) q.set("force_mark", optMark.checked ? "1" : "0");
  if (optEx) q.set("force_exchange_pnl", optEx.checked ? "1" : "0");
  if (optRoe) q.set("force_roe", optRoe.checked ? "1" : "0");
  const url = "/api/positions_summary?" + q.toString();
  try {
    const j = await fetchJSON(url);
    render(j.items || []);
  } catch (e) {
    console.error(e);
  }
}

function applyFontSizeFromQuery() {
  const fs = Number(qs("fs"));
  if (!Number.isNaN(fs) && fs > 0) {
    document.body.style.fontSize = fs + "px";
  }
}

function fitToViewportHeight() {
  const board = document.getElementById("board");
  if (!board) return;
  const startFs = parseFloat(getComputedStyle(document.body).fontSize);
  const queryFs = Number(qs("fs"));
  const maxFs = !Number.isNaN(queryFs) && queryFs > 0 ? queryFs : startFs;
  const minFsQuery = Number(qs("minfs"));
  const minFs = !Number.isNaN(minFsQuery) && minFsQuery > 0 ? minFsQuery : 12;
  let fs = parseFloat(getComputedStyle(document.body).fontSize) || maxFs;
  const availableH = window.innerHeight - 4;
  let guard = 0;
  while (board.scrollHeight > availableH && fs > minFs && guard < 40) {
    fs -= 1;
    document.body.style.fontSize = fs + "px";
    guard++;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  applyFontSizeFromQuery();
  refresh();
  const r = Number(qs("refresh") || 5);
  if (r > 0) setInterval(refresh, r * 1000);
  const applyBtn = document.getElementById("opt-apply");
  if (applyBtn) applyBtn.addEventListener("click", refresh);
});

window.addEventListener("resize", fitToViewportHeight);
