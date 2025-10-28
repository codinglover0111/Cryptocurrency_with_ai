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

function renderStats(data) {
  el("stats").innerHTML = `
    <div>실현손익: ${data.realized_pnl}</div>
    <div>거래 수: ${data.trades}</div>
    <div>승률: ${(data.win_rate * 100).toFixed(1)}%</div>
    <div>평균 손익: ${data.avg_pnl}</div>
  `;
}

async function refreshAll() {
  try {
    const [status, stats, syms] = await Promise.all([
      fetchJSON("/status"),
      fetchJSON("/stats"),
      fetchJSON("/symbols"),
    ]);
    renderBalance(status);
    renderPositions(status);
    renderOrders(status);
    renderStats(stats);
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
  refreshAll();
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
    const q = new URLSearchParams({
      limit: String(limit),
      today_only: "1",
      ascending: "1",
    });
    const j = await fetchJSON(`/api/journals?${q.toString()}`);
    const items = j.items || [];
    const rows = items
      .map((it, idx) => {
        const ts = new Date(it.ts);
        const tsStr = ts.toLocaleString("ko-KR", {
          year: "2-digit",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "Asia/Seoul",
        });
        const title = `${(it.entry_type || "").toUpperCase()}${
          it.symbol ? " · " + it.symbol : ""
        }`;
        return `
          <tr>
            <td style="width: 120px" class="muted">${tsStr}</td>
            <td>${title}</td>
            <td style="width: 100px; text-align:right">
              <button class="btn secondary" onclick='showJournalDetail(${JSON.stringify(
                JSON.stringify(it)
              )})'>상세</button>
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

function showJournalDetail(raw) {
  try {
    const it = JSON.parse(raw);
    const ts = new Date(it.ts);
    const tsStr = ts.toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
    const body = `
시간: ${tsStr}\n유형: ${it.entry_type}\n심볼: ${it.symbol || "-"}\n사유: ${
      it.reason || "-"
    }\n내용:\n${it.content || "-"}\n\n메타:\n${JSON.stringify(
      it.meta || {},
      null,
      2
    )}
    `;
    // 간단 구현: alert. 필요시 모달로 교체 가능
    alert(body);
  } catch (e) {
    alert("보기 중 오류: " + e.message);
  }
}
