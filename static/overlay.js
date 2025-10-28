async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return await res.json();
}

function qs(name) {
  const url = new URL(window.location.href);
  return url.searchParams.get(name);
}

function render(items) {
  const board = document.getElementById("board");
  if (!board) return;
  if (!items || items.length === 0) {
    board.innerHTML = '<div class="card">표시할 항목 없음</div>';
    return;
  }
  board.innerHTML = items
    .map((it) => {
      const ts = new Date(it.ts);
      const tz = qs("tz");
      const opt = {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: tz && tz.startsWith("UTC") ? "UTC" : undefined,
      };
      let tsStr;
      const m = tz && tz.match(/^UTC([+-]\d{1,2})$/);
      if (m) {
        const offsetH = parseInt(m[1], 10);
        const adj = new Date(ts.getTime() + offsetH * 3600000);
        tsStr = adj.toLocaleString(
          "ko-KR",
          Object.assign({}, opt, { timeZone: "UTC" })
        );
      } else {
        tsStr = ts.toLocaleString("ko-KR", opt);
      }
      const title = `${it.entry_type.toUpperCase()}${
        it.symbol ? " · " + it.symbol : ""
      }`;
      return `
        <div class="card">
          <div class="ts">${tsStr}</div>
          <div class="title">${title}</div>
          ${it.reason ? `<div class="reason">${it.reason}</div>` : ""}
          <div class="content">${(it.content || "").replace(/</g, "&lt;")}</div>
        </div>
      `;
    })
    .join("");
}

async function refresh() {
  const limit = Number(qs("limit") || 10);
  const symbol = qs("symbol");
  const types = qs("types");
  const today_only = Number(qs("today_only") || 1);
  const ascending = Number(qs("ascending") || 0);
  const url =
    `/api/journals_filtered?limit=${limit}&today_only=${today_only}&ascending=${ascending}` +
    (symbol ? `&symbol=${encodeURIComponent(symbol)}` : "") +
    (types ? `&types=${encodeURIComponent(types)}` : "");
  try {
    const j = await fetchJSON(url);
    render(j.items || []);
  } catch (e) {
    console.error(e);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  refresh();
  const r = Number(qs("refresh") || 5);
  if (r > 0) setInterval(refresh, r * 1000);
});
