/**
 * CryptoBot Public Dashboard - jQuery Version
 * Converted from inline script for CSP compliance
 */

// Store activity items for modal
let activityItems = [];

// ============================================
// Utility Functions
// ============================================

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderMarkdownToHtml(raw) {
  if (raw === null || raw === undefined) {
    return '<p class="no-data">내용 없음</p>';
  }

  let text = String(raw);
  if (!text.trim()) {
    return '<p class="no-data">내용 없음</p>';
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

  const applyInlineMarkdown = (line) => {
    let result = escapeHtml(line);
    result = result.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    result = result.replace(/\*(.+?)\*/g, "<em>$1</em>");
    result = result.replace(/`([^`]+)`/g, "<code>$1</code>");
    return result;
  };

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
      flushPendingInOrder();
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

    paragraphBuffer.push(applyInlineMarkdown(line));
    if (paragraphBufferSince === null) {
      paragraphBufferSince = lineIndex;
    }
  });

  flushPendingInOrder();

  if (!blocks.length) {
    return '<p class="no-data">내용 없음</p>';
  }

  let html = blocks.join("");
  codeBlocks.forEach((blockHtml, index) => {
    const placeholder = `@@CODE_BLOCK_${index}@@`;
    html = html.replace(placeholder, blockHtml);
  });

  return html;
}

function formatNumber(num, decimals = 2) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num);
}

function formatUSD(num) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  return "$" + formatNumber(num);
}

function formatPercent(num) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  const sign = num >= 0 ? "+" : "";
  return sign + formatNumber(num) + "%";
}

// ============================================
// Data Loading Functions
// ============================================

async function loadStatus() {
  try {
    const data = await $.getJSON("/status");

    if (data.balance) {
      const equity = parseFloat(data.balance.equity || data.balance.total || 0);
      $("#stat-balance").text(formatUSD(equity));

      const unrealized = parseFloat(data.balance.unrealized_pnl || 0);
      $("#stat-unrealized")
        .text(formatUSD(unrealized))
        .attr(
          "class",
          "stat-value " + (unrealized >= 0 ? "positive" : "negative")
        );
    }
  } catch (e) {
    console.error("Status load error:", e);
  }
}

async function loadStats() {
  try {
    const data = await $.getJSON("/stats");

    $("#stat-trades").text(data.total_trades || 0);

    const winrate = data.win_rate;
    if (winrate !== null && winrate !== undefined) {
      $("#stat-winrate").text(formatNumber(winrate) + "%");
    }
  } catch (e) {
    console.error("Stats load error:", e);
  }
}

async function loadPositions() {
  const $container = $("#positions-container");

  try {
    const data = await $.getJSON("/api/positions_summary");
    const items = data.items || [];

    if (items.length === 0) {
      $container.html(`
        <div class="empty-state">
          <div class="empty-state-icon">📭</div>
          <p>활성 포지션이 없습니다</p>
        </div>
      `);
      return;
    }

    let html = `
      <table class="positions-table">
        <thead>
          <tr>
            <th>심볼</th>
            <th>방향</th>
            <th>진입가</th>
            <th>현재가</th>
            <th>손익</th>
            <th>레버리지</th>
          </tr>
        </thead>
        <tbody>
    `;

    items.forEach((p) => {
      const side = (p.side || "").toLowerCase();
      const sideClass =
        side.includes("long") || side.includes("buy") ? "long" : "short";
      const sideLabel = sideClass === "long" ? "LONG" : "SHORT";
      const pnlClass = (p.pnl || 0) >= 0 ? "positive" : "negative";

      html += `
        <tr>
          <td>${p.symbol || "-"}</td>
          <td><span class="side-badge ${sideClass}">${sideLabel}</span></td>
          <td>${formatNumber(p.entryPrice, 4)}</td>
          <td>${formatNumber(p.lastPrice, 4)}</td>
          <td class="${pnlClass}">${formatUSD(p.pnl)} (${formatPercent(
        p.pnlPct
      )})</td>
          <td>${p.leverage || "-"}x</td>
        </tr>
      `;
    });

    html += "</tbody></table>";
    $container.html(html);
  } catch (e) {
    console.error("Positions load error:", e);
    $container.html(`
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>포지션을 불러올 수 없습니다</p>
      </div>
    `);
  }
}

async function loadActivity() {
  const $container = $("#activity-list");

  try {
    const data = await $.getJSON(
      "/api/journals_filtered?types=decision,action,review,error&limit=20&today_only=1"
    );
    const items = data.items || [];
    activityItems = items;

    if (items.length === 0) {
      $container.html(`
        <div class="empty-state">
          <div class="empty-state-icon">📝</div>
          <p>오늘의 활동이 없습니다</p>
        </div>
      `);
      return;
    }

    const icons = {
      decision: { icon: "🎯", class: "decision" },
      action: { icon: "⚡", class: "action" },
      review: { icon: "📋", class: "review" },
      error: { icon: "❌", class: "error" },
    };

    let html = "";
    items.slice(0, 15).forEach((item, index) => {
      const type = item.entry_type || "decision";
      const config = icons[type] || icons.decision;

      let timeStr = "-";
      if (item.ts) {
        const date = new Date(item.ts);
        timeStr = date.toLocaleTimeString("ko-KR", {
          hour: "2-digit",
          minute: "2-digit",
        });
      }

      let content = item.content || "";
      if (content.length > 60) content = content.substring(0, 60) + "...";

      html += `
        <div class="activity-item" data-action="show-analysis" data-index="${index}" title="클릭하여 상세 분석 보기">
          <div class="activity-icon ${config.class}">${config.icon}</div>
          <div class="activity-content">
            <div class="activity-title">${escapeHtml(
              item.symbol || "System"
            )} — ${escapeHtml(content)}</div>
            <div class="activity-meta">${timeStr} · ${type}</div>
          </div>
        </div>
      `;
    });

    $container.html(html);
  } catch (e) {
    console.error("Activity load error:", e);
    $container.html(`
      <div class="empty-state">
        <div class="empty-state-icon">⚠️</div>
        <p>활동을 불러올 수 없습니다</p>
      </div>
    `);
  }
}

// ============================================
// Modal Functions
// ============================================

function showAgentModal(index) {
  console.log("[showAgentModal] 호출됨, index:", index);

  const item = activityItems[index];
  console.log("[showAgentModal] item:", item);

  if (!item) {
    console.warn("[showAgentModal] item이 없습니다");
    return;
  }

  const $modal = $("#agent-modal");
  const $contentsEl = $("#agent-contents");
  console.log(
    "[showAgentModal] $modal 길이:",
    $modal.length,
    "$contentsEl 길이:",
    $contentsEl.length
  );

  const meta = item.meta || {};
  const agents = meta.agents || {};
  console.log("[showAgentModal] agents:", agents);

  const symbolLabel = item.symbol || meta.symbol || "System";
  $("#agent-modal-symbol").text(`(${symbolLabel})`);

  $contentsEl.html(`
    ${renderIndicatorContent(agents.indicator)}
    ${renderPatternContent(agents.pattern)}
    ${renderTrendContent(agents.trend)}
    ${renderDecisionContent(agents.decision || meta.decision || meta)}
  `);

  setupAgentTabs();

  // Force show modal
  $modal.removeAttr("hidden");
  $modal.prop("hidden", false);
  $modal.css("display", "flex");
  $("body").css("overflow", "hidden");
}

function closeAgentModal() {
  const $modal = $("#agent-modal");
  $modal.prop("hidden", true);
  $modal.attr("hidden", "");
  $modal.css("display", "none");
  $("body").css("overflow", "");
}

function setupAgentTabs() {
  const $tabs = $(".agent-tab");

  $tabs.off("click").on("click", function () {
    $tabs.removeClass("active");
    $(this).addClass("active");

    const tabName = $(this).data("tab");
    $(".agent-content").removeClass("active");
    $(`#agent-${tabName}`).addClass("active");
  });

  $('.agent-tab[data-tab="indicator"]').trigger("click");
}

function renderIndicatorContent(data) {
  if (!data) {
    return `
      <div class="agent-content" id="agent-indicator">
        <div class="agent-card">
          <div class="agent-card-header">
            <div class="agent-card-icon indicator">📊</div>
            <div class="agent-card-title">Indicator Agent</div>
          </div>
          <div class="no-data">분석 데이터가 없습니다</div>
        </div>
      </div>
    `;
  }

  const signalClass =
    data.macd_signal === "bullish"
      ? "bullish"
      : data.macd_signal === "bearish"
      ? "bearish"
      : "neutral";

  return `
    <div class="agent-content" id="agent-indicator">
      <div class="agent-card">
        <div class="agent-card-header">
          <div class="agent-card-icon indicator">📊</div>
          <div class="agent-card-title">Indicator Agent</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
          <div class="agent-field">
            <div class="agent-field-label">RSI</div>
            <div class="agent-field-value">${formatNumber(data.rsi, 2)}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">MACD Signal</div>
            <div class="agent-field-value ${signalClass}">${escapeHtml(
    data.macd_signal || "-"
  )}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Stochastic</div>
            <div class="agent-field-value">${formatNumber(
              data.stochastic,
              2
            )}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Bollinger Position</div>
            <div class="agent-field-value">${escapeHtml(
              data.bollinger_position || "-"
            )}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Momentum Score</div>
            <div class="agent-field-value">${formatNumber(
              data.momentum_score,
              4
            )}</div>
          </div>
        </div>
        ${
          data.summary
            ? `
          <div class="agent-field" style="margin-top: 16px;">
            <div class="agent-field-label">분석 요약</div>
            <div class="agent-analysis">${renderMarkdownToHtml(
              data.summary
            )}</div>
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;
}

function renderPatternContent(data) {
  if (!data) {
    return `
      <div class="agent-content" id="agent-pattern">
        <div class="agent-card">
          <div class="agent-card-header">
            <div class="agent-card-icon pattern">🔮</div>
            <div class="agent-card-title">Pattern Agent</div>
          </div>
          <div class="no-data">분석 데이터가 없습니다</div>
        </div>
      </div>
    `;
  }

  const signalClass =
    data.pattern_signal === "bullish"
      ? "bullish"
      : data.pattern_signal === "bearish"
      ? "bearish"
      : "neutral";
  const patterns = Array.isArray(data.patterns_found)
    ? data.patterns_found
    : [];

  return `
    <div class="agent-content" id="agent-pattern">
      <div class="agent-card">
        <div class="agent-card-header">
          <div class="agent-card-icon pattern">🔮</div>
          <div class="agent-card-title">Pattern Agent</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
          <div class="agent-field">
            <div class="agent-field-label">Pattern Signal</div>
            <div class="agent-field-value ${signalClass}">${escapeHtml(
    data.pattern_signal || "-"
  )}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Confidence</div>
            <div class="agent-field-value">${formatNumber(
              (data.confidence || 0) * 100,
              1
            )}%</div>
          </div>
        </div>
        ${
          patterns.length > 0
            ? `
          <div class="agent-field" style="margin-top: 16px;">
            <div class="agent-field-label">발견된 패턴</div>
            <div class="agent-field-value">${
              patterns.map((p) => escapeHtml(p)).join(", ") || "-"
            }</div>
          </div>
        `
            : ""
        }
        ${
          data.analysis
            ? `
          <div class="agent-field" style="margin-top: 16px;">
            <div class="agent-field-label">분석 내용</div>
            <div class="agent-analysis">${renderMarkdownToHtml(
              data.analysis
            )}</div>
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;
}

function renderTrendContent(data) {
  if (!data) {
    return `
      <div class="agent-content" id="agent-trend">
        <div class="agent-card">
          <div class="agent-card-header">
            <div class="agent-card-icon trend">📈</div>
            <div class="agent-card-title">Trend Agent</div>
          </div>
          <div class="no-data">분석 데이터가 없습니다</div>
        </div>
      </div>
    `;
  }

  const trendClass =
    data.trend_direction === "uptrend"
      ? "bullish"
      : data.trend_direction === "downtrend"
      ? "bearish"
      : "neutral";
  const supports = Array.isArray(data.support_levels)
    ? data.support_levels
    : [];
  const resistances = Array.isArray(data.resistance_levels)
    ? data.resistance_levels
    : [];

  return `
    <div class="agent-content" id="agent-trend">
      <div class="agent-card">
        <div class="agent-card-header">
          <div class="agent-card-icon trend">📈</div>
          <div class="agent-card-title">Trend Agent</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
          <div class="agent-field">
            <div class="agent-field-label">Trend Direction</div>
            <div class="agent-field-value ${trendClass}">${escapeHtml(
    data.trend_direction || "-"
  )}</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Volatility (ATR %)</div>
            <div class="agent-field-value">${formatNumber(
              data.volatility,
              2
            )}%</div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
          <div class="agent-field">
            <div class="agent-field-label">Support Levels</div>
            <div class="agent-field-value">${
              supports.length > 0
                ? supports.map((s) => formatNumber(s, 2)).join(", ")
                : "-"
            }</div>
          </div>
          <div class="agent-field">
            <div class="agent-field-label">Resistance Levels</div>
            <div class="agent-field-value">${
              resistances.length > 0
                ? resistances.map((r) => formatNumber(r, 2)).join(", ")
                : "-"
            }</div>
          </div>
        </div>
        ${
          data.analysis
            ? `
          <div class="agent-field" style="margin-top: 16px;">
            <div class="agent-field-label">분석 내용</div>
            <div class="agent-analysis">${renderMarkdownToHtml(
              data.analysis
            )}</div>
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;
}

function renderDecisionContent(data) {
  if (!data) {
    return `
      <div class="agent-content" id="agent-decision">
        <div class="agent-card">
          <div class="agent-card-header">
            <div class="agent-card-icon decision">🎯</div>
            <div class="agent-card-title">Decision Agent</div>
          </div>
          <div class="no-data">결정 데이터가 없습니다</div>
        </div>
      </div>
    `;
  }

  const status = (data.status || data.Status || "").toLowerCase();
  const statusClass =
    status === "long" ? "bullish" : status === "short" ? "bearish" : "neutral";

  return `
    <div class="agent-content" id="agent-decision">
      <div class="agent-card">
        <div class="agent-card-header">
          <div class="agent-card-icon decision">🎯</div>
          <div class="agent-card-title">Decision Agent</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px;">
          <div class="agent-field">
            <div class="agent-field-label">Status</div>
            <div class="agent-field-value ${statusClass}" style="font-weight: 700; font-size: 18px;">${escapeHtml(
    (data.status || data.Status || "-").toUpperCase()
  )}</div>
          </div>
          ${
            data.tp
              ? `
            <div class="agent-field">
              <div class="agent-field-label">Take Profit</div>
              <div class="agent-field-value bullish">${formatNumber(
                data.tp,
                4
              )}</div>
            </div>
          `
              : ""
          }
          ${
            data.sl
              ? `
            <div class="agent-field">
              <div class="agent-field-label">Stop Loss</div>
              <div class="agent-field-value bearish">${formatNumber(
                data.sl,
                4
              )}</div>
            </div>
          `
              : ""
          }
          ${
            data.leverage
              ? `
            <div class="agent-field">
              <div class="agent-field-label">Leverage</div>
              <div class="agent-field-value">${data.leverage}x</div>
            </div>
          `
              : ""
          }
        </div>
        ${
          data.explain
            ? `
          <div class="agent-field" style="margin-top: 16px;">
            <div class="agent-field-label">판단 근거</div>
            <div class="agent-analysis">${renderMarkdownToHtml(
              data.explain
            )}</div>
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;
}

// ============================================
// Refresh Function
// ============================================

function refreshAll() {
  loadStatus();
  loadStats();
  loadPositions();
  loadActivity();
}

// ============================================
// Document Ready - Event Bindings
// ============================================

$(document).ready(function () {
  console.log("[index.js] 초기화 시작");

  // Initial data load
  refreshAll();

  // Auto refresh every 30 seconds
  setInterval(refreshAll, 30000);

  // Event delegation for actions
  $(document).on("click", "[data-action]", function (e) {
    const $target = $(this);
    const action = $target.data("action");
    console.log("[index.js] 클릭 이벤트:", action);

    if (action === "refresh-all") {
      refreshAll();
    } else if (action === "refresh-positions") {
      loadPositions();
    } else if (action === "show-analysis") {
      const index = parseInt($target.data("index"), 10);
      console.log("[index.js] show-analysis, index:", index);
      if (!isNaN(index)) {
        showAgentModal(index);
      }
    } else if (action === "close-modal") {
      closeAgentModal();
    }
  });

  console.log("[index.js] 초기화 완료");

  // Close modal on backdrop click
  $("#agent-modal").on("click", function (e) {
    if ($(e.target).hasClass("modal-backdrop")) {
      closeAgentModal();
    }
  });

  // Close modal on Escape key
  $(document).on("keydown", function (e) {
    if (e.key === "Escape") {
      closeAgentModal();
    }
  });
});
