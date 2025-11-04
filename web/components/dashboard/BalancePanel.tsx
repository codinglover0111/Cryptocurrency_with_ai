import {
  firstAvailableNumber,
  formatCurrency,
  formatNumber,
  formatPercent,
} from "@/lib/format";
import type { PositionItem, StatusResponse } from "@/lib/types";
import { useMemo } from "react";

function normalizeSide(side: unknown): string {
  const raw = String(side ?? "").toLowerCase();
  if (raw === "long" || raw === "buy") return "롱";
  if (raw === "short" || raw === "sell") return "숏";
  if (!raw) return "-";
  return String(side);
}

function mapPositions(positions: PositionItem[] | null | undefined) {
  if (!positions || positions.length === 0) return [];
  return positions.map((p) => {
    const info = isRecord(p.info) ? (p.info as Record<string, unknown>) : {};
    const symbol = (p.symbol ?? info.symbol ?? "-") as string;
    const sideRaw = p.side ?? info.side;
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
      info.unrealisedPnl,
      info.unrealizedPnl,
    ]);
    const sideLower = String(sideRaw ?? "").toLowerCase();
    if (
      pnlRaw == null &&
      entryRaw != null &&
      sizeRaw != null &&
      lastRaw != null &&
      ["long", "buy", "short", "sell"].includes(sideLower)
    ) {
      pnlRaw =
        sideLower === "long" || sideLower === "buy"
          ? (lastRaw - entryRaw) * sizeRaw
          : (entryRaw - lastRaw) * sizeRaw;
    }

    let pctRaw = firstAvailableNumber([
      p.pnlPct,
      p.roe,
      info.pnlPct,
      info.unrealisedPnlPcnt,
      info.roe,
      info.percentage,
    ]);
    if (pctRaw == null && pnlRaw != null) {
      const initMargin = firstAvailableNumber([
        p.initialMargin,
        p.margin,
        info.positionIM,
        info.positionInitialMargin,
        info.positionMargin,
      ]);
      if (initMargin != null && initMargin !== 0) {
        pctRaw = (pnlRaw / initMargin) * 100;
      } else if (
        entryRaw != null &&
        lastRaw != null &&
        ["long", "buy", "short", "sell"].includes(sideLower)
      ) {
        pctRaw =
          sideLower === "long" || sideLower === "buy"
            ? ((lastRaw - entryRaw) / entryRaw) * 100
            : ((entryRaw - lastRaw) / entryRaw) * 100;
      }
    }

    return {
      symbol,
      sideLabel: normalizeSide(sideRaw ?? info.side),
      entry: formatNumber(entryRaw),
      size: formatNumber(sizeRaw),
      pnl: formatCurrency(pnlRaw),
      pnlValue: pnlRaw,
      pct: formatPercent(pctRaw),
      pctValue: pctRaw,
    };
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

interface BalancePanelProps {
  status: StatusResponse | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
  hintMessage?: string | null;
  cooldownSeconds: number;
}

export function BalancePanel({
  status,
  loading,
  onRefresh,
  hintMessage,
  cooldownSeconds,
}: BalancePanelProps) {
  const balance = status?.balance;
  const summaryRows = status?.positionsSummary ?? null;

  const positions = useMemo(() => {
    if (summaryRows && summaryRows.length > 0) {
      return summaryRows.map((row) => {
        const pnlValue = Number(row.pnl);
        const pctValue = Number(row.pnlPct);
        return {
          symbol: row.symbol ?? "-",
          sideLabel: normalizeSide(row.side),
          entry: formatNumber(row.entryPrice),
          size: formatNumber(row.size),
          pnl: formatCurrency(row.pnl),
          pnlValue,
          pct: formatPercent(row.pnlPct),
          pctValue,
        };
      });
    }
    return mapPositions(status?.positions);
  }, [status?.positions, summaryRows]);

  const cooldownActive = cooldownSeconds > 0;
  const hintText = hintMessage
    ? hintMessage
    : cooldownActive
    ? `${cooldownSeconds}초 후 새로고침 가능`
    : "";

  return (
    <article className="card panel panel--stack" id="card-balance">
      <header className="panel-header">
        <div>
          <h2 className="panel-title">잔고 / 포지션</h2>
          <p className="panel-subtitle">실시간 자산 현황</p>
        </div>
        <div className="panel-actions">
          <button
            type="button"
            className="btn btn--ghost"
            onClick={onRefresh}
            disabled={loading}
          >
            새로고침
          </button>
          <span className="panel-actions__hint">{hintText}</span>
        </div>
      </header>
      <div className="panel-body">
        <div className="balance-summary" id="balance">
          <div>통화: {balance?.currency ?? "USDT"}</div>
          <div>
            총액: {formatCurrency(balance?.total, { signed: false }) ?? "-"}
          </div>
          <div>
            가용: {formatCurrency(balance?.free, { signed: false }) ?? "-"}
          </div>
          <div>
            사용중: {formatCurrency(balance?.used, { signed: false }) ?? "-"}
          </div>
        </div>
        <div className="table-wrapper" id="positions">
          <table>
            <thead>
              <tr>
                <th>심볼</th>
                <th>사이드</th>
                <th className="text-right">진입가</th>
                <th className="text-right">수량</th>
                <th className="text-right">PNL (USDT)</th>
                <th className="text-right">수익률</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
                    className="muted"
                    style={{ textAlign: "center" }}
                  >
                    포지션이 없습니다.
                  </td>
                </tr>
              ) : (
                positions.map((row) => {
                  const pnlClass =
                    row.pnlValue && row.pnlValue !== 0
                      ? row.pnlValue > 0
                        ? "pnl-positive"
                        : "pnl-negative"
                      : "";
                  const pctClass =
                    row.pctValue && row.pctValue !== 0
                      ? row.pctValue > 0
                        ? "pnl-positive"
                        : "pnl-negative"
                      : "";
                  return (
                    <tr key={`${row.symbol}-${row.entry}-${row.size}`}>
                      <td>{row.symbol}</td>
                      <td>{row.sideLabel}</td>
                      <td className="text-right">{row.entry}</td>
                      <td className="text-right">{row.size}</td>
                      <td className={`text-right ${pnlClass}`.trim()}>
                        {row.pnl}
                      </td>
                      <td className={`text-right ${pctClass}`.trim()}>
                        {row.pct}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </article>
  );
}
