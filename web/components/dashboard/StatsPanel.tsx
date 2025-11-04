import { formatCurrency } from "@/lib/format";
import type { StatsRangeResponse, SymbolInfo } from "@/lib/types";

export interface StatsFilters {
  since: string;
  until: string;
  symbol: string;
  group: string;
}

interface StatsPanelProps {
  filters: StatsFilters;
  onFiltersChange: (patch: Partial<StatsFilters>) => void;
  onRefresh: () => Promise<void>;
  data: StatsRangeResponse | null;
  loading: boolean;
  symbols: SymbolInfo[];
}

export function StatsPanel({
  filters,
  onFiltersChange,
  onRefresh,
  data,
  loading,
  symbols,
}: StatsPanelProps) {
  const handleToday = () => {
    const now = new Date();
    const iso = (d: Date) =>
      `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(
        2,
        "0"
      )}-${String(d.getUTCDate()).padStart(2, "0")}`;
    const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);
    onFiltersChange({ since: iso(now), until: iso(tomorrow), group: "day" });
    void onRefresh();
  };

  const handleThisMonth = () => {
    const d = new Date();
    const y = d.getUTCFullYear();
    const m = d.getUTCMonth();
    const first = new Date(Date.UTC(y, m, 1));
    const next = new Date(Date.UTC(y, m + 1, 1));
    const iso = (date: Date) =>
      `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(
        2,
        "0"
      )}-${String(date.getUTCDate()).padStart(2, "0")}`;
    onFiltersChange({ since: iso(first), until: iso(next), group: "day" });
    void onRefresh();
  };

  const handleLast7d = () => {
    const now = new Date();
    const last7 = new Date(now.getTime() - 6 * 24 * 60 * 60 * 1000);
    const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);
    const iso = (date: Date) =>
      `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(
        2,
        "0"
      )}-${String(date.getUTCDate()).padStart(2, "0")}`;
    onFiltersChange({ since: iso(last7), until: iso(tomorrow), group: "day" });
    void onRefresh();
  };

  const summary = data?.summary ?? null;
  const bySymbol = data?.by_symbol ?? [];
  const series = data?.series ?? [];

  return (
    <article className="card panel" id="card-stats">
      <header className="panel-header">
        <div>
          <h2 className="panel-title">통계</h2>
          <p className="panel-subtitle">기간별 성과를 비교해 보세요.</p>
        </div>
      </header>
      <div className="panel-body">
        <div className="stats-grid">
          <div className="input-field">
            <label htmlFor="st-since">시작 (UTC YYYY-MM-DD)</label>
            <input
              id="st-since"
              placeholder="2025-01-01"
              value={filters.since}
              onChange={(event) =>
                onFiltersChange({ since: event.target.value })
              }
            />
          </div>
          <div className="input-field">
            <label htmlFor="st-until">종료 (UTC YYYY-MM-DD)</label>
            <input
              id="st-until"
              placeholder="2025-12-31"
              value={filters.until}
              onChange={(event) =>
                onFiltersChange({ until: event.target.value })
              }
            />
          </div>
          <div className="input-field">
            <label htmlFor="st-symbol">심볼 (선택)</label>
            <select
              id="st-symbol"
              value={filters.symbol}
              onChange={(event) => {
                onFiltersChange({ symbol: event.target.value });
                void onRefresh();
              }}
            >
              <option value="">전체</option>
              {symbols.map((symbol) => {
                const value = symbol.contract || symbol.code || "";
                const label = symbol.code || symbol.contract || "-";
                return (
                  <option key={value || label} value={value}>
                    {label}
                  </option>
                );
              })}
            </select>
          </div>
          <div className="input-field">
            <label htmlFor="st-group">그룹</label>
            <select
              id="st-group"
              value={filters.group}
              onChange={(event) => {
                onFiltersChange({ group: event.target.value });
                void onRefresh();
              }}
            >
              <option value="">없음</option>
              <option value="day">일별</option>
              <option value="week">주별</option>
              <option value="month">월별</option>
            </select>
          </div>
        </div>

        <div className="stats-actions">
          <button
            type="button"
            className="btn btn--ghost"
            onClick={onRefresh}
            disabled={loading}
            id="st-refresh"
          >
            조회
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleToday}
            disabled={loading}
            id="st-today"
          >
            오늘
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleThisMonth}
            disabled={loading}
            id="st-this-month"
          >
            이번 달
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleLast7d}
            disabled={loading}
            id="st-last-7d"
          >
            최근 7일
          </button>
        </div>

        <div className="stats-output" id="stats">
          <div className="kpis">
            <div className="kpi">
              <div className="label">실현손익</div>
              <div className="value">
                {formatCurrency(summary?.realized_pnl, { signed: true })}
              </div>
            </div>
            <div className="kpi">
              <div className="label">거래 수</div>
              <div className="value">{summary?.trades ?? 0}</div>
            </div>
            <div className="kpi">
              <div className="label">승률</div>
              <div className="value">
                {summary?.win_rate != null
                  ? `${(Number(summary.win_rate) * 100).toFixed(1)}%`
                  : "-"}
              </div>
            </div>
            <div className="kpi">
              <div className="label">평균 손익</div>
              <div className="value">
                {formatCurrency(summary?.avg_pnl, { signed: true })}
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div>
              <h3 className="subtle-heading">심볼별</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>심볼</th>
                      <th className="text-right">거래수</th>
                      <th className="text-right">실현손익</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bySymbol.length === 0 ? (
                      <tr>
                        <td
                          colSpan={3}
                          className="muted"
                          style={{ textAlign: "center" }}
                        >
                          데이터가 없습니다.
                        </td>
                      </tr>
                    ) : (
                      bySymbol.map((row) => (
                        <tr key={row.symbol}>
                          <td>{row.symbol}</td>
                          <td className="text-right">{row.trades ?? 0}</td>
                          <td className="text-right">
                            {formatCurrency(row.realized_pnl)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="subtle-heading">시계열</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>기간</th>
                      <th className="text-right">거래수</th>
                      <th className="text-right">실현손익</th>
                    </tr>
                  </thead>
                  <tbody>
                    {series.length === 0 ? (
                      <tr>
                        <td
                          colSpan={3}
                          className="muted"
                          style={{ textAlign: "center" }}
                        >
                          데이터가 없습니다.
                        </td>
                      </tr>
                    ) : (
                      series.map((row) => (
                        <tr key={row.t}>
                          <td>{row.t}</td>
                          <td className="text-right">{row.trades ?? 0}</td>
                          <td className="text-right">
                            {formatCurrency(row.realized_pnl)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}
