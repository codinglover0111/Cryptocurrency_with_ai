"use client";

import { Modal } from "@/components/common/Modal";
import {
  extractDecisionInfo,
  formatNumberBrief,
  formatTimeWithTZ,
} from "@/lib/format";
import type { JournalItem, JournalsResponse, SymbolInfo } from "@/lib/types";
import { useMemo, useState } from "react";

export interface JournalFilters {
  tz: string;
  range: "today" | "recent15";
  sort: "asc" | "desc";
  limit: number;
  page: number;
  symbol: string;
  types: Record<string, boolean>;
  statuses: Record<string, boolean>;
}

interface JournalsPanelProps {
  filters: JournalFilters;
  onFiltersChange: (
    patch: Partial<Omit<JournalFilters, "types" | "statuses">> & {
      types?: Partial<Record<string, boolean>>;
      statuses?: Partial<Record<string, boolean>>;
    },
    options?: { triggerRefresh?: boolean }
  ) => void;
  onRefresh: () => Promise<void>;
  onReset: () => void;
  data: JournalsResponse | null;
  loading: boolean;
  symbols: SymbolInfo[];
}

export const JOURNAL_PAGE_SIZES = [10, 30, 50, 100];
const DEFAULT_TYPES = ["decision", "action", "review", "error"];
const DEFAULT_STATUSES = ["long", "hold", "short"];

function buildPageList(current: number, totalPages: number, maxLength = 7) {
  if (totalPages <= maxLength) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }
  const pages: Array<number | "..."> = [1];
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
  for (let page = start; page <= end; page += 1) {
    pages.push(page);
  }
  if (end < totalPages - 1) {
    pages.push("...");
  }

  pages.push(totalPages);
  return pages;
}

function JournalModalContent({ item, tz }: { item: JournalItem; tz: string }) {
  const decision = extractDecisionInfo(item);
  const entry = formatNumberBrief(decision.entry);
  const tp = formatNumberBrief(decision.tp);
  const sl = formatNumberBrief(decision.sl);
  return (
    <div>
      <p>
        <strong>시간</strong>: {formatTimeWithTZ(item.ts, tz)}
      </p>
      <p>
        <strong>유형</strong>: {item.entry_type}
      </p>
      <p>
        <strong>심볼</strong>: {item.symbol ?? "-"}
      </p>
      {item.reason ? (
        <p>
          <strong>사유</strong>: {item.reason}
        </p>
      ) : null}
      {decision.status ? (
        <p>
          <strong>포지션</strong>: {decision.status.toUpperCase()}
        </p>
      ) : null}
      {(entry || tp || sl) && (
        <p>
          <strong>트레이드</strong>: {entry ? `진입 ${entry}` : ""}
          {tp ? ` · TP ${tp}` : ""}
          {sl ? ` · SL ${sl}` : ""}
        </p>
      )}
      <div>
        <strong>내용</strong>
        <pre>{item.content ?? "-"}</pre>
      </div>
      <div>
        <strong>메타</strong>
        <pre>{item.meta ? JSON.stringify(item.meta, null, 2) : "{}"}</pre>
      </div>
    </div>
  );
}

export function JournalsPanel({
  filters,
  onFiltersChange,
  onRefresh,
  onReset,
  data,
  loading,
  symbols,
}: JournalsPanelProps) {
  const [selected, setSelected] = useState<JournalItem | null>(null);

  const items = data?.items ?? [];
  const total = data?.total ?? items.length;
  const pageSize = data?.page_size ?? filters.limit;
  const currentPage = data?.page ?? filters.page;
  const totalPages = pageSize > 0 ? Math.max(1, Math.ceil(total / pageSize)) : 1;
  const pages = buildPageList(currentPage, totalPages);

  const timezoneOptions = useMemo(() => ["UTC", "UTC+9", "UTC+1", "UTC-5"], []);

  const handleTypeToggle = (type: string) => {
    const next = { ...filters.types, [type]: !filters.types[type] };
    onFiltersChange({ types: next });
  };

  const handleStatusToggle = (status: string) => {
    const next = { ...filters.statuses, [status]: !filters.statuses[status] };
    onFiltersChange({ statuses: next });
  };

  return (
    <section className="card panel">
      <header className="panel-header">
        <div>
          <h2 className="panel-title">저널 조회</h2>
          <p className="panel-subtitle">
            오늘 생성된 최신 기록을 시간순으로 확인하세요.
          </p>
        </div>
      </header>
      <div className="panel-body">
        <div className="journal-controls">
          <label className="journal-limit" htmlFor="jr-limit">
            페이지당 개수
            <select
              id="jr-limit"
              value={String(filters.limit)}
              onChange={(event) =>
                onFiltersChange(
                  {
                    limit: Number(event.target.value),
                    page: 1,
                  },
                  { triggerRefresh: true }
                )
              }
            >
              {JOURNAL_PAGE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}개
                </option>
              ))}
            </select>
          </label>
          <label className="journal-limit" htmlFor="tz-select">
            타임존
            <select
              id="tz-select"
              value={filters.tz}
              onChange={(event) =>
                onFiltersChange(
                  { tz: event.target.value },
                  { triggerRefresh: true }
                )
              }
            >
              {timezoneOptions.map((tz) => (
                <option key={tz} value={tz}>
                  {tz}
                </option>
              ))}
            </select>
          </label>
          <label className="journal-limit" htmlFor="jr-symbol">
            코인
            <select
              id="jr-symbol"
              value={filters.symbol}
              onChange={(event) =>
                onFiltersChange(
                  { symbol: event.target.value, page: 1 },
                  { triggerRefresh: true }
                )
              }
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
          </label>
          <label className="journal-sort" htmlFor="jr-sort">
            정렬
            <select
              id="jr-sort"
              value={filters.sort}
              onChange={(event) =>
                onFiltersChange(
                  { sort: event.target.value as "asc" | "desc", page: 1 },
                  { triggerRefresh: true }
                )
              }
            >
              <option value="desc">최신순</option>
              <option value="asc">오래된순</option>
            </select>
          </label>
          <label className="journal-sort" htmlFor="jr-range">
            기간
            <select
              id="jr-range"
              value={filters.range}
              onChange={(event) =>
                onFiltersChange(
                  { range: event.target.value as "today" | "recent15", page: 1 },
                  { triggerRefresh: true }
                )
              }
            >
              <option value="today">오늘</option>
              <option value="recent15">최근 15분</option>
            </select>
          </label>
          <div id="jr-type-filters" className="journal-type-filters">
            <span className="journal-type-label">유형</span>
            {DEFAULT_TYPES.map((type) => (
              <label key={type}>
                <input
                  type="checkbox"
                  checked={!!filters.types[type]}
                  onChange={() => handleTypeToggle(type)}
                />
                {type}
              </label>
            ))}
          </div>
          <div id="jr-status-filters" className="journal-type-filters">
            <span className="journal-type-label">포지션</span>
            {DEFAULT_STATUSES.map((status) => (
              <label key={status}>
                <input
                  type="checkbox"
                  checked={!!filters.statuses[status]}
                  onChange={() => handleStatusToggle(status)}
                />
                {status}
              </label>
            ))}
          </div>
          <button
            type="button"
            className="btn btn--ghost"
            id="jr-refresh"
            onClick={onRefresh}
            disabled={loading}
          >
            조회
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            id="jr-reset"
            onClick={onReset}
            disabled={loading}
          >
            초기화
          </button>
        </div>

        <div className="table-wrapper" id="journals">
          <table>
            <thead>
              <tr>
                <th style={{ width: "220px" }}>시간</th>
                <th>항목</th>
                <th className="text-right" style={{ width: "100px" }} />
              </tr>
            </thead>
            <tbody>
              {items.length === 0 ? (
                <tr>
                  <td colSpan={3} className="muted" style={{ textAlign: "center" }}>
                    기록이 없습니다.
                  </td>
                </tr>
              ) : (
                items.map((item) => {
                  const decision = extractDecisionInfo(item);
                  const statusLabel = decision.status
                    ? decision.status === "long"
                      ? "롱"
                      : decision.status === "short"
                      ? "숏"
                      : decision.status.toUpperCase()
                    : null;
                  const extraParts: string[] = [];
                  if (statusLabel) extraParts.push(statusLabel);
                  if (
                    decision.status === "long" ||
                    decision.status === "short"
                  ) {
                    const entry = formatNumberBrief(decision.entry);
                    const tp = formatNumberBrief(decision.tp);
                    const sl = formatNumberBrief(decision.sl);
                    const tradeParts = [
                      entry ? `진입 ${entry}` : null,
                      tp ? `TP ${tp}` : null,
                      sl ? `SL ${sl}` : null,
                    ].filter(Boolean);
                    if (tradeParts.length) {
                      extraParts.push(tradeParts.join(" / "));
                    }
                  }
                  const title = `${item.entry_type.toUpperCase()}${
                    item.symbol ? ` · ${item.symbol}` : ""
                  }${extraParts.length ? ` · ${extraParts.join(" · ")}` : ""}`;
                  return (
                    <tr key={`${item.ts}-${item.entry_type}-${item.symbol}`}>
                      <td>{formatTimeWithTZ(item.ts, filters.tz)}</td>
                      <td>{title}</td>
                      <td className="text-right">
                        <button
                          type="button"
                          className="btn secondary"
                          onClick={() => setSelected(item)}
                        >
                          상세
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {totalPages > 1 ? (
          <div className="journal-pagination" id="jr-pagination">
            {currentPage > 1 && (
              <button
                type="button"
                className="pager-btn"
                onClick={() =>
                  onFiltersChange(
                    { page: currentPage - 1 },
                    { triggerRefresh: true }
                  )
                }
                disabled={loading}
              >
                이전
              </button>
            )}
            {pages.map((page, index) =>
              page === "..." ? (
                <span key={`ellipsis-${index}`} className="pager-ellipsis">
                  …
                </span>
              ) : (
                <button
                  key={page}
                  type="button"
                  className={`pager-btn ${page === currentPage ? "pager-btn--active" : ""}`.trim()}
                  onClick={() =>
                    onFiltersChange({ page }, { triggerRefresh: true })
                  }
                  disabled={loading || page === currentPage}
                >
                  {page}
                </button>
              )
            )}
            {currentPage < totalPages && (
              <button
                type="button"
                className="pager-btn"
                onClick={() =>
                  onFiltersChange(
                    { page: currentPage + 1 },
                    { triggerRefresh: true }
                  )
                }
                disabled={loading}
              >
                다음
              </button>
            )}
          </div>
        ) : null}

        <Modal
          open={!!selected}
          title="저널 상세"
          onClose={() => setSelected(null)}
        >
          {selected ? (
            <JournalModalContent item={selected} tz={filters.tz} />
          ) : null}
        </Modal>
      </div>
    </section>
  );
}

