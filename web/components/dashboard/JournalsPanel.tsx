"use client";

import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { Label } from "@/components/common/Label";
import { Modal } from "@/components/common/Modal";
import { Select } from "@/components/common/Select";
import { extractDecisionInfo, formatNumberBrief, formatTimeWithTZ } from "@/lib/format";
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
    <div className="space-y-2">
      <div>
        <strong>시간</strong>: {formatTimeWithTZ(item.ts, tz)}
      </div>
      <div>
        <strong>유형</strong>: {item.entry_type}
      </div>
      <div>
        <strong>심볼</strong>: {item.symbol ?? "-"}
      </div>
      {item.reason ? (
        <div>
          <strong>사유</strong>: {item.reason}
        </div>
      ) : null}
      {decision.status ? (
        <div>
          <strong>포지션</strong>: {decision.status.toUpperCase()}
        </div>
      ) : null}
      {(entry || tp || sl) && (
        <div className="text-sm text-slate-600">
          <strong>트레이드</strong>: {entry ? `진입 ${entry}` : ""}
          {tp ? ` · TP ${tp}` : ""}
          {sl ? ` · SL ${sl}` : ""}
        </div>
      )}
      <div>
        <strong>내용</strong>
      </div>
      <pre className="whitespace-pre-wrap rounded-lg bg-slate-100 p-3 text-xs text-slate-700">
        {item.content ?? "-"}
      </pre>
      <div>
        <strong>메타</strong>
      </div>
      <pre className="whitespace-pre-wrap rounded-lg bg-slate-100 p-3 text-xs text-slate-700">
        {item.meta ? JSON.stringify(item.meta, null, 2) : "{}"}
      </pre>
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

  const activeTypes = Object.entries(filters.types)
    .filter(([, enabled]) => enabled)
    .map(([key]) => key);

  const symbolParamSegment = filters.symbol ? `&symbol=${encodeURIComponent(filters.symbol)}` : "";
  const typeParamSegment = activeTypes.length
    ? `&types=${encodeURIComponent(activeTypes.join(","))}`
    : "";

  const overlayUrl = `/overlay?limit=${filters.limit}&today_only=${filters.range === "today" ? 1 : 0}&ascending=${filters.sort === "asc" ? 1 : 0}${symbolParamSegment}${typeParamSegment}`;
  const positionsOverlayUrl = `/overlay_positions?refresh=5&fs=18`;

  return (
    <Card className="flex flex-col gap-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">저널 조회</h2>
          <p className="text-sm text-slate-600">
            오늘 생성된 최신 기록을 시간순으로 확인하세요.
          </p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Label>
          페이지당 개수
          <Select
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
          </Select>
        </Label>
        <Label>
          타임존
          <Select
            value={filters.tz}
            onChange={(event) =>
              onFiltersChange({ tz: event.target.value }, { triggerRefresh: true })
            }
          >
            {timezoneOptions.map((tz) => (
              <option key={tz} value={tz}>
                {tz}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          코인
          <Select
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
          </Select>
        </Label>
        <Label>
          정렬
          <Select
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
          </Select>
        </Label>
        <Label>
          기간
          <Select
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
          </Select>
        </Label>
      </div>

      <div className="flex flex-wrap gap-4">
        <div>
          <span className="mb-2 block text-xs font-semibold text-slate-500">유형</span>
          <div className="flex flex-wrap gap-3 text-sm text-slate-600">
            {DEFAULT_TYPES.map((type) => (
              <label key={type} className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  checked={!!filters.types[type]}
                  onChange={() => handleTypeToggle(type)}
                />
                {type}
              </label>
            ))}
          </div>
        </div>
        <div>
          <span className="mb-2 block text-xs font-semibold text-slate-500">포지션</span>
          <div className="flex flex-wrap gap-3 text-sm text-slate-600">
            {DEFAULT_STATUSES.map((status) => (
              <label key={status} className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  checked={!!filters.statuses[status]}
                  onChange={() => handleStatusToggle(status)}
                />
                {status}
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button variant="ghost" size="sm" onClick={onRefresh} disabled={loading}>
          조회
        </Button>
        <Button variant="ghost" size="sm" onClick={onReset} disabled={loading}>
          초기화
        </Button>
        <Button asChild variant="primary" size="sm">
          <a href={overlayUrl} target="_blank" rel="noreferrer">
            판단 오버레이
          </a>
        </Button>
        <Button asChild variant="primary" size="sm">
          <a href={positionsOverlayUrl} target="_blank" rel="noreferrer">
            포지션 오버레이
          </a>
        </Button>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-200">
        <div className="max-h-[520px] overflow-auto">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left" style={{ width: "220px" }}>
                  시간
                </th>
                <th className="px-4 py-3 text-left">항목</th>
                <th className="px-4 py-3 text-right" style={{ width: "100px" }} />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {items.length === 0 ? (
                <tr>
                  <td
                    colSpan={3}
                    className="px-4 py-6 text-center text-sm text-slate-500"
                  >
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
                      <td className="px-4 py-3 text-xs text-slate-500">
                        {formatTimeWithTZ(item.ts, filters.tz)}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-700">{title}</td>
                      <td className="px-4 py-3 text-right">
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => setSelected(item)}
                        >
                          상세
                        </Button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {totalPages > 1 ? (
        <div className="flex flex-wrap items-center gap-2">
          {currentPage > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                onFiltersChange({ page: currentPage - 1 }, { triggerRefresh: true })
              }
              disabled={loading}
            >
              이전
            </Button>
          )}
          {pages.map((page, index) =>
            page === "..." ? (
              <span key={`ellipsis-${index}`} className="px-2 text-sm text-slate-400">
                …
              </span>
            ) : (
              <Button
                key={page}
                variant={page === currentPage ? "primary" : "ghost"}
                size="sm"
                onClick={() =>
                  onFiltersChange({ page }, { triggerRefresh: true })
                }
                disabled={loading || page === currentPage}
              >
                {page}
              </Button>
            )
          )}
          {currentPage < totalPages && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                onFiltersChange({ page: currentPage + 1 }, { triggerRefresh: true })
              }
              disabled={loading}
            >
              다음
            </Button>
          )}
        </div>
      ) : null}

      <Modal
        open={!!selected}
        title="저널 상세"
        onClose={() => setSelected(null)}
      >
        {selected ? <JournalModalContent item={selected} tz={filters.tz} /> : null}
      </Modal>
    </Card>
  );
}

