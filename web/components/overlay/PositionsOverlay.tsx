"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { formatNumberBrief } from "@/lib/format";
import type { PositionsOverlayItem, PositionsOverlayResponse } from "@/lib/types";

async function fetchPositions(query: string): Promise<PositionsOverlayItem[]> {
  const res = await fetch(`/api/positions_summary?${query}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`/api/positions_summary ${res.status}`);
  }
  const data = (await res.json()) as PositionsOverlayResponse;
  return data.items ?? [];
}

function formatDecimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

export function PositionsOverlay() {
  const searchParams = useSearchParams();
  const symbol = searchParams.get("symbol") ?? "";
  const refreshSeconds = Math.max(0, Number(searchParams.get("refresh") ?? 5));
  const baseFontSize = Number(searchParams.get("fs") ?? 0);

  const [forceMark, setForceMark] = useState<boolean>(false);
  const [forceExchangePnl, setForceExchangePnl] = useState<boolean>(false);
  const [forceRoe, setForceRoe] = useState<boolean>(false);
  const [items, setItems] = useState<PositionsOverlayItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (baseFontSize > 0) {
      const original = document.body.style.fontSize;
      document.body.style.fontSize = `${baseFontSize}px`;
      return () => {
        document.body.style.fontSize = original;
      };
    }
    return undefined;
  }, [baseFontSize]);

  const queryString = useMemo(() => {
    const params = new URLSearchParams();
    if (symbol) params.set("symbol", symbol);
    params.set("force_mark", forceMark ? "1" : "0");
    params.set("force_exchange_pnl", forceExchangePnl ? "1" : "0");
    params.set("force_roe", forceRoe ? "1" : "0");
    return params.toString();
  }, [forceExchangePnl, forceMark, forceRoe, symbol]);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchPositions(queryString);
      setItems(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [queryString]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (refreshSeconds <= 0) return undefined;
    const id = window.setInterval(() => {
      void refresh();
    }, refreshSeconds * 1000);
    return () => window.clearInterval(id);
  }, [refresh, refreshSeconds]);

  const pnlClass = (pnl?: number | null) =>
    pnl !== null && pnl !== undefined && Number.isFinite(pnl)
      ? pnl > 0
        ? "text-emerald-400"
        : pnl < 0
        ? "text-rose-400"
        : "text-slate-200"
      : "text-slate-200";

  return (
    <div className="min-h-screen bg-black/95 px-6 py-6 text-slate-100">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-slate-600 text-blue-500 focus:ring-blue-500"
              checked={forceMark}
              onChange={(event) => setForceMark(event.target.checked)}
            />
            마크가격 고정
          </label>
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-slate-600 text-blue-500 focus:ring-blue-500"
              checked={forceExchangePnl}
              onChange={(event) => setForceExchangePnl(event.target.checked)}
            />
            거래소 PnL만 사용
          </label>
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-slate-600 text-blue-500 focus:ring-blue-500"
              checked={forceRoe}
              onChange={(event) => setForceRoe(event.target.checked)}
            />
            ROE 퍼센트 강제
          </label>
          <button
            type="button"
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-slate-700"
            onClick={() => void refresh()}
          >
            적용
          </button>
        </div>

        {error ? (
          <div className="rounded-xl border border-rose-700 bg-rose-800/40 px-4 py-3 text-sm text-rose-100">
            데이터를 불러오지 못했습니다: {error}
          </div>
        ) : null}

        {loading && items.length === 0 ? (
          <div className="rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-8 text-center text-sm text-slate-300">
            불러오는 중...
          </div>
        ) : null}

        {!loading && items.length === 0 ? (
          <div className="rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-8 text-center text-sm text-slate-300">
            포지션이 없습니다.
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2">
          {items.map((item) => {
            const pnlValue = item.pnl ?? null;
            const pnlPercent = item.pnlPct ?? null;
            return (
              <div
                key={`${item.symbol}-${item.entryPrice}-${item.size}`}
                className="rounded-2xl border border-slate-700 bg-slate-800/70 px-6 py-5 shadow-lg"
              >
                <div className="text-lg font-semibold text-slate-50">
                  {item.symbol} · {(item.side ?? "").toString().toUpperCase()}
                  {item.leverage ? ` · ${item.leverage}x` : ""}
                </div>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-slate-400">진입가</span>
                  <span className="font-semibold text-slate-100">
                    {formatDecimal(item.entryPrice)}
                  </span>
                </div>
                <div className="mt-2 flex justify-between text-sm">
                  <span className="text-slate-400">현재가</span>
                  <span className="font-semibold text-slate-100">
                    {formatDecimal(item.lastPrice)}
                  </span>
                </div>
                <div className="mt-2 flex justify-between text-sm">
                  <span className="text-slate-400">수량</span>
                  <span className="font-semibold text-slate-100">
                    {formatDecimal(item.size, 3)}
                  </span>
                </div>
                <div className="mt-2 flex justify-between text-sm">
                  <span className="text-slate-400">익절가</span>
                  <span className="font-semibold text-slate-100">
                    {formatDecimal(item.tp)}
                  </span>
                </div>
                <div className="mt-2 flex justify-between text-sm">
                  <span className="text-slate-400">손절가</span>
                  <span className="font-semibold text-slate-100">
                    {formatDecimal(item.sl)}
                  </span>
                </div>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-slate-400">손익</span>
                  <span className={`font-semibold ${pnlClass(pnlValue ?? null)}`}>
                    {formatNumberBrief(pnlValue) ?? "-"}
                    {pnlPercent !== null && pnlPercent !== undefined
                      ? ` (${formatNumberBrief(pnlPercent) ?? "-"}%)`
                      : ""}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

