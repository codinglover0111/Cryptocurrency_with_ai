"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { BalancePanel } from "@/components/dashboard/BalancePanel";
import { JournalsPanel, type JournalFilters, JOURNAL_PAGE_SIZES } from "@/components/dashboard/JournalsPanel";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { StatsPanel, type StatsFilters } from "@/components/dashboard/StatsPanel";
import { fetchJSON } from "@/lib/api";
import type {
  JournalsResponse,
  StatsRangeResponse,
  StatusResponse,
  SymbolInfo,
  SymbolsResponse,
} from "@/lib/types";

const BALANCE_REFRESH_COOLDOWN_MS = 5000;

const INITIAL_STATS_FILTERS: StatsFilters = {
  since: "",
  until: "",
  symbol: "",
  group: "",
};

function createInitialJournalFilters(): JournalFilters {
  return {
    tz: "UTC+9",
    range: "recent15",
    sort: "desc",
    limit: 30,
    page: 1,
    symbol: "",
    types: {
      decision: true,
      action: true,
      review: true,
      error: true,
    },
    statuses: {
      long: true,
      hold: true,
      short: true,
    },
  };
}

export function DashboardPage() {
  const [statusData, setStatusData] = useState<StatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState<boolean>(true);
  const [statusHint, setStatusHint] = useState<string | null>(null);
  const [lastManualStatusTime, setLastManualStatusTime] = useState<number>(0);

  const [statsFilters, setStatsFilters] = useState<StatsFilters>(INITIAL_STATS_FILTERS);
  const [statsData, setStatsData] = useState<StatsRangeResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState<boolean>(false);

  const [journalFilters, setJournalFilters] = useState<JournalFilters>(createInitialJournalFilters);
  const [journalData, setJournalData] = useState<JournalsResponse | null>(null);
  const [journalLoading, setJournalLoading] = useState<boolean>(false);

  const [symbols, setSymbols] = useState<SymbolInfo[]>([]);

  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => window.clearInterval(id);
  }, []);

  const cooldownSeconds = useMemo(() => {
    const elapsed = now - lastManualStatusTime;
    const remaining = BALANCE_REFRESH_COOLDOWN_MS - elapsed;
    return remaining > 0 ? Math.ceil(remaining / 1000) : 0;
  }, [lastManualStatusTime, now]);

  const buildStatsRangeUrl = useCallback(() => {
    const params = new URLSearchParams();
    if (statsFilters.since) params.set("since", statsFilters.since);
    if (statsFilters.until) params.set("until", statsFilters.until);
    if (statsFilters.symbol) params.set("symbol", statsFilters.symbol);
    if (statsFilters.group) params.set("group", statsFilters.group);
    const query = params.toString();
    return `/stats_range${query ? `?${query}` : ""}`;
  }, [statsFilters]);

  const refreshStatus = useCallback(
    async (manual = false) => {
      if (manual) {
        const nowTs = Date.now();
        const elapsed = nowTs - lastManualStatusTime;
        const withinCooldown = elapsed < BALANCE_REFRESH_COOLDOWN_MS;
        if (withinCooldown) {
          const waitMs = BALANCE_REFRESH_COOLDOWN_MS - elapsed;
          const waitSec = Math.max(1, Math.ceil(waitMs / 1000));
          setStatusHint(`최근에 새로고침했습니다 · ${waitSec}초 후 새로운 데이터 적용`);
        } else {
          setStatusHint("새로고침 중...");
        }
      }

      setStatusLoading(true);
      try {
        const result = await fetchJSON<StatusResponse>("/status");
        setStatusData(result);
        if (manual) {
          const nowTs = Date.now();
          const elapsed = nowTs - lastManualStatusTime;
          const withinCooldown = elapsed < BALANCE_REFRESH_COOLDOWN_MS;
          if (!withinCooldown) {
            setStatusHint("업데이트 완료");
            setTimeout(() => setStatusHint(null), 2000);
            setLastManualStatusTime(nowTs);
          } else {
            setTimeout(() => setStatusHint(null), 2000);
          }
        }
      } catch (error) {
        console.error("/status", error);
        if (manual) {
          setStatusHint("새로고침 실패");
          setTimeout(() => setStatusHint(null), 3000);
        }
      } finally {
        setStatusLoading(false);
      }
    },
    [lastManualStatusTime]
  );

  const refreshStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      try {
        await fetchJSON("/stats");
      } catch (error) {
        console.warn("/stats 호출 실패", error);
      }
      const url = buildStatsRangeUrl();
      const result = await fetchJSON<StatsRangeResponse>(url);
      setStatsData(result);
    } catch (error) {
      console.error("/stats_range", error);
    } finally {
      setStatsLoading(false);
    }
  }, [buildStatsRangeUrl]);

  const doFetchJournals = useCallback(
    async (filtersToUse: JournalFilters) => {
      const selectedTypes = Object.entries(filtersToUse.types)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key);
      if (selectedTypes.length === 0) {
        setJournalData({ items: [], total: 0, page: 1, page_size: filtersToUse.limit });
        return;
      }

      const selectedStatuses = Object.entries(filtersToUse.statuses)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key);
      if (selectedStatuses.length === 0) {
        setJournalData({ items: [], total: 0, page: 1, page_size: filtersToUse.limit });
        return;
      }

      setJournalLoading(true);
      try {
        const params = new URLSearchParams({
          limit: String(filtersToUse.limit),
          page: String(filtersToUse.page),
          today_only: filtersToUse.range === "today" ? "1" : "0",
          ascending: filtersToUse.sort === "asc" ? "1" : "0",
        });
        if (filtersToUse.range === "recent15") {
          params.set("recent_minutes", "15");
        }
        if (selectedTypes.length > 0) {
          params.set("types", selectedTypes.join(","));
        }
        if (filtersToUse.symbol) {
          params.set("symbol", filtersToUse.symbol);
        }
        if (selectedStatuses.length > 0 && selectedStatuses.length < Object.keys(filtersToUse.statuses).length) {
          params.set("decision_statuses", selectedStatuses.join(","));
        }

        const result = await fetchJSON<JournalsResponse>(
          `/api/journals_filtered?${params.toString()}`
        );

        setJournalData(result);

        if (typeof result.page === "number" && result.page > 0 && result.page !== filtersToUse.page) {
          setJournalFilters((prev) => ({ ...prev, page: result.page ?? prev.page }));
        }
        if (
          typeof result.page_size === "number" &&
          JOURNAL_PAGE_SIZES.includes(result.page_size) &&
          result.page_size !== filtersToUse.limit
        ) {
          setJournalFilters((prev) => ({ ...prev, limit: result.page_size ?? prev.limit }));
        }
      } catch (error) {
        console.error("/api/journals_filtered", error);
      } finally {
        setJournalLoading(false);
      }
    },
    []
  );

  const refreshJournals = useCallback(async () => {
    await doFetchJournals(journalFilters);
  }, [doFetchJournals, journalFilters]);

  const refreshAll = useCallback(async () => {
    setStatusLoading(true);
    setStatsLoading(true);
    try {
      const [status, stats, syms] = await Promise.all([
        fetchJSON<StatusResponse>("/status"),
        fetchJSON<StatsRangeResponse>(buildStatsRangeUrl()),
        fetchJSON<SymbolsResponse>("/symbols"),
      ]);
      setStatusData(status);
      setStatsData(stats);
      setSymbols(syms.symbols ?? []);
    } catch (error) {
      console.error("refreshAll", error);
    } finally {
      setStatusLoading(false);
      setStatsLoading(false);
    }
  }, [buildStatsRangeUrl]);

  useEffect(() => {
    void refreshAll();
    void refreshJournals();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshAll();
    }, 10000);
    return () => window.clearInterval(id);
  }, [refreshAll]);

  const updateStatsFilters = useCallback((patch: Partial<StatsFilters>) => {
    setStatsFilters((prev) => ({ ...prev, ...patch }));
  }, []);

  const updateJournalFilters = useCallback(
    (
      patch: Partial<Omit<JournalFilters, "types" | "statuses">> & {
        types?: Partial<Record<string, boolean>>;
        statuses?: Partial<Record<string, boolean>>;
      },
      options?: { triggerRefresh?: boolean }
    ) => {
      setJournalFilters((prev) => {
        const { types: typesPatch, statuses: statusesPatch, ...rest } = patch;
        const next: JournalFilters = {
          ...prev,
          ...rest,
          types: typesPatch ? { ...prev.types, ...typesPatch } : prev.types,
          statuses: statusesPatch
            ? { ...prev.statuses, ...statusesPatch }
            : prev.statuses,
        };
        if (options?.triggerRefresh) {
          void doFetchJournals(next);
        }
        return next;
      });
    },
    [doFetchJournals]
  );

  const resetJournalFilters = useCallback(() => {
    const initial = createInitialJournalFilters();
    setJournalFilters(initial);
    void doFetchJournals(initial);
  }, [doFetchJournals]);

  return (
    <div className="page-shell">
      <PageHeader
        title="Crypto Bot Dashboard"
        subtitle="상태와 통계를 한눈에 확인하고 최근 저널을 빠르게 탐색하세요."
        badgeText="조회 전용"
      />

      <main className="page-main">
        <section className="panel-grid">
          <BalancePanel
            status={statusData}
            loading={statusLoading}
            onRefresh={() => refreshStatus(true)}
            hintMessage={statusHint}
            cooldownSeconds={cooldownSeconds}
          />
          <StatsPanel
            filters={statsFilters}
            onFiltersChange={updateStatsFilters}
            onRefresh={refreshStats}
            data={statsData}
            loading={statsLoading}
            symbols={symbols}
          />
        </section>

        <JournalsPanel
          filters={journalFilters}
          onFiltersChange={updateJournalFilters}
          onRefresh={refreshJournals}
          onReset={resetJournalFilters}
          data={journalData}
          loading={journalLoading}
          symbols={symbols}
        />
      </main>
    </div>
  );
}

