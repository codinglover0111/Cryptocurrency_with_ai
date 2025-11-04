export interface BalanceInfo {
  currency?: string | null;
  total?: number | string | null;
  free?: number | string | null;
  used?: number | string | null;
}

export interface PositionSummaryItem {
  symbol?: string | null;
  side?: string | null;
  entryPrice?: number | string | null;
  size?: number | string | null;
  pnl?: number | string | null;
  pnlPct?: number | string | null;
}

export interface PositionItem {
  symbol?: string | null;
  side?: string | null;
  entryPrice?: number | string | null;
  contracts?: number | string | null;
  amount?: number | string | null;
  size?: number | string | null;
  pnl?: number | string | null;
  pnlPct?: number | string | null;
  lastPrice?: number | string | null;
  markPrice?: number | string | null;
  leverage?: number | string | null;
  [key: string]: unknown;
}

export interface StatusResponse {
  balance?: BalanceInfo | null;
  positionsSummary?: PositionSummaryItem[] | null;
  positions?: PositionItem[] | null;
  [key: string]: unknown;
}

export interface StatsSummary {
  realized_pnl?: number | string | null;
  trades?: number | string | null;
  win_rate?: number | string | null;
  avg_pnl?: number | string | null;
  [key: string]: unknown;
}

export interface StatsBySymbolItem {
  symbol: string;
  trades?: number | string | null;
  realized_pnl?: number | string | null;
  [key: string]: unknown;
}

export interface StatsSeriesItem {
  t: string;
  trades?: number | string | null;
  realized_pnl?: number | string | null;
  [key: string]: unknown;
}

export interface StatsRangeResponse {
  summary?: StatsSummary | null;
  by_symbol?: StatsBySymbolItem[] | null;
  series?: StatsSeriesItem[] | null;
  [key: string]: unknown;
}

export interface SymbolInfo {
  code?: string | null;
  contract?: string | null;
  spot?: string | null;
  [key: string]: unknown;
}

export interface SymbolsResponse {
  symbols?: SymbolInfo[] | null;
  [key: string]: unknown;
}

export interface JournalItem {
  id?: string | number;
  ts: string;
  entry_type: string;
  symbol?: string | null;
  reason?: string | null;
  content?: string | null;
  meta?: Record<string, unknown> | null;
  decision_status?: string | null;
  decision_entry?: number | string | null;
  decision_tp?: number | string | null;
  decision_sl?: number | string | null;
  [key: string]: unknown;
}

export interface JournalsResponse {
  items?: JournalItem[] | null;
  page?: number;
  page_size?: number;
  total?: number;
  [key: string]: unknown;
}

export interface DecisionInfo {
  status: string | null;
  entry: number | null;
  tp: number | null;
  sl: number | null;
}

export interface PositionsOverlayItem {
  symbol: string;
  side?: string | null;
  leverage?: number | string | null;
  entryPrice?: number | null;
  lastPrice?: number | null;
  size?: number | null;
  tp?: number | null;
  sl?: number | null;
  pnl?: number | null;
  pnlPct?: number | null;
  [key: string]: unknown;
}

export interface PositionsOverlayResponse {
  items?: PositionsOverlayItem[] | null;
  [key: string]: unknown;
}
