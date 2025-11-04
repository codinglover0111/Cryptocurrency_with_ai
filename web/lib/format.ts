import type { DecisionInfo, JournalItem } from "@/lib/types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

export function toNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function formatCurrency(
  value: unknown,
  options: {
    currency?: string;
    maximumFractionDigits?: number;
    signed?: boolean;
  } = {}
): string {
  const {
    currency = "USD",
    maximumFractionDigits = 2,
    signed = true,
  } = options;
  const num = toNumber(value);
  if (num === null) return "-";
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits,
  });
  const formatted = formatter.format(num);
  if (!signed) return formatted;
  return num > 0 && !formatted.startsWith("+") ? `+${formatted}` : formatted;
}

export function formatPercent(value: unknown, digits?: number): string {
  const num = toNumber(value);
  if (num === null) return "-";
  const precision =
    digits ?? (Math.abs(num) >= 100 ? 1 : Math.abs(num) >= 10 ? 2 : 2);
  const body = num.toFixed(precision);
  return `${num > 0 ? "+" : ""}${body}%`;
}

export function formatNumberBrief(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  const abs = Math.abs(num);
  const maximumFractionDigits = abs >= 100 ? 2 : abs >= 10 ? 4 : 6;
  return num.toLocaleString("en-US", { maximumFractionDigits });
}

export function formatNumber(value: unknown, fallback = "-"): string {
  const brief = formatNumberBrief(value);
  if (brief !== null) return brief;
  if (value === null || value === undefined) return fallback;
  return String(value);
}

export function normalizeUtcTimestamp(raw: unknown): string | null {
  if (raw == null) return null;
  let value = String(raw).trim();
  if (!value) return value;
  if (!value.includes("T") && value.includes(" ")) {
    value = value.replace(" ", "T");
  }
  if (!/[zZ]|[+-]\d{2}:?\d{2}$/.test(value)) {
    value = `${value}Z`;
  }
  return value;
}

export function formatTimeWithTZ(
  ts: string,
  tz: string,
  options: Intl.DateTimeFormatOptions = {}
): string {
  try {
    const normalized = normalizeUtcTimestamp(ts);
    if (!normalized) return String(ts ?? "");
    const base = new Date(normalized);
    if (Number.isNaN(base.getTime())) return String(ts ?? "");

    const baseOptions: Intl.DateTimeFormatOptions = {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      timeZone: "UTC",
      ...options,
    };

    let displayDate = base;
    let label = "UTC";
    if (tz && tz.startsWith("UTC")) {
      const match = tz.match(/^UTC([+-]\d{1,2})$/);
      if (match) {
        const offsetH = parseInt(match[1], 10);
        displayDate = new Date(base.getTime() + offsetH * 3600000);
        label = tz;
      }
    }

    const formatted = displayDate.toLocaleString("ko-KR", baseOptions);
    return `${formatted} ${label}`;
  } catch {
    return String(ts ?? "");
  }
}

export function maybeParseJSON<T = unknown>(value: unknown): T | null {
  if (typeof value !== "string") return null;
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

export function normalizeDecisionStatus(value: unknown): string | null {
  if (value == null) return null;
  const raw = String(value).trim().toLowerCase();
  if (!raw) return null;
  if (["long", "buy"].includes(raw)) return "long";
  if (["short", "sell"].includes(raw)) return "short";
  if (["hold", "watch"].includes(raw)) return "hold";
  if (["stop"].includes(raw)) return "stop";
  if (["skip"].includes(raw)) return "skip";
  return raw;
}

export function firstAvailableNumber(values: unknown[]): number | null {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const num = Number(value);
    if (Number.isFinite(num)) return num;
  }
  return null;
}

export function extractDecisionInfo(item: JournalItem): DecisionInfo {
  const info: DecisionInfo = {
    status: null,
    entry: null,
    tp: null,
    sl: null,
  };
  if (!item) return info;

  const meta = isRecord(item.meta) ? item.meta : {};
  const decisionMeta = isRecord(meta["decision"])
    ? (meta["decision"] as Record<string, unknown>)
    : null;
  const contentObj = maybeParseJSON<Record<string, unknown>>(
    item.content ?? undefined
  );

  const statusCandidates = [
    item.decision_status,
    meta["status"],
    meta["side"],
    decisionMeta?.["status"],
    decisionMeta?.["Status"],
    decisionMeta?.["ai_status"],
    contentObj?.["status"],
  ];

  for (const candidate of statusCandidates) {
    const normalized = normalizeDecisionStatus(candidate);
    if (normalized) {
      info.status = normalized;
      break;
    }
  }

  const entryCandidates = [
    item.decision_entry,
    meta["entry_price"],
    meta["entry"],
    decisionMeta?.["entry"],
    decisionMeta?.["entry_price"],
    decisionMeta?.["price"],
    contentObj?.["entry"],
    contentObj?.["entry_price"],
    contentObj?.["price"],
  ];

  info.entry = firstAvailableNumber(entryCandidates);

  const tpCandidates = [
    item.decision_tp,
    meta["tp"],
    decisionMeta?.["tp"],
    contentObj?.["tp"],
  ];
  info.tp = firstAvailableNumber(tpCandidates);

  const slCandidates = [
    item.decision_sl,
    meta["sl"],
    decisionMeta?.["sl"],
    contentObj?.["sl"],
  ];
  info.sl = firstAvailableNumber(slCandidates);

  return info;
}

export function escapeHtml(value: unknown): string {
  const str = String(value ?? "");
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
