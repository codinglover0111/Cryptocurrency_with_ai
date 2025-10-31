import { Suspense } from "react";
import { loadJournals, loadStats, loadStatus } from "./actions";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { compactCount, formatCurrency, formatNumber } from "@/lib/utils";

async function DashboardContent() {
  const [statusResult, statsResult, journalsResult] = await Promise.allSettled([
    loadStatus(),
    loadStats(),
    loadJournals({ limit: 10 })
  ]);

  const status = statusResult.status === "fulfilled" ? statusResult.value : null;
  const stats = statsResult.status === "fulfilled" ? statsResult.value : null;
  const journals = journalsResult.status === "fulfilled" ? journalsResult.value : null;

  const balance =
    (status as any)?.balance ??
    (status as any)?.account ??
    (status as any)?.data ??
    {};

  const positionsSummary = Array.isArray((status as any)?.positionsSummary)
    ? ((status as any)?.positionsSummary as Array<Record<string, unknown>>)
    : [];

  const journalItems = Array.isArray((journals as any)?.items)
    ? ((journals as any)?.items as Array<Record<string, unknown>>)
    : [];

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle>Total Equity</CardTitle>
            <CardDescription>Aggregated account balance</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-3xl font-semibold">
              {formatCurrency((balance as any)?.totalEquity ?? (balance as any)?.walletBalance ?? null)}
            </p>
            <p className="text-sm text-muted-foreground">
              Wallet balance: {formatCurrency((balance as any)?.walletBalance ?? null)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
            <CardDescription>Per-symbol exposure overview</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-3xl font-semibold">{positionsSummary.length}</p>
            <div className="space-y-1">
              {positionsSummary.slice(0, 3).map((item, index) => {
                const entry = item as Record<string, unknown>;
                const rawSymbol = entry.symbol;
                const symbol = typeof rawSymbol === "string" && rawSymbol.trim() ? rawSymbol : `#${index + 1}`;
                const pnlValue = Number(entry.pnl);
                return (
                  <div key={`${symbol}-${index}`} className="flex items-center justify-between text-sm">
                    <span className="font-medium">{symbol}</span>
                    <span>{formatCurrency(Number.isFinite(pnlValue) ? pnlValue : null)}</span>
                  </div>
                );
              })}
              {positionsSummary.length > 3 ? (
                <p className="text-xs text-muted-foreground">+{positionsSummary.length - 3} more symbols</p>
              ) : null}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Realized PnL</CardTitle>
            <CardDescription>Lifetime realized profits</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold">
              {formatCurrency((stats as any)?.realized_pnl ?? null)}
            </p>
            <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
              <span>Win rate</span>
              <Badge variant="success">
                {formatNumber(((stats as any)?.win_rate ?? 0) * 100, 1)}%
              </Badge>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Trades</CardTitle>
            <CardDescription>Closed position summary</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-3xl font-semibold">
              {compactCount((stats as any)?.total_trades ?? null)}
            </p>
            <p className="text-sm text-muted-foreground">
              Avg. PnL: {formatCurrency((stats as any)?.avg_pnl ?? null)}
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card className="h-full">
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle>Recent Journals</CardTitle>
              <CardDescription>Latest model thoughts & actions</CardDescription>
            </div>
            <Button asChild variant="secondary" size="sm">
              <a href="/api/journals" rel="noreferrer" target="_blank">
                Open raw feed
              </a>
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            {journalItems.length === 0 ? (
              <p className="text-sm text-muted-foreground">No journal entries recorded for the selected window.</p>
            ) : (
              journalItems.slice(0, 6).map((item, idx) => {
                const entry = item as Record<string, unknown>;
                const rawTs = typeof entry.ts === "string" ? entry.ts : null;
                const ts = rawTs ? new Date(rawTs) : null;
                const typeLabel = typeof entry.entry_type === "string" ? entry.entry_type : "unknown";
                const content = typeof entry.content === "string" ? entry.content : "(no content)";
                const symbol = typeof entry.symbol === "string" ? entry.symbol : null;
                return (
                  <article key={`${rawTs ?? idx}`} className="space-y-2 rounded-lg border border-border bg-background/60 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                      <span className="font-medium uppercase tracking-wide text-foreground">{typeLabel}</span>
                      <span>{ts ? ts.toLocaleString() : ""}</span>
                    </div>
                    <p className="text-sm leading-relaxed text-foreground/90">{content}</p>
                    {symbol ? (
                      <Badge variant="outline" className="text-xs uppercase">{symbol}</Badge>
                    ) : null}
                  </article>
                );
              })
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API Diagnostics</CardTitle>
            <CardDescription>Connectivity to trading backend</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <div className="flex items-center justify-between">
              <span>Status API</span>
              <Badge variant={status ? "success" : "destructive"}>
                {status ? "healthy" : "offline"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span>Statistics API</span>
              <Badge variant={stats ? "success" : "destructive"}>
                {stats ? "healthy" : "offline"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span>Journal API</span>
              <Badge variant={journals ? "success" : "destructive"}>
                {journals ? "healthy" : "offline"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">AI Trading Control Center</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Monitor live account metrics, review the agent journal, and inspect backend health.
        </p>
      </div>
      <Suspense fallback={<p className="text-sm text-muted-foreground">Loading dashboard…</p>}>
        {/* @ts-expect-error Async Server Component */}
        <DashboardContent />
      </Suspense>
    </main>
  );
}
