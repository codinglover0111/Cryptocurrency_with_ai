-- Schema for Supabase to mirror the TradeStore tables used by the bot.
-- Execute this in the SQL editor or via `supabase db push` before running the app.

create table if not exists public.trades (
    id bigserial primary key,
    ts timestamptz default timezone('utc', now()),
    symbol text,
    side text,
    type text,
    price double precision,
    quantity double precision,
    tp double precision,
    sl double precision,
    leverage double precision,
    status text,
    order_id text,
    pnl double precision
);

create index if not exists trades_symbol_ts_idx on public.trades (symbol, ts desc);
create index if not exists trades_order_idx on public.trades (order_id);

create table if not exists public.journals (
    id bigserial primary key,
    ts timestamptz default timezone('utc', now()),
    symbol text,
    entry_type text,
    content text,
    reason text,
    meta jsonb,
    ref_order_id text
);

create index if not exists journals_symbol_ts_idx on public.journals (symbol, ts desc);
create index if not exists journals_type_idx on public.journals (entry_type);
