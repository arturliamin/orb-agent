"""
Opening Range Breakout (ORB) Strategy Backtester - v2 (with signal ranking)
============================================================================

ENTRY (first hour only, 9:30-10:30 ET):
  - Opening range = high/low of first 15 minutes (9:30-9:45 ET)
  - Direction filter: EMA9 > EMA20 for long, EMA9 < EMA20 for short
  - Trigger: price breaks range high (long) or range low (short)
  - Confirmation: volume >= 2x the average for that time-of-day 1-min bucket
  - VWAP filter: long only above VWAP, short only below VWAP

SIGNAL RANKING (new in v2) - every qualified breakout gets a 0-100 score:
  Volume multiple at breakout ......... 30  (capped at 5x)
  Stop tightness ...................... 25  (tighter than the 2% cap = higher)
  Relative strength vs SPY ............ 20  (move since open minus SPY's, in trade direction)
  Trending rank ....................... 15  (rank on that morning's top-100 list)
  Catalyst present .................... 10  (binary; no feed wired in yet -> 0)
  Only the TOP-scoring signal per day is taken, and only if score >= MIN_SCORE.

EXIT:
  - Initial stop = min(structural stop [opposite side of range], 2% loss)
  - Trailing stop arms at +5% gain, exits on a 2-percentage-point giveback from peak
  - EOD exit 10 minutes before close

SIZING:
  - Position size ($) = $50 / stop_distance_pct, capped at 30% of equity

COMPLIANCE:
  - Max 3 trades per rolling 5 business days (portfolio-wide)

UNIVERSE:
  - If trending_log.csv exists (from log_trending.py), each day's candidates
    are that day's logged top-100. Otherwise falls back to WATCHLIST.

USAGE:
  pip install alpaca-py pandas numpy --break-system-packages
  export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
  python orb_backtest.py
  -> trade_log.csv, signal_log.csv (every qualified signal with its score)
"""

import os
import re
import math
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "YOUR_KEY_ID_HERE")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
DATA_FEED = "iex"  # "iex" = free tier, "sip" = paid consolidated tape

TRENDING_LOG = os.environ.get("TRENDING_LOG_PATH", "trending_log.csv")
WATCHLIST = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]   # fallback only
BENCHMARK = "SPY"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/user-data/outputs")

ACCOUNT_EQUITY = 10_000.0
RISK_PER_TRADE = 50.0
MAX_POSITION_PCT = 0.30
STOP_CAP_PCT = 0.02
TRAIL_ARM_PCT = 0.05
TRAIL_GIVEBACK_PCT = 0.02
VOLUME_SPIKE_MULT = 2.0
MAX_TRADES_PER_5_DAYS = 3

# Ranking weights (sum to 100)
W_VOLUME, W_STOP, W_RS, W_TREND, W_CATALYST = 30, 25, 20, 15, 10
VOLUME_CAP_MULT = 5.0      # volume multiple saturates at 5x
MIN_SCORE = 60.0           # signals below this are not alerted / not traded

TZ = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
RANGE_END = dtime(9, 45)
ENTRY_CUTOFF = dtime(10, 30)
MARKET_CLOSE = dtime(16, 0)
EOD_WARN_MINUTES = 10

END_DATE = datetime.now(TZ).date()
START_DATE = END_DATE - timedelta(days=60)
TEST_TRADING_DAYS = 20
VOLUME_LOOKBACK_DAYS = 10

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def fetch_bars(symbols) -> pd.DataFrame:
    """1-min bars for a list of symbols, batched to respect rate limits."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    frames = []
    symbols = sorted(set(symbols))
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i + 50]
        print(f"  fetching bars {i + 1}-{i + len(batch)} of {len(symbols)}")
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(START_DATE, dtime(0, 0)),
            end=datetime.combine(END_DATE, dtime(23, 59)),
            feed=DataFeed.IEX if DATA_FEED == "iex" else DataFeed.SIP,
        )
        df = client.get_stock_bars(req).df
        if not df.empty:
            frames.append(df.reset_index())
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_convert(TZ)
    return bars


ETF_NAME_RE = re.compile(r"\b(ETF|ETN|Trust|Fund|Index|Shares|iShares|ProShares|SPDR|Invesco|Direxion|"
                         r"Vanguard|VanEck|WisdomTree|Schwab|Global X|Ultra|Bull|Bear|2x|3x)\b", re.I)
ETF_EXCHANGES = {"ARCA", "BATS"}
ETF_FALLBACK = {"SPY","QQQ","IWM","DIA","TQQQ","SQQQ","SOXL","SOXS","SPXL","SPXS","UVXY","VXX","XLF","XLE","XLK",
                "ARKK","GLD","SLV","TLT","HYG","LQD","EEM","EFA","VTI","VOO","IVV","XLV","XLI","XLY","XLP","XBI",
                "SMH","KRE","GDX","USO","UNG","TNA","TZA","LABU","LABD","NVDL","TSLL","MSTU","MSTZ","BITO","IBIT"}


def etf_symbols():
    """ETF/ETN/fund symbols to exclude from the candidate universe (SPY is still fetched as benchmark)."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass
        tc = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        out = set()
        for a in tc.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)):
            ex = str(a.exchange.value if hasattr(a.exchange, "value") else a.exchange)
            if ex in ETF_EXCHANGES or ETF_NAME_RE.search(a.name or ""):
                out.add(a.symbol)
        return out or ETF_FALLBACK
    except Exception as e:  # noqa: BLE001
        print(f"ETF lookup failed ({e}); using fallback list")
        return ETF_FALLBACK


def load_trending_log():
    """{date: {symbol: rank}} from log_trending.py output, or None."""
    if not os.path.exists(TRENDING_LOG):
        return None
    log = pd.read_csv(TRENDING_LOG, parse_dates=["date"])
    log["date"] = log["date"].dt.date
    out = {}
    for d, g in log.groupby("date"):
        out[d] = dict(zip(g["symbol"], g["rank"]))
    return out


# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def regular_hours(day_df):
    t = day_df["timestamp"].dt.time
    return day_df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].reset_index(drop=True)


def add_indicators(day_df):
    df = day_df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    return df


def volume_baseline(days, current_date, minute_of_day):
    vols = []
    prior = sorted(d for d in days if d < current_date)[-VOLUME_LOOKBACK_DAYS:]
    for d in prior:
        m = days[d][days[d]["timestamp"].dt.time == minute_of_day]
        if not m.empty:
            vols.append(m["volume"].iloc[0])
    return sum(vols) / len(vols) if vols else float("inf")


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_signal(vol_mult, stop_dist_pct, rel_strength, trend_rank, catalyst):
    """Composite 0-100 score. Each factor normalized to 0-1 then weighted."""
    f_vol = min(vol_mult, VOLUME_CAP_MULT) / VOLUME_CAP_MULT
    f_stop = 1.0 - min(stop_dist_pct, STOP_CAP_PCT) / STOP_CAP_PCT
    f_rs = min(max(rel_strength, 0.0), 0.02) / 0.02          # +2% vs SPY saturates
    f_trend = (1.0 - (trend_rank - 1) / 99.0) if trend_rank else 0.5   # neutral if unknown
    f_cat = 1.0 if catalyst else 0.0
    score = (W_VOLUME * f_vol + W_STOP * f_stop + W_RS * f_rs
             + W_TREND * f_trend + W_CATALYST * f_cat)
    parts = dict(vol=f_vol, stop=f_stop, rs=f_rs, trend=f_trend, cat=f_cat)
    return round(score, 1), parts


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

class Trade:
    def __init__(self, sig):
        self.__dict__.update(sig)
        self.exit_time = self.exit_price = self.exit_reason = None
        self.peak_gain_pct = 0.0
        self.trail_armed = False

    def _diff(self, px):
        return (px - self.entry_price) if self.direction == "long" else (self.entry_price - px)

    def pnl_dollars(self):
        return self._diff(self.exit_price) * self.shares if self.exit_price else 0.0

    def pnl_pct(self):
        return self._diff(self.exit_price) / self.entry_price if self.exit_price else 0.0


def size_position(entry_price, structural_dist_pct):
    stop_dist = min(structural_dist_pct, STOP_CAP_PCT)
    value = min(RISK_PER_TRADE / stop_dist, ACCOUNT_EQUITY * MAX_POSITION_PCT)
    shares = math.floor(value / entry_price)
    return shares, shares * entry_price, stop_dist


def find_signals_for_day(symbol, date, days, spy_day, trend_rank):
    """Return the FIRST qualified breakout for this symbol on this date, scored."""
    day_df = add_indicators(days[date])
    t = day_df["timestamp"].dt.time
    opening = day_df[(t >= MARKET_OPEN) & (t < RANGE_END)]
    if opening.empty:
        return None
    range_high, range_low = opening["high"].max(), opening["low"].min()
    day_open = opening["open"].iloc[0]

    spy_open = spy_day["open"].iloc[0] if spy_day is not None and not spy_day.empty else None

    window = day_df[(t >= RANGE_END) & (t < ENTRY_CUTOFF)]
    for _, row in window.iterrows():
        base = volume_baseline(days, date, row["timestamp"].time())
        vol_mult = row["volume"] / base if base and base != float("inf") else 0.0
        if vol_mult < VOLUME_SPIKE_MULT:
            continue
        bull = row["ema9"] > row["ema20"] and row["close"] > row["vwap"]
        bear = row["ema9"] < row["ema20"] and row["close"] < row["vwap"]
        if row["close"] > range_high and bull:
            direction = "long"
        elif row["close"] < range_low and bear:
            direction = "short"
        else:
            continue

        entry = row["close"]
        structural = range_low if direction == "long" else range_high
        structural_dist = abs(entry - structural) / entry
        shares, value, stop_dist = size_position(entry, structural_dist)
        if shares <= 0:
            return None
        stop_price = entry * (1 - stop_dist) if direction == "long" else entry * (1 + stop_dist)

        # relative strength vs SPY since open, in the trade's direction
        rel = 0.0
        if spy_open:
            spy_now = spy_day[spy_day["timestamp"] <= row["timestamp"]]
            if not spy_now.empty:
                stock_move = (entry - day_open) / day_open
                spy_move = (spy_now["close"].iloc[-1] - spy_open) / spy_open
                rel = stock_move - spy_move
                if direction == "short":
                    rel = -rel

        score, parts = score_signal(vol_mult, stop_dist, rel, trend_rank, catalyst=False)
        return dict(symbol=symbol, date=date, direction=direction,
                    entry_time=row["timestamp"], entry_price=entry,
                    stop_price=stop_price, stop_distance_pct=stop_dist,
                    shares=shares, position_value=value,
                    vol_mult=round(vol_mult, 2), rel_strength=round(rel * 100, 2),
                    trend_rank=trend_rank, score=score, **{f"f_{k}": round(v, 2) for k, v in parts.items()})
    return None


def manage_trade(trade, day_df):
    rest = day_df[day_df["timestamp"] > trade.entry_time]
    is_long = trade.direction == "long"
    ep = trade.entry_price
    close_dt = datetime.combine(trade.date, MARKET_CLOSE, tzinfo=TZ)
    eod_cutoff = close_dt - timedelta(minutes=EOD_WARN_MINUTES)

    for _, row in rest.iterrows():
        best = row["high"] if is_long else row["low"]
        worst = row["low"] if is_long else row["high"]
        best_gain = (best - ep) / ep if is_long else (ep - best) / ep
        worst_gain = (worst - ep) / ep if is_long else (ep - worst) / ep

        trade.peak_gain_pct = max(trade.peak_gain_pct, best_gain)
        if trade.peak_gain_pct >= TRAIL_ARM_PCT:
            trade.trail_armed = True

        stop_hit = (worst <= trade.stop_price) if is_long else (worst >= trade.stop_price)
        giveback = trade.peak_gain_pct - TRAIL_GIVEBACK_PCT if trade.trail_armed else None
        trail_hit = trade.trail_armed and worst_gain <= giveback

        if stop_hit and not trade.trail_armed:
            trade.exit_time, trade.exit_price, trade.exit_reason = row["timestamp"], trade.stop_price, "INITIAL_STOP"
            return
        if trail_hit:
            trade.exit_time = row["timestamp"]
            trade.exit_price = ep * (1 + giveback) if is_long else ep * (1 - giveback)
            trade.exit_reason = "TRAIL_STOP"
            return
        if row["timestamp"] >= eod_cutoff:
            trade.exit_time, trade.exit_price, trade.exit_reason = row["timestamp"], row["close"], "EOD_CLOSE"
            return
    if not rest.empty:
        last = rest.iloc[-1]
        trade.exit_time, trade.exit_price, trade.exit_reason = last["timestamp"], last["close"], "DATA_END"


def run_backtest():
    trending = load_trending_log()
    if trending:
        universe = {s for d in trending.values() for s in d}
        print(f"Using trending log: {len(trending)} days, {len(universe)} unique symbols")
    else:
        universe = set(WATCHLIST)
        print("No trending_log.csv found - using fallback WATCHLIST")
    etfs = etf_symbols()
    universe = {s for s in universe if s not in etfs}
    print(f"After ETF exclusion: {len(universe)} symbols")
    universe.add(BENCHMARK)

    bars = fetch_bars(universe)
    if bars.empty:
        print("No data returned.")
        return [], []
    bars["date"] = bars["timestamp"].dt.date

    per_symbol = {}
    for sym, g in bars.groupby("symbol"):
        days = {d: regular_hours(gg.sort_values("timestamp")) for d, gg in g.groupby("date")}
        per_symbol[sym] = {d: v for d, v in days.items() if not v.empty}

    spy_days = per_symbol.get(BENCHMARK, {})
    all_dates = sorted({d for s in per_symbol.values() for d in s})
    test_dates = all_dates[-TEST_TRADING_DAYS:]

    all_signals, chosen = [], []
    for date in test_dates:
        candidates = (set(trending[date]) if trending and date in trending else universe) - etfs - {BENCHMARK}
        day_signals = []
        for sym in candidates:
            days = per_symbol.get(sym)
            if not days or date not in days:
                continue
            rank = trending[date].get(sym) if trending and date in trending else None
            sig = find_signals_for_day(sym, date, days, spy_days.get(date), rank)
            if sig:
                day_signals.append(sig)
        all_signals.extend(day_signals)
        if not day_signals:
            continue
        # top score wins; ties -> earliest breakout
        day_signals.sort(key=lambda s: (-s["score"], s["entry_time"]))
        best = day_signals[0]
        if best["score"] < MIN_SCORE:
            continue
        tr = Trade(best)
        manage_trade(tr, add_indicators(per_symbol[best["symbol"]][date]))
        if tr.exit_price is not None:
            chosen.append(tr)

    # 3-per-5-business-days cap
    chosen.sort(key=lambda t: t.entry_time)
    kept, window = [], []
    for tr in chosen:
        d = tr.entry_time.date()
        window = [w for w in window if np.busday_count(w, d) < 5]
        if len(window) < MAX_TRADES_PER_5_DAYS:
            kept.append(tr)
            window.append(d)
    return kept, all_signals


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def summarize(trades, signals):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if signals:
        pd.DataFrame(signals).sort_values(["date", "score"], ascending=[True, False]) \
            .to_csv(os.path.join(OUTPUT_DIR, "signal_log.csv"), index=False)
    if not trades:
        print("\nNo trades taken in this window.")
        return
    rows = [{
        "date": t.date, "symbol": t.symbol, "direction": t.direction, "score": t.score,
        "vol_mult": t.vol_mult, "stop_pct": round(t.stop_distance_pct * 100, 2),
        "rel_strength_pct": t.rel_strength, "trend_rank": t.trend_rank,
        "entry_time": t.entry_time, "entry_price": round(t.entry_price, 2),
        "exit_time": t.exit_time, "exit_price": round(t.exit_price, 2), "exit_reason": t.exit_reason,
        "shares": t.shares, "position_value": round(t.position_value, 2),
        "peak_gain_pct": round(t.peak_gain_pct * 100, 2),
        "pnl_dollars": round(t.pnl_dollars(), 2), "pnl_pct": round(t.pnl_pct() * 100, 2),
    } for t in trades]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "trade_log.csv"), index=False)

    wins = df[df["pnl_dollars"] > 0]
    print("\n" + "=" * 60 + "\nORB BACKTEST SUMMARY (v2, ranked)\n" + "=" * 60)
    print(f"Window: {START_DATE} to {END_DATE}")
    print(f"Qualified signals (all symbols, all days): {len(signals)}")
    print(f"Days with a signal >= {MIN_SCORE}: {df['date'].nunique()}")
    print(f"Trades taken after 3-per-5-day cap: {len(df)}")
    print(f"Wins: {len(wins)} | Losses: {len(df) - len(wins)} | Win rate: {len(wins) / len(df) * 100:.1f}%")
    print(f"Total P&L: ${df['pnl_dollars'].sum():.2f} | Avg/trade: ${df['pnl_dollars'].mean():.2f}")
    print(f"Largest win: ${df['pnl_dollars'].max():.2f} | Largest loss: ${df['pnl_dollars'].min():.2f}")
    print(f"Avg score of taken trades: {df['score'].mean():.1f}")
    print("\nExit reasons:\n" + df["exit_reason"].value_counts().to_string())
    print("\nWritten: trade_log.csv, signal_log.csv")


if __name__ == "__main__":
    trades, signals = run_backtest()
    summarize(trades, signals)
