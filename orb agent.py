"""
ORB Live Alert Agent  (notify-only; you place the trades)
=========================================================
Daily schedule (America/New_York):
  09:15  Pull Alpaca most-actives, drop ETFs/ETNs/funds, keep top 100 priced >= $5, log to trending_log.csv,
         pull 10-day 1-min history for those names (volume baseline) + SPY.
  09:30  Start collecting 1-min bars.
  09:45  Opening range locked. Alert RANGE_SET summary.
  09:45-10:30  Every minute: scan candidates for ORB breakout (EMA9/20, VWAP, 2x volume),
         score each, and alert the single best signal of the day if score >= MIN_SCORE
         and the 3-per-5-business-day cap allows. One entry per day.
  10:30-15:00  If no ORB entry: scan for MA50 BOUNCE setups (stock declining into its
         50-day SMA from above, touches it, then a 1-min bar closes back above it on 2x volume).
         Same scoring, sizing, exits. ORB has exclusive priority in the first hour.
  After entry: monitor every minute for INITIAL_STOP / TRAIL_STOP; alert on hit.
  15:50  EOD_CLOSE alert if the position is still open.
  Then sleep until the next trading day (uses Alpaca's market clock, skips holidays).

Assumption: when a BUY/SELL signal is sent, the agent tracks a hypothetical position
from the alert price so it can send exit alerts. If you skip a trade, ignore its exits.

Earnings catalyst (Finnhub, free tier): names that reported after yesterday's close or before
today's open get the +10 catalyst score points and a "Reported: beat/miss, gap" line in the alert.
Names reporting tonight / tomorrow pre-market are dropped from the MA50-bounce watch (a bounce
into an earnings gap is a coin flip). Set FINNHUB_API_KEY to enable; without it the layer is off.

Environment variables (put them in /etc/orb-agent.env):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, PUSHOVER_TOKEN, PUSHOVER_USER, FINNHUB_API_KEY (optional)
Optional: ACCOUNT_EQUITY (default 10000), STATE_DIR (default /var/lib/orb-agent)
"""

import os, sys, json, csv, time, math, traceback, re
from datetime import datetime, timedelta, time as dtime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest, MostActivesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, MostActivesBy
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass

# ------------------------------------------------------------------ config
API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
PO_TOKEN = os.environ["PUSHOVER_TOKEN"]
PO_USER = os.environ["PUSHOVER_USER"]
ACCOUNT_EQUITY = float(os.environ.get("ACCOUNT_EQUITY", "10000"))
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
STATE_DIR = os.environ.get("STATE_DIR", "/var/lib/orb-agent")
FEED = DataFeed.IEX

MIN_PRICE = 5.0
TOP_N = 100
RISK_PER_TRADE = 50.0
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", "1.0"))   # 1.0 = one position may use the whole account
STOP_CAP_PCT = 0.02
TRAIL_ARM_PCT = 0.05
TRAIL_GIVEBACK_PCT = 0.02
VOLUME_SPIKE_MULT = 2.0
VOLUME_LOOKBACK_DAYS = 10
MAX_TRADES_PER_5_DAYS = 3
W_VOLUME, W_STOP, W_RS, W_TREND, W_CATALYST = 30, 25, 20, 15, 10
VOLUME_CAP_MULT = 5.0
MIN_SCORE = 50.0

TZ = ZoneInfo("America/New_York")
T_SCREEN = dtime(9, 15)
T_OPEN = dtime(9, 30)
T_RANGE_END = dtime(9, 45)
T_ENTRY_CUTOFF = dtime(10, 30)     # ORB entries end here; bounce scan starts here
T_BOUNCE_CUTOFF = dtime(15, 0)     # last bounce entry
BOUNCE_MIN_DECLINE = 0.05          # >= 5% below its 20-day high
BOUNCE_TOUCH_ABOVE = 0.005         # low within 0.5% above the SMA counts as a touch...
BOUNCE_TOUCH_BELOW = 0.01          # ...or pierces it by < 1%
BOUNCE_LOOKBACK_ABOVE = 20         # must have closed above the SMA within the last 20 sessions
T_EOD_ALERT = dtime(15, 50)
T_CLOSE = dtime(16, 0)

os.makedirs(STATE_DIR, exist_ok=True)
TRADES_FILE = os.path.join(STATE_DIR, "trades.json")
LOG_CSV = os.path.join(STATE_DIR, "trending_log.csv")

data = StockHistoricalDataClient(API_KEY, SECRET_KEY)
screener = ScreenerClient(API_KEY, SECRET_KEY)
trading = TradingClient(API_KEY, SECRET_KEY, paper=True)

ETF_NAME_RE = re.compile(r"\b(ETF|ETN|Trust|Fund|Index|Shares|iShares|ProShares|SPDR|Invesco|Direxion|"
                         r"Vanguard|VanEck|WisdomTree|Schwab|Global X|Ultra|Bull|Bear|2x|3x)\b", re.I)
ETF_EXCHANGES = {"ARCA", "BATS"}
_etf_cache = {"date": None, "symbols": set()}


def etf_symbols():
    """Set of symbols that look like ETFs/ETNs/funds. Refreshed once per day."""
    if _etf_cache["date"] == now().date():
        return _etf_cache["symbols"]
    out = set()
    try:
        assets = trading.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
        for a in assets:
            ex = str(a.exchange.value if hasattr(a.exchange, "value") else a.exchange)
            if ex in ETF_EXCHANGES or ETF_NAME_RE.search(a.name or ""):
                out.add(a.symbol)
        log(f"ETF filter: {len(out)} symbols excluded")
    except Exception as e:  # noqa: BLE001
        log(f"ETF lookup failed ({e}); using fallback list")
        out = {"SPY","QQQ","IWM","DIA","TQQQ","SQQQ","SOXL","SOXS","SPXL","SPXS","UVXY","VXX","XLF","XLE","XLK",
               "ARKK","GLD","SLV","TLT","HYG","LQD","EEM","EFA","VTI","VOO","IVV","XLV","XLI","XLY","XLP","XBI",
               "SMH","KRE","GDX","USO","UNG","TNA","TZA","LABU","LABD","NVDL","TSLL","MSTU","MSTZ","BITO","IBIT"}
    _etf_cache.update(date=now().date(), symbols=out)
    return out


# ------------------------------------------------------------------ helpers
def now():
    return datetime.now(TZ)


def log(msg):
    print(f"{now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def push(title, message, priority=0):
    """priority: -1 quiet, 0 normal, 1 high (bypasses quiet hours), 2 emergency (repeats until acked)."""
    payload = {"token": PO_TOKEN, "user": PO_USER, "title": title, "message": message, "priority": priority}
    if priority == 2:
        payload.update(retry=60, expire=1800)
    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=10)
        if r.status_code != 200:
            log(f"Pushover error {r.status_code}: {r.text}")
    except Exception as e:  # noqa: BLE001
        log(f"Pushover failed: {e}")
    log(f"PUSH [{title}] {message.replace(chr(10), ' | ')}")


def sleep_until(t: datetime):
    s = (t - now()).total_seconds()
    if s > 0:
        time.sleep(s)


def load_trades():
    if os.path.exists(TRADES_FILE):
        return json.load(open(TRADES_FILE))
    return []


def save_trades(trades):
    json.dump(trades, open(TRADES_FILE, "w"), indent=1, default=str)


def trades_in_window(trades, d: date):
    return [t for t in trades if np.busday_count(date.fromisoformat(t["date"]), d) < 5]


# ------------------------------------------------------------------ data
def fetch_screener():
    for top in (200, 100):
        try:
            actives = screener.get_most_actives(MostActivesRequest(top=top, by=MostActivesBy.VOLUME)).most_actives
            break
        except Exception as e:  # noqa: BLE001
            log(f"screener top={top} failed: {e}")
    else:
        return []
    syms = [a.symbol for a in actives]
    prices = {}
    for i in range(0, len(syms), 100):
        snaps = data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=syms[i:i + 100], feed=FEED))
        for s, snap in snaps.items():
            px = None
            for attr in ("latest_trade", "daily_bar", "previous_daily_bar"):
                obj = getattr(snap, attr, None)
                if obj is not None:
                    px = getattr(obj, "price", None) or getattr(obj, "close", None)
                    if px:
                        break
            if px:
                prices[s] = float(px)
    etfs = etf_symbols()
    ranked = []
    for a in actives:
        px = prices.get(a.symbol)
        if a.symbol in etfs:
            continue
        if px and px >= MIN_PRICE:
            ranked.append((a.symbol, px, a.volume))
        if len(ranked) >= TOP_N:
            break
    return ranked


def append_trending_log(ranked, d: date):
    rows = []
    if os.path.exists(LOG_CSV):
        rows = [r for r in csv.DictReader(open(LOG_CSV)) if r["date"] != d.isoformat()]
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "rank", "symbol", "volume", "price"])
        w.writeheader()
        w.writerows(rows)
        for i, (s, px, vol) in enumerate(ranked, 1):
            w.writerow({"date": d.isoformat(), "rank": i, "symbol": s, "volume": vol, "price": px})


def fetch_bars(symbols, start: datetime, end: datetime):
    frames = []
    symbols = sorted(set(symbols))
    for i in range(0, len(symbols), 50):
        req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 50], timeframe=TimeFrame.Minute,
                               start=start, end=end, feed=FEED)
        df = data.get_stock_bars(req).df
        if not df.empty:
            frames.append(df.reset_index())
    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_convert(TZ)
    return bars


def regular_hours(df):
    t = df["timestamp"].dt.time
    return df[(t >= T_OPEN) & (t < T_CLOSE)]


def build_volume_baseline(hist):
    """{symbol: {time: avg_volume}} from prior-day regular-hours 1-min bars."""
    hist = regular_hours(hist).copy()
    hist["tod"] = hist["timestamp"].dt.time
    out = {}
    for sym, g in hist.groupby("symbol"):
        out[sym] = g.groupby("tod")["volume"].mean().to_dict()
    return out


def add_indicators(df):
    df = df.sort_values("timestamp").copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    return df


# ------------------------------------------------------------------ scoring & sizing
def size_position(entry, structural_dist):
    stop_dist = min(structural_dist, STOP_CAP_PCT)
    value = min(RISK_PER_TRADE / stop_dist, ACCOUNT_EQUITY * MAX_POSITION_PCT)
    shares = math.floor(value / entry)
    return shares, shares * entry, stop_dist


def score_signal(vol_mult, stop_dist, rel, rank, catalyst=False):
    f_vol = min(vol_mult, VOLUME_CAP_MULT) / VOLUME_CAP_MULT
    f_stop = 1.0 - min(stop_dist, STOP_CAP_PCT) / STOP_CAP_PCT
    f_rs = min(max(rel, 0.0), 0.02) / 0.02
    f_trend = (1.0 - (rank - 1) / 99.0) if rank else 0.5
    f_cat = 1.0 if catalyst else 0.0
    return round(W_VOLUME * f_vol + W_STOP * f_stop + W_RS * f_rs + W_TREND * f_trend + W_CATALYST * f_cat, 1)


def evaluate_symbol(sym, day_bars, rng, baseline, spy_bars, rank, catalyst=False):
    """Check the LATEST bar for a fresh breakout. Returns a signal dict or None."""
    if day_bars.empty or len(day_bars) < 20:
        return None
    df = add_indicators(day_bars)
    row = df.iloc[-1]
    base = baseline.get(sym, {}).get(row["timestamp"].time())
    if not base:
        return None
    vol_mult = row["volume"] / base
    if vol_mult < VOLUME_SPIKE_MULT:
        return None
    bull = row["ema9"] > row["ema20"] and row["close"] > row["vwap"]
    bear = row["ema9"] < row["ema20"] and row["close"] < row["vwap"]
    if row["close"] > rng["high"] and bull:
        direction = "long"
    elif row["close"] < rng["low"] and bear:
        direction = "short"
    else:
        return None
    entry = float(row["close"])
    structural = rng["low"] if direction == "long" else rng["high"]
    structural_dist = abs(entry - structural) / entry
    shares, value, stop_dist = size_position(entry, structural_dist)
    if shares <= 0:
        return None
    stop_price = entry * (1 - stop_dist) if direction == "long" else entry * (1 + stop_dist)

    rel = 0.0
    if not spy_bars.empty:
        s = spy_bars.sort_values("timestamp")
        spy_move = (s["close"].iloc[-1] - s["open"].iloc[0]) / s["open"].iloc[0]
        stock_move = (entry - df["open"].iloc[0]) / df["open"].iloc[0]
        rel = stock_move - spy_move
        if direction == "short":
            rel = -rel
    score = score_signal(vol_mult, stop_dist, rel, rank, catalyst)
    return dict(symbol=sym, direction=direction, setup="ORB", entry_price=entry, stop_price=stop_price,
                stop_dist=stop_dist, shares=shares, value=value, vol_mult=vol_mult,
                rel=rel, rank=rank, score=score, catalyst=catalyst, time=row["timestamp"].isoformat())


# ------------------------------------------------------------------ earnings catalyst
def fetch_earnings(today: date):
    """Returns (reported, upcoming). reported = {sym: {eps_actual, eps_est, hour, date}} for names that
    reported after the prior session's close or before today's open. upcoming = set reporting tonight or
    tomorrow pre-market."""
    if not FINNHUB_KEY:
        return {}, set()
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    nxt = today + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                         params={"from": prev.isoformat(), "to": nxt.isoformat(), "token": FINNHUB_KEY}, timeout=15)
        r.raise_for_status()
        rows = r.json().get("earningsCalendar", [])
    except Exception as e:  # noqa: BLE001
        log(f"earnings calendar failed: {e}")
        return {}, set()
    reported, upcoming = {}, set()
    for e in rows:
        sym, d, hour = e.get("symbol"), e.get("date"), (e.get("hour") or "").lower()
        if not sym or not d:
            continue
        if (d == prev.isoformat() and hour in ("amc", "dmh", "")) or (d == today.isoformat() and hour == "bmo"):
            reported[sym] = {"eps_actual": e.get("epsActual"), "eps_est": e.get("epsEstimate"), "hour": hour, "date": d}
        elif (d == today.isoformat() and hour in ("amc", "dmh", "")) or (d == nxt.isoformat() and hour == "bmo"):
            upcoming.add(sym)
    return reported, upcoming


def catalyst_line(sym, reported, prev_close=None, today_open=None):
    e = reported.get(sym)
    if not e:
        return ""
    a, est = e.get("eps_actual"), e.get("eps_est")
    if a is not None and est not in (None, 0):
        surprise = (a - est) / abs(est) * 100
        verdict = "BEAT" if a > est else ("MISS" if a < est else "INLINE")
        txt = f"Reported: EPS {a:.2f} vs {est:.2f} est ({verdict} {surprise:+.0f}%)"
    elif a is not None:
        txt = f"Reported: EPS {a:.2f}"
    else:
        txt = "Reported (EPS pending)"
    if prev_close and today_open:
        txt += f", gap {(today_open - prev_close) / prev_close * 100:+.1f}%"
    return txt


# ------------------------------------------------------------------ MA50 bounce
def fetch_daily(symbols, days=90):
    frames = []
    symbols = sorted(set(symbols))
    for i in range(0, len(symbols), 100):
        req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 100], timeframe=TimeFrame(1, TimeFrameUnit.Day),
                               start=now() - timedelta(days=days), end=now() - timedelta(days=1), feed=FEED)
        df = data.get_stock_bars(req).df
        if not df.empty:
            frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bounce_levels(daily):
    """{symbol: {sma50, touch_lo, touch_hi}} for names declining into their 50-day SMA from above."""
    out = {}
    for sym, g in daily.groupby("symbol"):
        g = g.sort_values("timestamp")
        if len(g) < 50:
            continue
        closes = g["close"].to_numpy()
        sma50 = closes[-50:].mean()
        high20 = g["high"].to_numpy()[-20:].max()
        last = closes[-1]
        # rolling SMA over the last 20 sessions to test "was above the SMA recently"
        sma_series = pd.Series(closes).rolling(50).mean().to_numpy()[-BOUNCE_LOOKBACK_ABOVE:]
        was_above = bool(np.nanmax(closes[-BOUNCE_LOOKBACK_ABOVE:] - sma_series) > 0)
        declining = (high20 - last) / high20 >= BOUNCE_MIN_DECLINE
        if was_above and declining and last > sma50 * (1 - BOUNCE_TOUCH_BELOW):
            out[sym] = {"sma50": float(sma50),
                        "touch_hi": float(sma50 * (1 + BOUNCE_TOUCH_ABOVE)),
                        "touch_lo": float(sma50 * (1 - BOUNCE_TOUCH_BELOW))}
    return out


def evaluate_bounce(sym, day_bars, lvl, baseline, spy_bars, rank, catalyst=False):
    """Touch of the 50-SMA earlier today, and the LATEST bar closes back above it on volume."""
    if day_bars.empty or len(day_bars) < 5:
        return None
    df = day_bars.sort_values("timestamp")
    touched = df[(df["low"] <= lvl["touch_hi"]) & (df["low"] >= lvl["touch_lo"])]
    if touched.empty:
        return None
    row = df.iloc[-1]
    if touched.index[-1] == row.name and len(touched) == 1:
        return None  # the touch bar itself is not a confirmation
    if not (row["close"] > lvl["sma50"] and row["close"] > row["open"]):
        return None
    base = baseline.get(sym, {}).get(row["timestamp"].time())
    if not base:
        return None
    vol_mult = row["volume"] / base
    if vol_mult < VOLUME_SPIKE_MULT:
        return None
    entry = float(row["close"])
    touch_low = float(touched["low"].min())
    structural_dist = (entry - touch_low) / entry
    shares, value, stop_dist = size_position(entry, structural_dist)
    if shares <= 0:
        return None
    stop_price = entry * (1 - stop_dist)
    # relative strength measured from the touch bar: how hard it bounced vs SPY over the same minutes
    rel = 0.0
    touch_time = touched["timestamp"].iloc[0]
    if not spy_bars.empty:
        sp = spy_bars.sort_values("timestamp")
        sp_since = sp[sp["timestamp"] >= touch_time]
        if not sp_since.empty:
            spy_move = (sp_since["close"].iloc[-1] - sp_since["open"].iloc[0]) / sp_since["open"].iloc[0]
            rel = (entry - touch_low) / touch_low - spy_move
    score = score_signal(vol_mult, stop_dist, rel, rank, catalyst)
    return dict(symbol=sym, direction="long", setup="MA50_BOUNCE", entry_price=entry, stop_price=stop_price,
                stop_dist=stop_dist, shares=shares, value=value, vol_mult=vol_mult, rel=rel, rank=rank,
                score=score, catalyst=catalyst, time=row["timestamp"].isoformat(), sma50=lvl["sma50"])


def announce_entry(best, slots, cat_text=""):
    side = "BUY" if best["direction"] == "long" else "SELL SHORT"
    tag = best.get("setup", "ORB") + (" +catalyst" if best.get("catalyst") else "")
    push(f"{side} {best['symbol']} — {tag} score {best['score']}",
         f"Entry ~{best['entry_price']:.2f} | stop {best['stop_price']:.2f} ({best['stop_dist']*100:.1f}%)\n"
         f"{best['shares']} sh ≈ ${best['value']:,.0f}\n"
         f"vol {best['vol_mult']:.1f}x · RS {best['rel']*100:+.1f}% · rank #{best['rank']}"
         + (f" · SMA50 {best['sma50']:.2f}" if "sma50" in best else "")
         + (f"\n{cat_text}" if cat_text else "") +
         f"\nTrail arms at +5%, 2pt giveback. Slots left after this: {slots-1}", 2)


def monitor_position(position, today):
    eod = datetime.combine(today, T_EOD_ALERT, tzinfo=TZ)
    is_long = position["direction"] == "long"
    ep = position["entry_price"]
    while now() < eod:
        next_tick = (now() + timedelta(minutes=1)).replace(second=15, microsecond=0)
        try:
            b = fetch_bars([position["symbol"]], datetime.fromisoformat(position["time"]), now())
            b = b[b["timestamp"] > datetime.fromisoformat(position["time"])]
            for _, row in b.iterrows():
                best = row["high"] if is_long else row["low"]
                worst = row["low"] if is_long else row["high"]
                best_gain = (best - ep) / ep if is_long else (ep - best) / ep
                worst_gain = (worst - ep) / ep if is_long else (ep - worst) / ep
                position["peak"] = max(position["peak"], best_gain)
                if position["peak"] >= TRAIL_ARM_PCT and not position["armed"]:
                    position["armed"] = True
                    push(f"{position['symbol']} trailing armed",
                         f"Up {position['peak']*100:.1f}%. Exit if it gives back 2 pts from peak.", 0)
                if not position["armed"] and ((worst <= position["stop_price"]) if is_long else (worst >= position["stop_price"])):
                    push(f"STOP HIT — exit {position['symbol']}",
                         f"Stop {position['stop_price']:.2f} touched. Close the position now.", 2)
                    return
                if position["armed"]:
                    gb = position["peak"] - TRAIL_GIVEBACK_PCT
                    if worst_gain <= gb:
                        lvl = ep * (1 + gb) if is_long else ep * (1 - gb)
                        push(f"TRAIL HIT — exit {position['symbol']}",
                             f"Gave back 2 pts from +{position['peak']*100:.1f}% peak (~{lvl:.2f}). Close now.", 2)
                        return
        except Exception as e:  # noqa: BLE001
            log(f"monitor error: {e}")
        sleep_until(next_tick)
    push(f"EOD — close {position['symbol']}", "10 minutes to close and no exit has triggered. Close the position.", 2)


# ------------------------------------------------------------------ daily loop
def run_trading_day(today: date):
    log(f"=== Trading day {today} ===")
    trades = load_trades()
    used = len(trades_in_window(trades, today))
    slots = MAX_TRADES_PER_5_DAYS - used

    # 09:15 screener
    sleep_until(datetime.combine(today, T_SCREEN, tzinfo=TZ))
    ranked = fetch_screener()
    if not ranked:
        push("ORB agent", "Screener returned nothing — no scan today.", 1)
        return
    append_trending_log(ranked, today)
    symbols = [s for s, _, _ in ranked]
    rank_of = {s: i for i, (s, _, _) in enumerate(ranked, 1)}
    log(f"Screener: {len(symbols)} candidates. Top 5: {symbols[:5]}")

    # volume baseline from prior 10 trading days (~16 calendar days)
    hist = fetch_bars(symbols, datetime.combine(today - timedelta(days=16), dtime(0, 0), tzinfo=TZ),
                      datetime.combine(today, dtime(0, 0), tzinfo=TZ))
    baseline = build_volume_baseline(hist)
    try:
        levels = bounce_levels(fetch_daily(symbols))
    except Exception as e:  # noqa: BLE001
        log(f"daily bars failed ({e}); bounce scan disabled today")
        levels = {}
    reported, upcoming = fetch_earnings(today)
    reported = {k: v for k, v in reported.items() if k in rank_of}
    dropped = [s for s in levels if s in upcoming]
    for s_ in dropped:
        levels.pop(s_, None)
    prev_close = {}
    try:
        for sym, g in fetch_daily(symbols, days=10).groupby("symbol"):
            prev_close[sym] = float(g.sort_values("timestamp")["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    log(f"Earnings: {len(reported)} candidates reported ({sorted(reported)[:8]}); "
        f"{len(dropped)} dropped from bounce watch for reporting tonight")
    log(f"MA50 bounce watchlist: {len(levels)} names: {sorted(levels)[:10]}")
    push("ORB agent ready", f"{len(symbols)} candidates, {slots} trade slot(s) left this week.\n"
         f"Top: {', '.join(symbols[:8])}\nMA50-bounce watch: {len(levels)}"
         + (f"\nEarnings catalysts: {', '.join(sorted(reported)[:8])}" if reported else "\nEarnings feed: off"), 0 if slots else 1)
    if slots <= 0:
        log("No trade slots this week; will still send RANGE_SET then sleep.")

    # 09:45 opening range
    sleep_until(datetime.combine(today, T_RANGE_END, tzinfo=TZ) + timedelta(seconds=20))
    day_start = datetime.combine(today, T_OPEN, tzinfo=TZ)
    bars = fetch_bars(symbols + ["SPY"], day_start, now())
    ranges = {}
    for sym, g in bars.groupby("symbol"):
        g = g[g["timestamp"].dt.time < T_RANGE_END]
        if not g.empty:
            ranges[sym] = {"high": float(g["high"].max()), "low": float(g["low"].min())}
    push("RANGE_SET 9:45", f"Opening ranges locked for {len(ranges)} names. Scanning until 10:30.", -1)
    if slots <= 0:
        return

    # 09:45-10:30 scan loop, one signal per day
    position = None
    cutoff = datetime.combine(today, T_ENTRY_CUTOFF, tzinfo=TZ)
    while now() < cutoff and position is None:
        next_tick = (now() + timedelta(minutes=1)).replace(second=15, microsecond=0)
        try:
            bars = fetch_bars(symbols + ["SPY"], day_start, now())
            bars = regular_hours(bars)
            spy = bars[bars["symbol"] == "SPY"]
            signals = []
            for sym, g in bars.groupby("symbol"):
                if sym == "SPY" or sym not in ranges:
                    continue
                sig = evaluate_symbol(sym, g, ranges[sym], baseline, spy, rank_of.get(sym), sym in reported)
                if sig:
                    signals.append(sig)
            if signals:
                signals.sort(key=lambda s: -s["score"])
                best = signals[0]
                log(f"{len(signals)} signal(s); best {best['symbol']} {best['score']}")
                if best["score"] >= MIN_SCORE:
                    position = dict(best, date=today.isoformat(), peak=0.0, armed=False)
                    trades.append({"date": today.isoformat(), "symbol": best["symbol"],
                                   "direction": best["direction"], "entry": best["entry_price"]})
                    save_trades(trades)
                    g0 = bars[bars["symbol"] == best["symbol"]].sort_values("timestamp")
                    announce_entry(best, slots, catalyst_line(best["symbol"], reported,
                                   prev_close.get(best["symbol"]), float(g0["open"].iloc[0]) if not g0.empty else None))
        except Exception as e:  # noqa: BLE001
            log(f"scan error: {e}\n{traceback.format_exc()}")
        sleep_until(next_tick)

    if position is None and levels:
        push("ORB window closed", f"No ORB entry. Scanning {len(levels)} names for MA50 bounces until 3:00.", -1)
        cutoff = datetime.combine(today, T_BOUNCE_CUTOFF, tzinfo=TZ)
        watch = [s for s in levels]
        while now() < cutoff and position is None:
            next_tick = (now() + timedelta(minutes=1)).replace(second=15, microsecond=0)
            try:
                bars = regular_hours(fetch_bars(watch + ["SPY"], day_start, now()))
                spy = bars[bars["symbol"] == "SPY"]
                signals = []
                for sym, g in bars.groupby("symbol"):
                    if sym in levels:
                        sig = evaluate_bounce(sym, g, levels[sym], baseline, spy, rank_of.get(sym), sym in reported)
                        if sig:
                            signals.append(sig)
                if signals:
                    signals.sort(key=lambda s: -s["score"])
                    best = signals[0]
                    log(f"{len(signals)} bounce signal(s); best {best['symbol']} {best['score']}")
                    if best["score"] >= MIN_SCORE:
                        position = dict(best, date=today.isoformat(), peak=0.0, armed=False)
                        trades.append({"date": today.isoformat(), "symbol": best["symbol"], "setup": "MA50_BOUNCE",
                                       "direction": "long", "entry": best["entry_price"]})
                        save_trades(trades)
                        g0 = bars[bars["symbol"] == best["symbol"]].sort_values("timestamp")
                    announce_entry(best, slots, catalyst_line(best["symbol"], reported,
                                   prev_close.get(best["symbol"]), float(g0["open"].iloc[0]) if not g0.empty else None))
            except Exception as e:  # noqa: BLE001
                log(f"bounce scan error: {e}\n{traceback.format_exc()}")
            sleep_until(next_tick)

    if position is None:
        push("No entry today", "No qualifying ORB or MA50-bounce signal.", -1)
        return
    monitor_position(position, today)


def main():
    push("ORB agent started", f"Running on {os.uname().nodename}. Will scan each trading day.", -1)
    while True:
        try:
            clock = trading.get_clock()
            next_open = clock.next_open.astimezone(TZ)
            today = now().date()
            market_today = clock.is_open or next_open.date() == today
            if market_today and now().time() < T_EOD_ALERT:
                run_trading_day(today)
            # sleep to 09:00 of the next trading day
            target_day = next_open.date() if next_open.date() > today or not market_today else today + timedelta(days=1)
            wake = datetime.combine(target_day, dtime(9, 0), tzinfo=TZ)
            if wake <= now():
                wake = now() + timedelta(hours=1)
            log(f"Sleeping until {wake}")
            sleep_until(wake)
        except Exception as e:  # noqa: BLE001
            log(f"main loop error: {e}\n{traceback.format_exc()}")
            push("ORB agent error", str(e)[:300], 1)
            time.sleep(300)


if __name__ == "__main__":
    main()
