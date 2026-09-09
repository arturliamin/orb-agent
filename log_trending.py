"""
Daily Trending-Stock Logger
===========================
Pulls Alpaca's most-actives screener pre-market, drops anything under $5,
keeps the top 100, and appends them to trending_log.csv.

Run once per trading day before the open (the GitHub Actions workflow in
this repo does that automatically). After ~4 weeks the log is the exact
candidate universe the live ORB agent would have seen each day, and the
backtester can replay it with no lookahead.

Columns written: date, rank, symbol, volume, trade_count, price, logged_at_utc
"""

import os
import sys
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MostActivesRequest, StockSnapshotRequest
from alpaca.data.enums import MostActivesBy, DataFeed

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")

MIN_PRICE = 5.00          # SEC penny-stock threshold
KEEP_TOP = 100
LOG_PATH = os.environ.get("TRENDING_LOG_PATH", "trending_log.csv")
TZ = ZoneInfo("America/New_York")


def fetch_most_actives(screener: ScreenerClient):
    """Ask for more than 100 so the price filter still leaves ~100.
    Fall back to 100 if the API rejects the larger request."""
    for top in (200, 100):
        try:
            res = screener.get_most_actives(
                MostActivesRequest(top=top, by=MostActivesBy.VOLUME))
            return res.most_actives
        except Exception as exc:  # noqa: BLE001
            print(f"top={top} failed: {exc}")
    sys.exit("Screener request failed.")


def latest_prices(data: StockHistoricalDataClient, symbols):
    """Latest trade price per symbol via snapshots (batched)."""
    prices = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        snaps = data.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=batch, feed=DataFeed.IEX))
        for sym, snap in snaps.items():
            px = None
            if snap.latest_trade is not None:
                px = snap.latest_trade.price
            elif snap.daily_bar is not None:
                px = snap.daily_bar.close
            elif snap.previous_daily_bar is not None:
                px = snap.previous_daily_bar.close
            if px is not None:
                prices[sym] = float(px)
    return prices


def main():
    screener = ScreenerClient(API_KEY, SECRET_KEY)
    data = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    actives = fetch_most_actives(screener)
    symbols = [a.symbol for a in actives]
    prices = latest_prices(data, symbols)

    today = datetime.now(TZ).date().isoformat()
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    rank = 0
    for a in actives:                      # already sorted by volume, descending
        px = prices.get(a.symbol)
        if px is None or px < MIN_PRICE:
            continue
        rank += 1
        rows.append({
            "date": today,
            "rank": rank,
            "symbol": a.symbol,
            "volume": a.volume,
            "trade_count": a.trade_count,
            "price": round(px, 2),
            "logged_at_utc": now_utc,
        })
        if rank >= KEEP_TOP:
            break

    # Idempotent: if today was already logged, replace it rather than duplicate.
    existing = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["date"] != today]

    fields = ["date", "rank", "symbol", "volume", "trade_count", "price", "logged_at_utc"]
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)
        w.writerows(rows)

    print(f"{today}: logged {len(rows)} symbols (>= ${MIN_PRICE:.2f}) to {LOG_PATH}")
    print("Top 10:", ", ".join(r["symbol"] for r in rows[:10]))


if __name__ == "__main__":
    main()
