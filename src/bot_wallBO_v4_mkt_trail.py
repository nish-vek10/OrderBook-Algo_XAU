#!/usr/bin/env python
"""
bot_wallBO_v4_mkt_trail.py
====================

WBv4 (Market-only) — OrderBook/PositionBook-informed Support/Resistance + Breakout/Bounce
---------------------------------------------------------------------------------------

GOALS
1) Reduce logging noise:
   - Save trimmed orderBook/positionBook snapshots (±BOOK_WINDOW_AROUND_PRICE) around rounded ref price.
   - Keep cadence integrity; detect & log gaps (BOOKS_GAP_DETECTED / CANDLES_GAP_DETECTED) with metadata only.

2) Trailing stop (well-behaved):
   - Trailing activates after +TRAIL_START_R
   - Once active: BE + buffer AND ATR trail; monotonic SL tightening only.

3) Strict risk rule:
   - HARD: at most ONE entry per candle_time (global, across both breakout + bounce, long + short).
   - Also respects MAX_OPEN_POSITIONS (counts open positions only; no limits in v4).

4) Strategy (market-only):
   - Uses latest wall snapshot where wall_dt <= candle_dt (regime).
   - Walls are treated as support/resistance levels derived from OANDA orderBook buckets:
       • support = strongest qualifying bucket BELOW ref_price
       • resistance = strongest qualifying bucket ABOVE ref_price
     (We keep CSV column names buy_wall_price/sell_wall_price for compatibility, but interpret them as support/resistance.)
   - Two play types within each wall regime:
       A) Breakout: retest-gated breakout through support/resistance with EMA filter
       B) Bounce: rejection at support/resistance (mean-revert) with EMA sanity filter

DATA LOGGING
- OANDA orderBook + positionBook snapshots -> JSONL (append-only, dedupe by time)
  Trimmed buckets only around rounded ref price (±BOOK_WINDOW_AROUND_PRICE)
- OANDA candles M1 (mid) -> JSONL (append-only, dedupe by candle time)
  Polled frequently to avoid 20-minute latency; still supports catching up missed candles
- Walls extraction -> CSV (append-only)
  Using support/resistance selection within ±BOOK_RANGE_DOLLARS of ref_price

EXECUTION
- Trades placed on MT5 (ICMarkets demo etc.)
- Risk sizing uses MT5 order_calc_profit so cash-risk aligns to RISK_CASH
- SL/TP broker-managed; time exit closes at market
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import MetaTrader5 as mt5
from pytz import timezone as pytz_timezone


# =============================================================================
# MT5 ACCOUNT CONFIG
# =============================================================================
MT5_LOGIN         = 52699804
MT5_PASSWORD      = "agWVZ&0YssnIQF"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\ALGO-OrderBook_XAU-trail_limits\terminal64.exe"

LOCAL_TZ   = pytz_timezone("Europe/London")
MT5_SYMBOL = "XAUUSD"   # adjust if broker uses suffix
SYMBOL_INFO = None      # filled by init_mt5()

# =============================================================================
# OANDA CONFIG
# =============================================================================
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"  # practice
OANDA_TOKEN   = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
INSTRUMENT    = "XAU_USD"

# =============================================================================
# BOOKS LOGGER CONFIG
# =============================================================================
BOOK_STEP_SECONDS        = 20 * 60          # OANDA books cadence
BOOK_GRACE_SECONDS       = 60              # retry window when expecting a new snapshot
BOOK_RETRY_EVERY_SECONDS = 10
BOOK_CATCHUP_MAX_SECONDS = 60              # if gap detected, keep trying for up to 60s

BOOK_RANGE_DOLLARS = 25.0                  # for selecting walls
WALL_TOTAL_MIN     = 0.08
WALL_IMB_MIN       = 0.06

BOOK_WINDOW_AROUND_PRICE = 150.0           # trimming saved buckets around rounded ref price (noise reduction)

# =============================================================================
# CANDLES LOGGER CONFIG
# =============================================================================
GRANULARITY           = "M1"
PRICE_TYPE            = "M"                 # midpoint

# Poll frequently to avoid 20-minute latency; still append-only dedupe.
CANDLE_POLL_SECONDS   = 10
CANDLE_LOOKBACK_MIN   = 5                   # fetch last N minutes each poll
CANDLE_GAP_WARN_MIN   = 2                   # if newest candle jumps by > N minutes, log gap

# =============================================================================
# STRATEGY CONFIG
# =============================================================================
EMA_SPAN          = 9
MIN_EMA_DIST      = 0.5

RETESTS_REQUIRED  = 6
TOUCH_DIST        = 0.75

BREAK_BUFFER      = 0.2
STOP_BUFFER       = 1.0
TP_R              = 2.5
MAX_WALL_DISTANCE = 12.0

# Bounce mode
ENABLE_BOUNCE     = True
BOUNCE_BUFFER     = 0.25           # must close back inside by this much after touching wall
TP_R_BOUNCE       = 1.5            # bounce TP in R units (often smaller than breakout)
BOUNCE_MAX_EMA_DIST = 6.0          # avoid bounce if price is extremely stretched vs EMA

# =============================================================================
# TRAILING STOP (BE + ATR)
# =============================================================================
ENABLE_TRAILING   = True

TRAIL_START_R     = 1.0            # start BE+ATR trailing once profit >= 1R
BE_BUFFER         = 0.50           # XAU units (0.50 = 50 cents)

ATR_TRAIL_PERIOD  = 20
ATR_TRAIL_MULT    = 2.5

# =============================================================================
# EXECUTION / RISK
# =============================================================================
RISK_CASH        = 500.0
MAX_HOLD_MINUTES = 180

ALLOW_LONGS      = True
ALLOW_SHORTS     = True

MAX_OPEN_POSITIONS      = 3
MAX_ENTRIES_PER_CANDLE  = 1        # hard per-candle cap (we enforce globally)

SIGNALS_ONLY     = False

MAGIC            = 2209001
DEVIATION        = 15

LOOP_SLEEP_SECONDS = 1.0

# =============================================================================
# PORTABLE PATHS (asset + strategy namespace)  [KEEP THIS STRUCTURE]
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

ASSET_TAG = INSTRUMENT.strip().replace("/", "_").replace(":", "_")
STRATEGY_TAG = f"WBv4_MKT_TRAIL-{ASSET_TAG}"

OUT_RAW_ASSET_DIR = BASE_DIR / "input" / "raw" / STRATEGY_TAG / ASSET_TAG
OUT_ORDER_DIR = OUT_RAW_ASSET_DIR / "orderbook"
OUT_POS_DIR   = OUT_RAW_ASSET_DIR / "positionbook"
OUT_CAND_DIR  = OUT_RAW_ASSET_DIR / "candles"

OUT_WALLS_DIR = BASE_DIR / "output" / "reports" / "walls" / STRATEGY_TAG / ASSET_TAG
OUT_LIVE_DIR  = BASE_DIR / "output" / "live" / STRATEGY_TAG / ASSET_TAG

for d in [OUT_ORDER_DIR, OUT_POS_DIR, OUT_CAND_DIR, OUT_WALLS_DIR, OUT_LIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRADES_JSONL = OUT_LIVE_DIR / "trades.jsonl"
STATE_JSON   = OUT_LIVE_DIR / "state.json"


# =============================================================================
# Helpers
# =============================================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def parse_oanda_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def parse_z_time(s: str) -> datetime:
    if "." in s:
        s = s.split(".")[0] + "Z"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def fmt_z_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def day_yyyymmdd(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d")

def round_to_whole_price(x: float) -> float:
    # nearest whole number (standard rounding)
    return float(int(round(x)))

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")

def read_last_jsonl_line(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None

def append_trade_log(row: Dict[str, Any]) -> None:
    obj = dict(row)
    obj.setdefault("schema_version", 4)
    obj.setdefault("dt_utc", fmt_z_time(datetime.now(timezone.utc)))
    with TRADES_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

def save_state(state: Dict[str, Any]) -> None:
    with STATE_JSON.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_JSON.exists():
        return None
    try:
        with STATE_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def apply_state_to_botstate(state: Any, data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if hasattr(state, k):
            setattr(state, k, v)

def get_book_time_and_price(book_obj: dict, key: str) -> Tuple[Optional[str], Optional[float]]:
    book = book_obj.get(key, {})
    t = book.get("time")
    p = book.get("price")
    return t, (safe_float(p) if p is not None else None)

def normalize_candles(raw: dict) -> List[Dict[str, Any]]:
    out = []
    for c in raw.get("candles", []):
        t = c.get("time")
        mid = c.get("mid", {})
        out.append({
            "time": t,
            "complete": bool(c.get("complete", False)),
            "volume": int(c.get("volume", 0)),
            "o": mid.get("o"),
            "h": mid.get("h"),
            "l": mid.get("l"),
            "c": mid.get("c"),
        })
    return out

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / float(period), adjust=False).mean()

def load_walls_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # robust UTC datetime parsing
    df["wall_dt"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["wall_dt"])

    # convert numeric columns safely (do NOT touch time / wall_dt)
    for c in df.columns:
        if c in ("time", "wall_dt"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("wall_dt").reset_index(drop=True)

def latest_wall_row(walls: pd.DataFrame, t: datetime) -> Optional[pd.Series]:
    if walls is None or walls.empty:
        return None
    if "wall_dt" not in walls.columns:
        return None

    # Ensure wall_dt is datetime64[ns] (UTC-naive) for numpy searchsorted
    wall_vals = walls["wall_dt"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    # Ensure t is UTC, then make it UTC-naive datetime64[ns]
    t_utc = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
    t64 = np.datetime64(t_utc.replace(tzinfo=None), "ns")

    idx = int(np.searchsorted(wall_vals, t64, side="right") - 1)
    if idx < 0 or idx >= len(walls):
        return None
    return walls.iloc[idx]

def append_walls_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    # Keep historical column names buy_wall/sell_wall for compatibility,
    # but interpret:
    #   buy_wall_price  = support_price
    #   sell_wall_price = resistance_price
    fieldnames = [
        "time", "ref_price", "bucket_width",

        "buy_wall_price", "buy_strength", "buy_imbalance", "buy_long", "buy_short", "buy_total",
        "sell_wall_price", "sell_strength", "sell_imbalance", "sell_long", "sell_short", "sell_total",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def trim_book_buckets(*, book: dict, window: float) -> dict:
    """
    Reduce noise: keep only buckets within ±window of rounded ref price.
    Works for both orderBook and positionBook.
    """
    if not isinstance(book, dict):
        return book
    price = safe_float(book.get("price"), default=float("nan"))
    buckets = book.get("buckets", [])
    if (not buckets) or (price != price):
        return book

    center = round_to_whole_price(price)
    lo = center - float(window)
    hi = center + float(window)

    trimmed = []
    for b in buckets:
        p = safe_float(b.get("price"), default=float("nan"))
        if p != p:
            continue
        if lo <= p <= hi:
            trimmed.append(b)

    out = dict(book)
    out["buckets"] = trimmed
    out["trim_center"] = center
    out["trim_window"] = float(window)
    out["trim_kept_n"] = len(trimmed)
    out["trim_total_n"] = len(buckets)
    return out

def compute_support_resistance_from_orderbook(
    *,
    ob: dict,
    range_dollars: float,
    total_min: float,
    imb_min: float,
) -> dict:
    """
    Support/Resistance selection within ref_price ± range_dollars.

    Support: best qualifying bucket BELOW ref_price
    Resistance: best qualifying bucket ABOVE ref_price

    "Best" = max (abs(imbalance), total)  (same style as v2, but split by side of ref)
    """
    book = ob.get("orderBook", {})
    t = book.get("time")
    ref_price = safe_float(book.get("price"), default=float("nan"))
    bucket_width = safe_float(book.get("bucketWidth"), default=float("nan"))

    buckets = book.get("buckets", [])
    if not buckets or ref_price != ref_price:
        return {
            "time": t,
            "ref_price": ref_price,
            "bucket_width": bucket_width,
            "buy_wall_price": None,
            "buy_strength": None,
            "buy_imbalance": None,
            "buy_long": None,
            "buy_short": None,
            "buy_total": None,
            "sell_wall_price": None,
            "sell_strength": None,
            "sell_imbalance": None,
            "sell_long": None,
            "sell_short": None,
            "sell_total": None,
        }

    lo = ref_price - range_dollars
    hi = ref_price + range_dollars

    best_support = None   # (strength, total, row)
    best_resist  = None

    for b in buckets:
        p = safe_float(b.get("price"), default=float("nan"))
        if p != p:
            continue
        if p < lo or p > hi:
            continue

        longp = safe_float(b.get("longCountPercent"))
        shortp = safe_float(b.get("shortCountPercent"))
        total = longp + shortp
        imb = longp - shortp

        if total < total_min:
            continue
        if abs(imb) < imb_min:
            continue

        row = {
            "price": p,
            "long": longp,
            "short": shortp,
            "total": total,
            "imb": imb,
            "strength": abs(imb),
        }

        key = (row["strength"], row["total"])

        if p <= ref_price:
            if (best_support is None) or (key > (best_support[0], best_support[1])):
                best_support = (row["strength"], row["total"], row)
        else:
            if (best_resist is None) or (key > (best_resist[0], best_resist[1])):
                best_resist = (row["strength"], row["total"], row)

    sup = None if best_support is None else best_support[2]
    res = None if best_resist is None else best_resist[2]

    # buy_wall_* columns carry SUPPORT
    # sell_wall_* columns carry RESISTANCE
    return {
        "time": t,
        "ref_price": ref_price,
        "bucket_width": bucket_width,

        "buy_wall_price": None if sup is None else sup["price"],
        "buy_strength":   None if sup is None else sup["strength"],
        "buy_imbalance":  None if sup is None else sup["imb"],
        "buy_long":       None if sup is None else sup["long"],
        "buy_short":      None if sup is None else sup["short"],
        "buy_total":      None if sup is None else sup["total"],

        "sell_wall_price": None if res is None else res["price"],
        "sell_strength":   None if res is None else res["strength"],
        "sell_imbalance":  None if res is None else res["imb"],
        "sell_long":       None if res is None else res["long"],
        "sell_short":      None if res is None else res["short"],
        "sell_total":      None if res is None else res["total"],
    }


# =============================================================================
# OANDA HTTP
# =============================================================================
def oanda_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def fetch_json(endpoint: str, timeout: int = 30, params: Optional[dict] = None) -> dict:
    url = f"{OANDA_API_URL}{endpoint}"
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

def fetch_books() -> Tuple[dict, dict]:
    ob = fetch_json(f"/instruments/{INSTRUMENT}/orderBook", timeout=20)
    pb = fetch_json(f"/instruments/{INSTRUMENT}/positionBook", timeout=20)
    return ob, pb

def fetch_candles(time_from: str, time_to: str) -> dict:
    params = {"granularity": GRANULARITY, "from": time_from, "to": time_to, "price": PRICE_TYPE}
    return fetch_json(f"/instruments/{INSTRUMENT}/candles", timeout=30, params=params)


# =============================================================================
# MT5 helpers
# =============================================================================
def init_mt5() -> None:
    global SYMBOL_INFO
    if not mt5.initialize(path=MT5_TERMINAL_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    if not mt5.symbol_select(MT5_SYMBOL, True):
        raise RuntimeError(f"symbol_select failed: {MT5_SYMBOL}")

    info = mt5.symbol_info(MT5_SYMBOL)
    if info is None:
        raise RuntimeError("symbol_info None")
    SYMBOL_INFO = info

    acc = mt5.account_info()
    print("filling_mode:", getattr(info, "filling_mode", None))
    print("trade_fill_mode:", getattr(info, "trade_fill_mode", None))
    print(f"[MT5] Connected login={acc.login} balance={acc.balance:.2f} equity={acc.equity:.2f}")
    print(f"[MT5] {MT5_SYMBOL} digits={info.digits} point={info.point} vol_min={info.volume_min} step={info.volume_step}")

def shutdown_mt5() -> None:
    mt5.shutdown()

def norm_price(x: float) -> float:
    return float(round(x, SYMBOL_INFO.digits))

def clamp_to_step(val: float, vmin: float, vmax: float, step: float) -> float:
    val = max(vmin, min(vmax, val))
    if step <= 0:
        return val
    n = round((val - vmin) / step)
    return vmin + n * step

def get_supported_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    fm = getattr(info, "filling_mode", None)
    if fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        return int(fm)
    return mt5.ORDER_FILLING_IOC

def get_open_positions_on_symbol(symbol: str) -> List[mt5.TradePosition]:
    poss = mt5.positions_get(symbol=symbol)
    if not poss:
        return []
    return [p for p in poss if int(getattr(p, "magic", 0)) == int(MAGIC)]

def calc_volume_for_cash_risk(symbol: str, side: str, entry: float, sl: float, risk_cash: float) -> float:
    if risk_cash <= 0:
        return 0.0
    order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
    profit_1lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
    if profit_1lot is None:
        raise RuntimeError(f"order_calc_profit None: {mt5.last_error()}")
    loss_1lot = abs(float(profit_1lot))
    if loss_1lot <= 0:
        return 0.0
    raw_vol = risk_cash / loss_1lot
    info = SYMBOL_INFO
    return float(clamp_to_step(raw_vol, info.volume_min, info.volume_max, info.volume_step))

def send_market_order(symbol: str, side: str, volume: float, sl: float, tp: float, comment: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"tick None: {mt5.last_error()}"

    price = float(tick.ask) if side == "long" else float(tick.bid)
    price = norm_price(price)
    sl = norm_price(sl)
    tp = norm_price(tp)

    order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
    modes_to_try = [get_supported_filling_mode(symbol), mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]
    seen = []

    for mode in modes_to_try:
        if mode in seen:
            continue
        seen.append(mode)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(DEVIATION),
            "magic": int(MAGIC),
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": int(mode),
        }

        res = mt5.order_send(req)
        if res is None:
            continue
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, res
        if "Unsupported filling mode" in str(res):
            continue
        return False, res

    return False, "All filling modes rejected"

def close_position_market(pos: mt5.TradePosition, reason: str) -> Tuple[bool, Any]:
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return False, f"tick None: {mt5.last_error()}"

    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)

    price = norm_price(price)

    modes_to_try = [
        get_supported_filling_mode(pos.symbol),
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_FOK,
    ]
    seen = set()

    for mode in modes_to_try:
        if mode in seen:
            continue
        seen.add(mode)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "volume": float(pos.volume),
            "type": order_type,
            "price": price,
            "deviation": int(DEVIATION),
            "magic": int(MAGIC),
            "comment": f"TIME_EXIT:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": int(mode),
        }

        res = mt5.order_send(req)
        if res is None:
            continue
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, res
        if "Unsupported filling mode" in str(res):
            continue
        return False, res

    return False, "All filling modes rejected"

def modify_position_sl_tp(pos: mt5.TradePosition, sl: Optional[float], tp: Optional[float]) -> Tuple[bool, Any]:
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": pos.ticket,
        "sl": float(norm_price(sl)) if sl is not None else float(pos.sl),
        "tp": float(norm_price(tp)) if tp is not None else float(pos.tp),
        "magic": int(MAGIC),
        "comment": "TRAIL_SLTP"[:31],
    }
    res = mt5.order_send(req)
    ok = (res is not None) and (res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES))
    return ok, res


# =============================================================================
# Bot State
# =============================================================================
@dataclass
class BotState:
    instrument: str = ""

    # books
    last_saved_book_time: Optional[str] = None
    next_books_wake_utc: Optional[str] = None

    # candles
    last_saved_candle_time: Optional[str] = None
    next_candle_poll_utc: Optional[str] = None

    # candles print aggregation (console only)
    candles_since_books: int = 0
    last_books_print_time: Optional[str] = None

    # one-time bootstrap flags
    did_bootstrap_books: bool = False
    did_bootstrap_candles: bool = False

    # strategy processing
    last_processed_candle_time: Optional[str] = None
    last_wall_time: Optional[str] = None
    retests_support: int = 0
    retests_resist: int = 0

    # strict per-candle entry guard
    last_entry_candle_time: Optional[str] = None
    entries_this_candle: int = 0

    # position tracking
    pos_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    tick_n: int = 0


# =============================================================================
# Books logger (trimmed save + walls CSV + gap detect)
# =============================================================================
def fetch_and_print_books_status(state: BotState) -> None:
    """
    One-shot bootstrap: fetch current books ONCE and print status,
    even if the snapshot time hasn't advanced yet.
    Does NOT spam, does NOT loop.
    """
    try:
        ob_raw, pb_raw = fetch_books()
        t_ob, p_ob = get_book_time_and_price(ob_raw, "orderBook")
        if not t_ob:
            print("[BOOKS] BOOTSTRAP: no time in snapshot")
            return

        # If it's a genuinely new snapshot, save it (dedupe still respected).
        if t_ob != state.last_saved_book_time:
            day = day_yyyymmdd(datetime.now(timezone.utc))
            order_path = OUT_ORDER_DIR / f"orderbook_{day}.jsonl"
            pos_path   = OUT_POS_DIR   / f"positionbook_{day}.jsonl"
            walls_csv  = OUT_WALLS_DIR / f"walls_{day}.csv"

            ob = dict(ob_raw)
            pb = dict(pb_raw)
            ob["orderBook"] = trim_book_buckets(book=ob_raw.get("orderBook", {}), window=BOOK_WINDOW_AROUND_PRICE)
            pb["positionBook"] = trim_book_buckets(book=pb_raw.get("positionBook", {}), window=BOOK_WINDOW_AROUND_PRICE)

            append_jsonl(order_path, ob)
            append_jsonl(pos_path, pb)
            state.last_saved_book_time = t_ob

            walls_row = compute_support_resistance_from_orderbook(
                ob=ob_raw,
                range_dollars=BOOK_RANGE_DOLLARS,
                total_min=WALL_TOTAL_MIN,
                imb_min=WALL_IMB_MIN,
            )
            append_walls_csv(walls_csv, walls_row)
        else:
            # Snapshot unchanged: still compute walls_row for printing only
            walls_row = compute_support_resistance_from_orderbook(
                ob=ob_raw,
                range_dollars=BOOK_RANGE_DOLLARS,
                total_min=WALL_TOTAL_MIN,
                imb_min=WALL_IMB_MIN,
            )

        candles_n = int(getattr(state, "candles_since_books", 0) or 0)
        try:
            open_n = len(get_open_positions_on_symbol(MT5_SYMBOL))
        except Exception:
            open_n = -1

        print(
            f"[BOOKS] BOOTSTRAP {t_ob} ref={p_ob} "
            f"SUP={walls_row.get('buy_wall_price')} RES={walls_row.get('sell_wall_price')} \n"
            f"[CANDLES] +{candles_n} newest={state.last_saved_candle_time} \n"
            f"[HB] open={open_n} last_book={state.last_saved_book_time} wall={state.last_wall_time}"
        )

        # schedule next normal wake off this snapshot time
        try:
            last_dt = parse_oanda_time(t_ob)
            next_expected = last_dt + timedelta(seconds=BOOK_STEP_SECONDS)
            state.next_books_wake_utc = fmt_z_time(next_expected + timedelta(seconds=5))
        except Exception:
            state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=BOOK_STEP_SECONDS))

        # reset candles counter after we printed it alongside books
        state.candles_since_books = 0
        state.last_books_print_time = t_ob

    except Exception as e:
        append_trade_log({"event": "BOOKS_BOOTSTRAP_ERR", "err": str(e)})
        print(f"[BOOKS] BOOTSTRAP_ERR: {e}")


def maybe_run_books_logger(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    day = day_yyyymmdd(now)

    order_path = OUT_ORDER_DIR / f"orderbook_{day}.jsonl"
    pos_path   = OUT_POS_DIR   / f"positionbook_{day}.jsonl"
    walls_csv  = OUT_WALLS_DIR / f"walls_{day}.csv"

    # init last_saved_book_time from file if missing
    if state.last_saved_book_time is None:
        last_obj = read_last_jsonl_line(order_path)
        if last_obj:
            t_last, _ = get_book_time_and_price(last_obj, "orderBook")
            if t_last:
                state.last_saved_book_time = t_last

    # compute next wake time if missing
    if state.next_books_wake_utc is None:
        if state.last_saved_book_time:
            try:
                last_dt = parse_oanda_time(state.last_saved_book_time)
                next_expected = last_dt + timedelta(seconds=BOOK_STEP_SECONDS)
                wake = next_expected + timedelta(seconds=5)
            except Exception:
                wake = now + timedelta(seconds=10)
        else:
            wake = now
        state.next_books_wake_utc = fmt_z_time(wake)

    wake_dt = parse_z_time(state.next_books_wake_utc)

    # Bootstrap: on first run, fetch books immediately once (ignore wake schedule)
    if not state.did_bootstrap_books:
        state.did_bootstrap_books = True
    else:
        if now < wake_dt:
            return

    # expected next time for gap detection
    expected_next_dt = None
    if state.last_saved_book_time:
        try:
            expected_next_dt = parse_oanda_time(state.last_saved_book_time) + timedelta(seconds=BOOK_STEP_SECONDS)
        except Exception:
            expected_next_dt = None

    deadline = now + timedelta(seconds=BOOK_GRACE_SECONDS)
    catchup_deadline = now + timedelta(seconds=BOOK_CATCHUP_MAX_SECONDS)
    attempts = 0

    while datetime.now(timezone.utc) <= max(deadline, catchup_deadline):
        attempts += 1
        try:
            ob_raw, pb_raw = fetch_books()
            t_ob, p_ob = get_book_time_and_price(ob_raw, "orderBook")
            t_pb, p_pb = get_book_time_and_price(pb_raw, "positionBook")

            if not t_ob:
                time.sleep(BOOK_RETRY_EVERY_SECONDS)
                continue

            if t_ob == state.last_saved_book_time:
                time.sleep(BOOK_RETRY_EVERY_SECONDS)
                continue

            # Gap detection (metadata only)
            if expected_next_dt is not None:
                got_dt = parse_oanda_time(t_ob)
                if got_dt > (expected_next_dt + timedelta(seconds=BOOK_STEP_SECONDS)):
                    append_trade_log({
                        "event": "BOOKS_GAP_DETECTED",
                        "last_saved": state.last_saved_book_time,
                        "expected_next": fmt_z_time(expected_next_dt),
                        "got": t_ob,
                        "delta_sec": int((got_dt - expected_next_dt).total_seconds()),
                        "attempts": attempts,
                    })

            # Trim before saving
            ob = dict(ob_raw)
            pb = dict(pb_raw)
            ob["orderBook"] = trim_book_buckets(book=ob_raw.get("orderBook", {}), window=BOOK_WINDOW_AROUND_PRICE)
            pb["positionBook"] = trim_book_buckets(book=pb_raw.get("positionBook", {}), window=BOOK_WINDOW_AROUND_PRICE)

            append_jsonl(order_path, ob)
            append_jsonl(pos_path, pb)
            state.last_saved_book_time = t_ob

            walls_row = compute_support_resistance_from_orderbook(
                ob=ob_raw,  # use full raw for wall selection within ±BOOK_RANGE_DOLLARS (not the trimmed one)
                range_dollars=BOOK_RANGE_DOLLARS,
                total_min=WALL_TOTAL_MIN,
                imb_min=WALL_IMB_MIN,
            )
            append_walls_csv(walls_csv, walls_row)

            append_trade_log({
                "event": "BOOKS_SAVED",
                "book_time": t_ob,
                "ref_price": p_ob,
                "support": walls_row.get("buy_wall_price"),
                "resistance": walls_row.get("sell_wall_price"),
                "support_strength": walls_row.get("buy_strength"),
                "resistance_strength": walls_row.get("sell_strength"),
                "trim_window": BOOK_WINDOW_AROUND_PRICE,
                "order_trim_kept_n": ob["orderBook"].get("trim_kept_n"),
                "pos_trim_kept_n": pb["positionBook"].get("trim_kept_n"),
                "attempts": attempts,
            })

            # schedule next wake
            try:
                last_dt = parse_oanda_time(state.last_saved_book_time)
                next_expected = last_dt + timedelta(seconds=BOOK_STEP_SECONDS)
                state.next_books_wake_utc = fmt_z_time(next_expected + timedelta(seconds=5))
            except Exception:
                state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=BOOK_STEP_SECONDS))

            candles_n = int(getattr(state, "candles_since_books", 0) or 0)

            # heartbeat fields (best-effort; never crash books logging)
            try:
                open_n = len(get_open_positions_on_symbol(MT5_SYMBOL))
            except Exception:
                open_n = -1

            print(
                f"[BOOKS] NEW {t_ob} ref={p_ob} "
                f"SUP={walls_row.get('buy_wall_price')} RES={walls_row.get('sell_wall_price')} \n"
                f"[CANDLES] CANDLES +{candles_n} newest={state.last_saved_candle_time} \n"
                f"[HB] open={open_n} last_book={state.last_saved_book_time} wall={state.last_wall_time}"
            )

            state.candles_since_books = 0
            state.last_books_print_time = t_ob

            return

        except Exception as e:
            append_trade_log({"event": "BOOKS_ERR", "err": str(e), "attempts": attempts})
            time.sleep(BOOK_RETRY_EVERY_SECONDS)

    # missed grace/catchup
    state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=10))
    print("[BOOKS] WARN: no new snapshot within grace")


# =============================================================================
# Candles logger (frequent poll + dedupe + gap detect)
# =============================================================================
def maybe_run_candles_logger(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    day = day_yyyymmdd(now)
    out_path = OUT_CAND_DIR / f"{INSTRUMENT}_{GRANULARITY}_{day}.jsonl"

    if state.last_saved_candle_time is None:
        last = read_last_jsonl_line(out_path)
        if last:
            state.last_saved_candle_time = last.get("time")

    if state.next_candle_poll_utc is None:
        state.next_candle_poll_utc = fmt_z_time(now)

    poll_dt = parse_z_time(state.next_candle_poll_utc)
    if now < poll_dt:
        return

    # poll window
    fetch_to = now.replace(second=0, microsecond=0)

    bootstrap_min = 180  # 3 hours of M1 candles on first run

    if (not state.did_bootstrap_candles) and (state.last_saved_candle_time is None):
        fetch_from = fetch_to - timedelta(minutes=max(CANDLE_LOOKBACK_MIN, bootstrap_min))
        did_bootstrap = True
    else:
        fetch_from = fetch_to - timedelta(minutes=CANDLE_LOOKBACK_MIN)
        did_bootstrap = False

    # pull and append new completed candles
    try:
        raw = fetch_candles(fmt_z_time(fetch_from), fmt_z_time(fetch_to))
        candles = [c for c in normalize_candles(raw) if c.get("complete")]

        if not candles:
            state.next_candle_poll_utc = fmt_z_time(now + timedelta(seconds=CANDLE_POLL_SECONDS))
            return

        # read last_dt
        last_dt = parse_z_time(state.last_saved_candle_time) if state.last_saved_candle_time else None

        new = []
        for c in candles:
            try:
                c_dt = parse_z_time(c["time"])
            except Exception:
                continue
            if (last_dt is None) or (c_dt > last_dt):
                new.append(c)

        if not new:
            state.next_candle_poll_utc = fmt_z_time(now + timedelta(seconds=CANDLE_POLL_SECONDS))
            return

        # gap detect (metadata only)
        if last_dt is not None:
            newest_dt = parse_z_time(new[-1]["time"])
            delta_min = int((newest_dt - last_dt).total_seconds() / 60)
            if delta_min > CANDLE_GAP_WARN_MIN:
                append_trade_log({
                    "event": "CANDLES_GAP_DETECTED",
                    "last_saved": state.last_saved_candle_time,
                    "newest": new[-1]["time"],
                    "delta_min": delta_min,
                    "n_new": len(new),
                })

        for c in new:
            append_jsonl(out_path, c)

        state.last_saved_candle_time = new[-1]["time"]

        if did_bootstrap:
            state.did_bootstrap_candles = True

        append_trade_log({
            "event": "CANDLES_SAVED",
            "n_new": len(new),
            "newest": state.last_saved_candle_time,
            "file": str(out_path),
        })

        if did_bootstrap:
            append_trade_log({
                "event": "CANDLES_BOOTSTRAP",
                "from": fmt_z_time(fetch_from),
                "to": fmt_z_time(fetch_to),
                "n_new": len(new),
            })

        state.candles_since_books += len(new)


    except Exception as e:
        append_trade_log({"event": "CANDLES_ERR", "err": str(e)})

    state.next_candle_poll_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=CANDLE_POLL_SECONDS))


# =============================================================================
# Load candles for EMA/ATR
# =============================================================================
def load_recent_candles_multi(days: List[str], keep_last: int = 4000) -> pd.DataFrame:
    rows = []
    for d in days:
        path = OUT_CAND_DIR / f"{INSTRUMENT}_{GRANULARITY}_{d}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not obj.get("complete", True):
                    continue
                rows.append({
                    "time": obj["time"],
                    "dt": parse_z_time(obj["time"]),
                    "o": safe_float(obj.get("o")),
                    "h": safe_float(obj.get("h")),
                    "l": safe_float(obj.get("l")),
                    "c": safe_float(obj.get("c")),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
    if keep_last and len(df) > keep_last:
        df = df.iloc[-keep_last:].reset_index(drop=True)

    df["ema"] = compute_ema(df["c"], EMA_SPAN)
    df["atr"] = compute_atr(df, ATR_TRAIL_PERIOD)
    return df


# =============================================================================
# Strategy runner (breakout + bounce) + time exits + trailing
# =============================================================================
def maybe_process_new_candles_and_trade(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    today = day_yyyymmdd(now)
    yday = day_yyyymmdd(now - timedelta(days=1))

    candles = load_recent_candles_multi([yday, today], keep_last=5000)
    if candles.empty:
        return
    if len(candles) < max(EMA_SPAN + 10, ATR_TRAIL_PERIOD + 10, 50):
        return

    if state.last_processed_candle_time is None:
        # warm start: process from last_saved_candle_time forward (no huge replay by default)
        if state.last_saved_candle_time:
            state.last_processed_candle_time = state.last_saved_candle_time
        else:
            state.last_processed_candle_time = str(candles.iloc[-1]["time"])

    new_rows = candles.loc[candles["time"] > state.last_processed_candle_time]
    if new_rows.empty:
        return

    if len(new_rows) > 3:
        append_trade_log({
            "event": "REPLAY_BATCH",
            "note": f"processing {len(new_rows)} new candles since {state.last_processed_candle_time}",
        })

    for _, row in new_rows.iterrows():
        candle_time = str(row["time"])
        candle_dt: datetime = row["dt"]
        o = float(row["o"]); h = float(row["h"]); l = float(row["l"]); c = float(row["c"])
        ema = float(row["ema"]) if not pd.isna(row["ema"]) else float("nan")
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else float("nan")

        # update processed marker immediately
        state.last_processed_candle_time = candle_time

        # reset per-candle entry counter when candle changes
        if state.last_entry_candle_time != candle_time:
            state.last_entry_candle_time = candle_time
            state.entries_this_candle = 0

        # locate walls file
        candle_day = candle_dt.strftime("%Y%m%d")
        walls_path = OUT_WALLS_DIR / f"walls_{candle_day}.csv"

        # fallback: if no walls row in today's file yet, also try yesterday
        walls = load_walls_csv(walls_path) if walls_path.exists() else pd.DataFrame()
        w = latest_wall_row(walls, candle_dt)
        if w is None:
            yday_path = OUT_WALLS_DIR / f"walls_{day_yyyymmdd(candle_dt - timedelta(days=1))}.csv"
            if yday_path.exists():
                w2 = latest_wall_row(load_walls_csv(yday_path), candle_dt)
                w = w2

        if w is None:
            append_trade_log({"event": "STRAT_SKIP_NO_WALL_ROW", "candle_time": candle_time, "candle_dt": fmt_z_time(candle_dt)})
            continue

        wall_time = str(w["time"])
        ref_price = float(w["ref_price"]) if pd.notna(w["ref_price"]) else None

        # Interpret:
        support = float(w["buy_wall_price"]) if pd.notna(w.get("buy_wall_price")) else None
        resistance = float(w["sell_wall_price"]) if pd.notna(w.get("sell_wall_price")) else None

        support_imb = float(w["buy_imbalance"]) if pd.notna(w.get("buy_imbalance")) else 0.0
        resist_imb  = float(w["sell_imbalance"]) if pd.notna(w.get("sell_imbalance")) else 0.0

        # regime reset
        if state.last_wall_time is None or wall_time != state.last_wall_time:
            state.retests_support = 0
            state.retests_resist  = 0
            state.last_wall_time  = wall_time

        state.tick_n += 1
        if state.tick_n % 20 == 0:
            append_trade_log({
                "event": "STRAT_TICK",
                "candle_time": candle_time,
                "close": c,
                "ema": ema,
                "atr": atr,
                "ref_price": ref_price,
                "support": support,
                "resistance": resistance,
                "retests_support": state.retests_support,
                "retests_resist": state.retests_resist,
                "wall_time": wall_time,
            })

        # ---- manage positions: meta, time exits, trailing ----
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)
        open_tickets = {str(getattr(p, "ticket", "")) for p in open_positions if getattr(p, "ticket", None) is not None}

        # purge stale meta
        for t in list(state.pos_meta.keys()):
            if t not in open_tickets:
                state.pos_meta.pop(t, None)

        # ensure meta for open positions
        for p in open_positions:
            ticket = str(getattr(p, "ticket", ""))
            if not ticket:
                continue
            if ticket not in state.pos_meta:
                opened_epoch = getattr(p, "time", None)
                opened_dt = datetime.fromtimestamp(int(opened_epoch), tz=timezone.utc) if opened_epoch else datetime.now(timezone.utc)
                side = "long" if p.type == mt5.POSITION_TYPE_BUY else "short"
                entry_px = float(getattr(p, "price_open", 0.0))
                sl_px = float(getattr(p, "sl", 0.0))
                risk_per_unit = abs(entry_px - sl_px) if (entry_px > 0 and sl_px > 0) else 0.0

                state.pos_meta[ticket] = {
                    "opened_time_utc": fmt_z_time(opened_dt),
                    "side": side,
                    "entry": entry_px,
                    "initial_sl": sl_px,
                    "risk_per_unit": risk_per_unit,
                    "max_exit_candle_time": fmt_z_time(opened_dt + timedelta(minutes=MAX_HOLD_MINUTES)),

                    # ---- trailing state (restart-safe) ----
                    "be_done": (
                        (sl_px >= (entry_px + BE_BUFFER)) if side == "long"
                        else (sl_px <= (entry_px - BE_BUFFER))
                    ) if (entry_px > 0 and sl_px > 0) else False,

                    "last_sl": sl_px if sl_px > 0 else 0.0,  # last SL we believe is set
                    "last_trail_time": None,  # last update time (UTC string)
                    "last_trail_reason": None,  # "BE" or "ATR"
                }

        # time exits
        for p in list(open_positions):
            ticket = str(getattr(p, "ticket", ""))
            meta = state.pos_meta.get(ticket, {})
            mx = meta.get("max_exit_candle_time")
            if not mx:
                continue
            try:
                max_exit_dt = parse_z_time(mx)
            except Exception:
                max_exit_dt = candle_dt + timedelta(minutes=MAX_HOLD_MINUTES)

            if candle_dt >= max_exit_dt:
                ok, res = close_position_market(p, "max_hold")
                append_trade_log({
                    "event": "TIME_EXIT",
                    "candle_time": candle_time,
                    "ticket": ticket,
                    "ok": ok,
                    "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                })

        # refresh after closes
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)

        # trailing (tick-based; staged: BE first, then ATR; monotonic only)
        if ENABLE_TRAILING and open_positions and (not np.isnan(atr)) and atr > 0:
            tick = mt5.symbol_info_tick(MT5_SYMBOL)
            if tick is not None:
                for p in list(open_positions):
                    ticket = str(getattr(p, "ticket", ""))
                    meta = state.pos_meta.get(ticket, {})
                    side = meta.get("side")
                    entry = float(meta.get("entry", 0.0))
                    risk_per_unit = float(meta.get("risk_per_unit", 0.0))
                    if side not in ("long", "short"):
                        continue
                    if entry <= 0 or risk_per_unit <= 0:
                        continue

                    # Live price: long uses bid for liquidation value; short uses ask
                    price_now = float(tick.bid) if side == "long" else float(tick.ask)
                    pnl_per_unit = (price_now - entry) if side == "long" else (entry - price_now)
                    r_mult = pnl_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

                    if r_mult < TRAIL_START_R:
                        continue

                    cur_sl = float(getattr(p, "sl", 0.0))
                    cur_tp = float(getattr(p, "tp", 0.0))

                    # Meta trailing fields (restart-safe)
                    be_done = bool(meta.get("be_done", False))
                    last_sl_meta = float(meta.get("last_sl", 0.0) or 0.0)

                    # --- compute BE SL (first action once eligible) ---
                    be_sl = (entry + BE_BUFFER) if side == "long" else (entry - BE_BUFFER)

                    # --- compute ATR trail SL (only after BE is done) ---
                    atr_sl = (price_now - ATR_TRAIL_MULT * atr) if side == "long" else (
                                price_now + ATR_TRAIL_MULT * atr)

                    # --- choose candidate SL (STAGED) ---
                    reason = None
                    new_sl = None

                    if not be_done:
                        # First action must be BE + buffer (monotonic tightening)
                        if side == "long":
                            # take the tightest among current + meta + BE
                            base = cur_sl if cur_sl > 0 else -1e18
                            base = max(base, last_sl_meta if last_sl_meta > 0 else -1e18)
                            candidate = max(base, be_sl)
                            if candidate < price_now:  # must be below live price for long
                                new_sl = candidate
                                reason = "BE"
                        else:
                            base = cur_sl if cur_sl > 0 else 1e18
                            base = min(base, last_sl_meta if last_sl_meta > 0 else 1e18)
                            candidate = min(base, be_sl)
                            if candidate > price_now:  # must be above live price for short
                                new_sl = candidate
                                reason = "BE"
                    else:
                        # After BE is done: allow ATR trailing, still monotonic
                        if side == "long":
                            base = cur_sl if cur_sl > 0 else -1e18
                            base = max(base, last_sl_meta if last_sl_meta > 0 else -1e18)
                            candidate = max(base, atr_sl)
                            if candidate < price_now:
                                new_sl = candidate
                                reason = "ATR"
                        else:
                            base = cur_sl if cur_sl > 0 else 1e18
                            base = min(base, last_sl_meta if last_sl_meta > 0 else 1e18)
                            candidate = min(base, atr_sl)
                            if candidate > price_now:
                                new_sl = candidate
                                reason = "ATR"

                    if new_sl is None:
                        continue

                    new_sl = norm_price(new_sl)

                    # Ignore micro changes (avoid noise / spam)
                    if cur_sl > 0 and abs(new_sl - cur_sl) < (SYMBOL_INFO.point * 2):
                        continue

                    ok, res = modify_position_sl_tp(p, new_sl, cur_tp)

                    # Only update meta when broker accepted the change
                    if ok:
                        meta["last_sl"] = float(new_sl)
                        meta["last_trail_time"] = fmt_z_time(datetime.now(timezone.utc))
                        meta["last_trail_reason"] = reason
                        if reason == "BE":
                            meta["be_done"] = True  # BE step completed; next updates may be ATR
                        state.pos_meta[ticket] = meta

                    append_trade_log({
                        "event": "TRAIL_UPDATE",
                        "candle_time": candle_time,
                        "ticket": ticket,
                        "side": side,
                        "trail_reason": reason,
                        "be_done": bool(meta.get("be_done", False)),
                        "r_mult": r_mult,
                        "atr": atr,
                        "entry": entry,
                        "price_now": price_now,
                        "old_sl": cur_sl,
                        "new_sl": new_sl,
                        "tp": cur_tp,
                        "ok": ok,
                        "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                    })

        # ---- capacity check ----
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        # ---- retest counting (support/resistance) ----
        if support is not None and l <= support + TOUCH_DIST:
            state.retests_support += 1
        if resistance is not None and h >= resistance - TOUCH_DIST:
            state.retests_resist += 1

        if ref_price is None:
            continue

        def within_mwd(level: float) -> bool:
            return abs(ref_price - level) <= MAX_WALL_DISTANCE

        # ---- strict one-entry-per-candle (global) ----
        def can_enter_this_candle() -> bool:
            if state.last_entry_candle_time == candle_time and state.entries_this_candle >= MAX_ENTRIES_PER_CANDLE:
                return False
            if state.entries_this_candle >= MAX_ENTRIES_PER_CANDLE:
                return False
            if len(get_open_positions_on_symbol(MT5_SYMBOL)) >= MAX_OPEN_POSITIONS:
                return False
            return True

        def mark_entered() -> None:
            state.last_entry_candle_time = candle_time
            state.entries_this_candle += 1

        # ---- helpers for logging checks (compact, not noisy) ----
        def log_eval(event: str, **extra):
            append_trade_log({
                "event": event,
                "candle_time": candle_time,
                "wall_time": wall_time,
                "close": c,
                "ema": ema,
                "atr": atr,
                "ref_price": ref_price,
                "support": support,
                "resistance": resistance,
                "retests_support": state.retests_support,
                "retests_resist": state.retests_resist,
                **extra
            })

        # ==========================================================
        # A) BREAKOUT MODE
        # ==========================================================
        # Short breakout below support
        if ALLOW_SHORTS and support is not None:
            mwd_ok = within_mwd(support)
            ret_ok = state.retests_support >= RETESTS_REQUIRED
            broken = (c <= support - BREAK_BUFFER)
            trend_ok = (np.isnan(ema)) or ((c < ema) and ((ema - c) >= MIN_EMA_DIST))

            if (ret_ok or broken) and mwd_ok:
                log_eval("BRK_SHORT_CHECK", mwd_ok=mwd_ok, ret_ok=ret_ok, broken=broken, trend_ok=trend_ok)

            if mwd_ok and ret_ok and broken and trend_ok and can_enter_this_candle():
                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                entry = float(tick.bid) if tick else c
                entry = norm_price(entry)
                sl = norm_price(support + STOP_BUFFER)
                risk = sl - entry
                if risk > 0:
                    tp = norm_price(entry - TP_R * risk)
                    vol = calc_volume_for_cash_risk(MT5_SYMBOL, "short", entry, sl, RISK_CASH)

                    append_trade_log({
                        "event": "SIGNAL_SHORT_MKT",
                        "candle_time": candle_time,
                        "mode": "breakout",
                        "entry_est": entry,
                        "sl": sl,
                        "tp": tp,
                        "volume": vol,
                        "wall_level": support,
                        "wall_imb": support_imb,
                    })

                    if not SIGNALS_ONLY and vol > 0:
                        ok, res = send_market_order(MT5_SYMBOL, "short", vol, sl, tp, comment=f"WBv4 BRK S {ASSET_TAG}")
                        append_trade_log({
                            "event": "ORDER_SHORT_MKT",
                            "candle_time": candle_time,
                            "ok": ok,
                            "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                            "vol": vol,
                        })
                        if ok:
                            mark_entered()
                            state.retests_support = 0
                continue  # one decision per candle loop

        # Long breakout above resistance
        if ALLOW_LONGS and resistance is not None:
            mwd_ok = within_mwd(resistance)
            ret_ok = state.retests_resist >= RETESTS_REQUIRED
            broken = (c >= resistance + BREAK_BUFFER)
            trend_ok = (np.isnan(ema)) or ((c > ema) and ((c - ema) >= MIN_EMA_DIST))

            if (ret_ok or broken) and mwd_ok:
                log_eval("BRK_LONG_CHECK", mwd_ok=mwd_ok, ret_ok=ret_ok, broken=broken, trend_ok=trend_ok)

            if mwd_ok and ret_ok and broken and trend_ok and can_enter_this_candle():
                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                entry = float(tick.ask) if tick else c
                entry = norm_price(entry)
                sl = norm_price(resistance - STOP_BUFFER)
                risk = entry - sl
                if risk > 0:
                    tp = norm_price(entry + TP_R * risk)
                    vol = calc_volume_for_cash_risk(MT5_SYMBOL, "long", entry, sl, RISK_CASH)

                    append_trade_log({
                        "event": "SIGNAL_LONG_MKT",
                        "candle_time": candle_time,
                        "mode": "breakout",
                        "entry_est": entry,
                        "sl": sl,
                        "tp": tp,
                        "volume": vol,
                        "wall_level": resistance,
                        "wall_imb": resist_imb,
                    })

                    if not SIGNALS_ONLY and vol > 0:
                        ok, res = send_market_order(MT5_SYMBOL, "long", vol, sl, tp, comment=f"WBv4 BRK L {ASSET_TAG}")
                        append_trade_log({
                            "event": "ORDER_LONG_MKT",
                            "candle_time": candle_time,
                            "ok": ok,
                            "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                            "vol": vol,
                        })
                        if ok:
                            mark_entered()
                            state.retests_resist = 0
                continue

        # ==========================================================
        # B) BOUNCE MODE (optional)
        # ==========================================================
        if ENABLE_BOUNCE and can_enter_this_candle():
            # bounce long at support
            if ALLOW_LONGS and support is not None:
                touched = (l <= support + TOUCH_DIST)
                rejected = (c >= support + BOUNCE_BUFFER)
                ema_ok = (np.isnan(ema)) or (abs(c - ema) <= BOUNCE_MAX_EMA_DIST)

                # orderflow sanity (light-touch): prefer bounce long when support imbalance is positive
                flow_ok = (support_imb >= 0.0)

                if touched:
                    log_eval("BNC_LONG_CHECK", touched=touched, rejected=rejected, ema_ok=ema_ok, flow_ok=flow_ok)

                if touched and rejected and ema_ok and flow_ok:
                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
                    entry = float(tick.ask) if tick else c
                    entry = norm_price(entry)
                    sl = norm_price(support - STOP_BUFFER)
                    risk = entry - sl
                    if risk > 0:
                        tp = norm_price(entry + TP_R_BOUNCE * risk)
                        vol = calc_volume_for_cash_risk(MT5_SYMBOL, "long", entry, sl, RISK_CASH)

                        append_trade_log({
                            "event": "SIGNAL_LONG_MKT",
                            "candle_time": candle_time,
                            "mode": "bounce",
                            "entry_est": entry,
                            "sl": sl,
                            "tp": tp,
                            "volume": vol,
                            "wall_level": support,
                            "wall_imb": support_imb,
                        })

                        if not SIGNALS_ONLY and vol > 0:
                            ok, res = send_market_order(MT5_SYMBOL, "long", vol, sl, tp, comment=f"WBv4 BNC L {ASSET_TAG}")
                            append_trade_log({
                                "event": "ORDER_LONG_MKT",
                                "candle_time": candle_time,
                                "ok": ok,
                                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                                "vol": vol,
                            })
                            if ok:
                                mark_entered()
                    continue

            # bounce short at resistance
            if ALLOW_SHORTS and resistance is not None:
                touched = (h >= resistance - TOUCH_DIST)
                rejected = (c <= resistance - BOUNCE_BUFFER)
                ema_ok = (np.isnan(ema)) or (abs(c - ema) <= BOUNCE_MAX_EMA_DIST)

                # prefer bounce short when resistance imbalance is negative
                flow_ok = (resist_imb <= 0.0)

                if touched:
                    log_eval("BNC_SHORT_CHECK", touched=touched, rejected=rejected, ema_ok=ema_ok, flow_ok=flow_ok)

                if touched and rejected and ema_ok and flow_ok:
                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
                    entry = float(tick.bid) if tick else c
                    entry = norm_price(entry)
                    sl = norm_price(resistance + STOP_BUFFER)
                    risk = sl - entry
                    if risk > 0:
                        tp = norm_price(entry - TP_R_BOUNCE * risk)
                        vol = calc_volume_for_cash_risk(MT5_SYMBOL, "short", entry, sl, RISK_CASH)

                        append_trade_log({
                            "event": "SIGNAL_SHORT_MKT",
                            "candle_time": candle_time,
                            "mode": "bounce",
                            "entry_est": entry,
                            "sl": sl,
                            "tp": tp,
                            "volume": vol,
                            "wall_level": resistance,
                            "wall_imb": resist_imb,
                        })

                        if not SIGNALS_ONLY and vol > 0:
                            ok, res = send_market_order(MT5_SYMBOL, "short", vol, sl, tp, comment=f"WBv4 BNC S {ASSET_TAG}")
                            append_trade_log({
                                "event": "ORDER_SHORT_MKT",
                                "candle_time": candle_time,
                                "ok": ok,
                                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                                "vol": vol,
                            })
                            if ok:
                                mark_entered()
                    continue


# =============================================================================
# State <-> dict helpers (restart-safe)
# =============================================================================
def botstate_to_dict(state: BotState) -> Dict[str, Any]:
    d = asdict(state)
    d["schema_version"] = 4
    d["saved_dt_utc"] = fmt_z_time(datetime.now(timezone.utc))
    return d

def dict_to_botstate(d: Dict[str, Any]) -> BotState:
    st = BotState()
    apply_state_to_botstate(st, d)
    # ensure dict fields exist
    if st.pos_meta is None:
        st.pos_meta = {}
    return st

def restore_or_create_state() -> BotState:
    data = load_state()
    if data:
        st = dict_to_botstate(data)
        append_trade_log({
            "event": "STATE_RESTORED",
            "state_file": str(STATE_JSON),
            "last_saved_book_time": st.last_saved_book_time,
            "last_saved_candle_time": st.last_saved_candle_time,
            "last_processed_candle_time": st.last_processed_candle_time,
            "last_wall_time": st.last_wall_time,
            "last_entry_candle_time": st.last_entry_candle_time,
            "entries_this_candle": st.entries_this_candle,
            "pos_meta_n": len(st.pos_meta or {}),
        })
        return st

    st = BotState(instrument=INSTRUMENT)
    append_trade_log({
        "event": "STATE_NEW",
        "state_file": str(STATE_JSON),
        "instrument": INSTRUMENT,
        "symbol": MT5_SYMBOL,
        "magic": MAGIC,
    })
    save_state(botstate_to_dict(st))
    return st


# =============================================================================
# Main loop
# =============================================================================
def main():
    if "PASTE" in OANDA_TOKEN:
        raise RuntimeError("Set OANDA_TOKEN at the top of the script.")
    if "PASTE" in MT5_PASSWORD:
        raise RuntimeError("Set MT5_PASSWORD at the top of the script.")

    print("\n=== WBv4 EMA + TRAIL + MKT (Support/Resistance, Breakout+Bounce) ===")
    print(f"[ROOT]  {BASE_DIR}")
    print(f"[ASSET] {ASSET_TAG}")
    print(f"[STRAT] {STRATEGY_TAG}")
    print(f"[OANDA] api={OANDA_API_URL} instrument={INSTRUMENT} gran={GRANULARITY} price={PRICE_TYPE}")
    print(f"[MT5]   term={MT5_TERMINAL_PATH}")
    print(f"[MT5]   server={MT5_SERVER} symbol={MT5_SYMBOL} magic={MAGIC}")
    print(f"[RISK]  cash={RISK_CASH} max_hold={MAX_HOLD_MINUTES}m max_open={MAX_OPEN_POSITIONS} max_entries/candle={MAX_ENTRIES_PER_CANDLE}")
    print(f"[BOOKS] cadence={BOOK_STEP_SECONDS}s grace={BOOK_GRACE_SECONDS}s retry={BOOK_RETRY_EVERY_SECONDS}s trim=±{BOOK_WINDOW_AROUND_PRICE}")
    print(f"[WALLS] range=±{BOOK_RANGE_DOLLARS} total_min={WALL_TOTAL_MIN} imb_min={WALL_IMB_MIN} max_wall_dist={MAX_WALL_DISTANCE}")
    print(f"[BRK]   retests={RETESTS_REQUIRED} touch={TOUCH_DIST} break_buf={BREAK_BUFFER} stop_buf={STOP_BUFFER} tpR={TP_R}")
    print(f"[BNC]   enabled={ENABLE_BOUNCE} buf={BOUNCE_BUFFER} tpR={TP_R_BOUNCE} max_ema_dist={BOUNCE_MAX_EMA_DIST}")
    print(f"[TRAIL] enabled={ENABLE_TRAILING} startR={TRAIL_START_R} BE_buf={BE_BUFFER} ATR(p={ATR_TRAIL_PERIOD}, mult={ATR_TRAIL_MULT})")
    print("--------------------------------------------------------------")

    state = restore_or_create_state()

    # MT5 connect
    init_mt5()

    # --------------------------------------------------------------
    # BOOTSTRAP ON START (one-shot)
    # Forces an immediate books + candles run so you see status quickly.
    # After this, normal cadence scheduling applies.
    # --------------------------------------------------------------
    try:
        maybe_run_books_logger(state)
        maybe_run_candles_logger(state)
        fetch_and_print_books_status(state)      # prints books + candles + HB immediately
        maybe_process_new_candles_and_trade(state)
        save_state(botstate_to_dict(state))
    except Exception as e:
        append_trade_log({"event": "BOOTSTRAP_ERR", "err": str(e)})

    # Persist state periodically (not noisy)
    last_state_save = time.time()
    STATE_SAVE_EVERY_SEC = 5.0

    try:
        while True:
            # 1) Books (20m cadence, dedupe, gap-detect, trimmed save, walls csv)
            maybe_run_books_logger(state)

            # 2) Candles (frequent poll, dedupe, gap-detect)
            maybe_run_candles_logger(state)

            # 3) Strategy (process new candles; handles time exits + trailing + entries)
            maybe_process_new_candles_and_trade(state)

            # periodic state save (survives restarts)
            if (time.time() - last_state_save) >= STATE_SAVE_EVERY_SEC:
                save_state(botstate_to_dict(state))
                last_state_save = time.time()

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        append_trade_log({"event": "STOP", "reason": "KeyboardInterrupt"})
        save_state(botstate_to_dict(state))
        print("\n[STOP] KeyboardInterrupt")

    except Exception as e:
        append_trade_log({"event": "FATAL", "err": str(e)})
        save_state(botstate_to_dict(state))
        raise

    finally:
        try:
            shutdown_mt5()
        except Exception:
            pass


if __name__ == "__main__":
    main()