#!/usr/bin/env python
"""
bot_wallBO_trail_limits-XAU.py
======================================

v3 = Script A parity + Script B trailing + LIMIT orders (books-driven)

DATA LOGGING (Script A parity)
- OANDA orderBook + positionBook snapshots -> JSONL (append-only, dedupe by orderBook.time)
- OANDA candles M1 (mid) -> JSONL (append-only, dedupe by candle time)
- Walls extraction -> CSV (append-only) using SAME compute_best_walls_from_orderbook_snapshot()
- Strategy uses wall regime selection: latest wall row where wall_dt <= candle_dt

STRATEGY MODES
- Mode A: Market breakout (same as Script A)
- Mode B: Limit orders at walls (both sides), refreshed each books snapshot:
  * cancel old pending limit orders
  * place BUY LIMIT near buy_wall & SELL LIMIT near sell_wall
  * SL/TP placed immediately
  * trailing begins only after +1R as per trailing config

RISK CONTROLS
- MAX_OPEN_POSITIONS (open positions + pending orders) cap
- Market entries: MAX_ENTRIES_PER_CANDLE=1 and hard block by candle_time
- Time exit after MAX_HOLD_MINUTES (checked on candle close)
- Trailing: BE @ +1R then ATR trail

PORTABLE PATHS
- Saves everything under script folder, grouped by ASSET_TAG and STRATEGY_TAG
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import requests
import MetaTrader5 as mt5
from pytz import timezone as pytz_timezone

# =========================
# MT5 ACCOUNT CONFIG
# =========================
MT5_LOGIN         = 52699804
MT5_PASSWORD      = "agWVZ&0YssnIQF"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\ALGO-OrderBook_XAU\terminal64.exe"

LOCAL_TZ   = pytz_timezone("Europe/London")
MT5_SYMBOL = "XAUUSD"   # adjust if broker uses suffix
SYMBOL_INFO = None      # filled by init_mt5()

# =========================
# OANDA CONFIG
# =========================
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"  # practice
OANDA_TOKEN   = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
INSTRUMENT    = "XAU_USD"

# =========================
# BOOKS LOGGER CONFIG
# =========================
BOOK_STEP_SECONDS        = 20 * 60
BOOK_GRACE_SECONDS       = 180
BOOK_RETRY_EVERY_SECONDS = 10

BOOK_RANGE_DOLLARS = 25.0
WALL_TOTAL_MIN     = 0.08
WALL_IMB_MIN       = 0.06

# =========================
# CANDLES LOGGER CONFIG
# =========================
GRANULARITY           = "M1"
PRICE_TYPE            = "M"
CANDLE_STEP_SECONDS   = 20 * 60
CANDLE_WAKE_DELAY_SEC = 8
CANDLE_GRACE_SECONDS  = 60
CANDLE_RETRY_EVERY    = 10

# =========================
# STRATEGY CONFIG (breakout)
# =========================
EMA_SPAN          = 9
MIN_EMA_DIST      = 0.5
RETESTS_REQUIRED  = 6
TOUCH_DIST        = 0.75
BREAK_BUFFER      = 0.2
STOP_BUFFER       = 1.0
TP_R              = 2.5
MAX_WALL_DISTANCE = 12.0

# =========================
# LIMIT ORDERS CONFIG (new)
# =========================
ENABLE_LIMITS            = True
PLACE_BOTH_SIDES_LIMITS  = True
LIMIT_ENTRY_OFFSET       = 0.00  # shift away from wall if needed (e.g. +0.10 long, -0.10 short)
LIMIT_MAX_DIST_FROM_MKT  = 15.0  # avoid placing limits too far from current MT5 price

# =========================
# TRAILING STOP (BE + ATR)
# =========================
ENABLE_TRAILING   = True

TRAIL_START_R     = 1.0     # BE + trailing logic starts once profit >= 1R
BE_BUFFER         = 0.50    # XAU units (0.20 = 20 cents)

ATR_TRAIL_PERIOD  = 20
ATR_TRAIL_MULT    = 2.5     # after BE active, trail = price -/+ ATR_MULT * ATR

# =========================
# EXECUTION / RISK
# =========================
RISK_CASH        = 500.0
MAX_HOLD_MINUTES = 180

ALLOW_LONGS      = True
ALLOW_SHORTS     = True

MAX_OPEN_POSITIONS      = 3
MAX_ENTRIES_PER_CANDLE  = 1

SIGNALS_ONLY     = False

MAGIC            = 2209001
DEVIATION        = 15

LOOP_SLEEP_SECONDS = 1.0

# =========================
# PORTABLE PATHS (asset + strategy namespace)
# =========================
BASE_DIR = Path(__file__).resolve().parent

ASSET_TAG = INSTRUMENT.strip().replace("/", "_").replace(":", "_")
STRATEGY_TAG = f"WBv3_EMA{EMA_SPAN}_TRAIL_LIMITS-{ASSET_TAG}"

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

# =========================
# Helpers
# =========================
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

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")

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

def append_walls_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

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

def load_walls_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["wall_dt"] = df["time"].apply(parse_z_time)
    for c in [
        "ref_price", "bucket_width",
        "buy_wall_price", "buy_strength", "buy_imbalance", "buy_long", "buy_short", "buy_total",
        "sell_wall_price", "sell_strength", "sell_imbalance", "sell_long", "sell_short", "sell_total",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("wall_dt").reset_index(drop=True)

def latest_wall_row(walls: pd.DataFrame, t: datetime) -> Optional[pd.Series]:
    if walls is None or walls.empty:
        return None
    idx = walls["wall_dt"].searchsorted(t, side="right") - 1
    if idx < 0 or idx >= len(walls):
        return None
    return walls.iloc[int(idx)]

def get_book_time_and_price(ob: dict) -> Tuple[Optional[str], Optional[float]]:
    book = ob.get("orderBook", {})
    t = book.get("time")
    p = book.get("price")
    return t, (safe_float(p) if p is not None else None)

def compute_best_walls_from_orderbook_snapshot(
    ob: dict,
    range_dollars: float,
    total_min: float,
    imb_min: float,
) -> dict:
    book = ob.get("orderBook", {})
    t = book.get("time")
    ref_price = safe_float(book.get("price"), default=float("nan"))
    bucket_width = safe_float(book.get("bucketWidth"), default=float("nan"))

    buckets = book.get("buckets", [])
    if not buckets or ref_price != ref_price:  # NaN check
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

    best_buy = None   # tuple(strength, total, rowdict)
    best_sell = None

    for b in buckets:
        p = safe_float(b.get("price"))
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
        if imb > 0:
            if (best_buy is None) or (key > (best_buy[0], best_buy[1])):
                best_buy = (row["strength"], row["total"], row)
        elif imb < 0:
            if (best_sell is None) or (key > (best_sell[0], best_sell[1])):
                best_sell = (row["strength"], row["total"], row)

    buy = None if best_buy is None else best_buy[2]
    sell = None if best_sell is None else best_sell[2]

    return {
        "time": t,
        "ref_price": ref_price,
        "bucket_width": bucket_width,

        "buy_wall_price": None if buy is None else buy["price"],
        "buy_strength": None if buy is None else buy["strength"],
        "buy_imbalance": None if buy is None else buy["imb"],
        "buy_long": None if buy is None else buy["long"],
        "buy_short": None if buy is None else buy["short"],
        "buy_total": None if buy is None else buy["total"],

        "sell_wall_price": None if sell is None else sell["price"],
        "sell_strength": None if sell is None else sell["strength"],
        "sell_imbalance": None if sell is None else sell["imb"],
        "sell_long": None if sell is None else sell["long"],
        "sell_short": None if sell is None else sell["short"],
        "sell_total": None if sell is None else sell["total"],
    }

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

def read_last_candle_time(path: Path) -> Optional[str]:
    last = read_last_jsonl_line(path)
    if not last:
        return None
    return last.get("time")

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    return atr

# =========================
# OANDA HTTP
# =========================
def oanda_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def fetch_json(endpoint: str, timeout: int = 45, params: Optional[dict] = None) -> dict:
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
    return fetch_json(f"/instruments/{INSTRUMENT}/candles", timeout=45, params=params)

# =========================
# Cadence helper
# =========================
def next_boundary(now: datetime, step_seconds: int) -> datetime:
    epoch = int(now.timestamp())
    next_epoch = ((epoch // step_seconds) + 1) * step_seconds
    return datetime.fromtimestamp(next_epoch, tz=timezone.utc).replace(second=0, microsecond=0)

# =========================
# MT5 helpers
# =========================
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

def get_pending_orders_on_symbol(symbol: str) -> List[Any]:
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return []
    return [o for o in orders if int(getattr(o, "magic", 0)) == int(MAGIC)]

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

def send_limit_order(symbol: str, side: str, volume: float, entry: float, sl: float, tp: float, comment: str):
    """
    BUY LIMIT / SELL LIMIT with SL/TP set immediately.
    """
    entry = norm_price(entry)
    sl = norm_price(sl)
    tp = norm_price(tp)

    if side == "long":
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    else:
        order_type = mt5.ORDER_TYPE_SELL_LIMIT

    # Pending orders typically work well with RETURN
    mode = mt5.ORDER_FILLING_RETURN

    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(entry),
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
        return False, f"order_send None: {mt5.last_error()}"
    ok = res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
    return ok, res

def cancel_order(ticket: int) -> Tuple[bool, Any]:
    req = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": int(ticket),
        "magic": int(MAGIC),
        "comment": "CANCEL_LIMIT"[:31],
    }
    res = mt5.order_send(req)
    if res is None:
        return False, f"order_send None: {mt5.last_error()}"
    ok = res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES)
    return ok, res

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

# =========================
# Live logging
# =========================
def append_trade_log(row: Dict[str, Any]) -> None:
    obj = dict(row)
    obj.setdefault("schema_version", 3)
    obj.setdefault("dt_utc", fmt_z_time(datetime.now(timezone.utc)))

    TRADES_JSONL.parent.mkdir(parents=True, exist_ok=True)
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

# =========================
# Bot State
# =========================
@dataclass
class BotState:
    instrument: str = ""

    # books
    last_saved_book_time: Optional[str] = None
    next_books_wake_utc: Optional[str] = None

    # candles
    last_saved_candle_time: Optional[str] = None
    next_candle_wake_utc: Optional[str] = None

    # strategy processing
    last_processed_candle_time: Optional[str] = None
    last_wall_time: Optional[str] = None
    retests_buy: int = 0
    retests_sell: int = 0

    # market-entry guards
    last_entry_candle_time: Optional[str] = None

    # position tracking
    pos_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    tick_n: int = 0

# =========================
# LIMIT order refresh on new books
# =========================
def refresh_limit_orders_on_new_books(*, walls_row: dict, book_time: str) -> None:
    if not ENABLE_LIMITS:
        return

    # Cancel existing pending orders (expiry rule: refresh on new books)
    pending = get_pending_orders_on_symbol(MT5_SYMBOL)
    for o in pending:
        ok, res = cancel_order(int(o.ticket))
        append_trade_log({
            "event": "LIMIT_CANCEL",
            "book_time": book_time,
            "order_ticket": int(o.ticket),
            "ok": ok,
            "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
        })

    # Capacity after cancelling
    open_positions = get_open_positions_on_symbol(MT5_SYMBOL)
    pending_after = get_pending_orders_on_symbol(MT5_SYMBOL)
    cap_left = MAX_OPEN_POSITIONS - (len(open_positions) + len(pending_after))

    if cap_left <= 0:
        append_trade_log({
            "event": "LIMIT_SKIP_CAP_FULL",
            "book_time": book_time,
            "open_n": len(open_positions),
            "pending_n": len(pending_after),
            "max_open": MAX_OPEN_POSITIONS,
        })
        return

    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None:
        append_trade_log({"event": "LIMIT_SKIP_NO_TICK", "book_time": book_time})
        return

    bid = float(tick.bid)
    ask = float(tick.ask)
    mid = (bid + ask) / 2.0

    ref_price = safe_float(walls_row.get("ref_price"), default=float("nan"))
    buy_wall  = walls_row.get("buy_wall_price")
    sell_wall = walls_row.get("sell_wall_price")

    buy_strength  = safe_float(walls_row.get("buy_strength"), default=0.0)
    sell_strength = safe_float(walls_row.get("sell_strength"), default=0.0)

    def within_mwd(level: float) -> bool:
        if ref_price != ref_price:
            return False
        return abs(ref_price - level) <= MAX_WALL_DISTANCE

    def within_market_dist(level: float) -> bool:
        return abs(mid - level) <= LIMIT_MAX_DIST_FROM_MKT

    # Decide placement order: stronger first (so if cap_left=1 we place best)
    candidates = []
    if ALLOW_LONGS and buy_wall is not None:
        try:
            bw = float(buy_wall)
            candidates.append(("long", bw, buy_strength))
        except Exception:
            pass
    if ALLOW_SHORTS and sell_wall is not None:
        try:
            sw = float(sell_wall)
            candidates.append(("short", sw, sell_strength))
        except Exception:
            pass

    # Sort by strength desc
    candidates.sort(key=lambda x: float(x[2]), reverse=True)

    placed = 0
    for side, wall_px, strength in candidates:
        if not PLACE_BOTH_SIDES_LIMITS and placed >= 1:
            break
        if cap_left <= 0:
            break

        if not within_market_dist(wall_px):
            append_trade_log({
                "event": "LIMIT_SKIP_TOO_FAR_FROM_MKT",
                "book_time": book_time,
                "side": side,
                "wall": wall_px,
                "mid": mid,
                "max_dist": LIMIT_MAX_DIST_FROM_MKT,
            })
            continue

        if not within_mwd(wall_px):
            append_trade_log({
                "event": "LIMIT_SKIP_MWD",
                "book_time": book_time,
                "side": side,
                "wall": wall_px,
                "ref_price": ref_price if ref_price == ref_price else None,
                "max_wall_distance": MAX_WALL_DISTANCE,
            })
            continue

        # Build limit entry price
        if side == "long":
            entry = wall_px + float(LIMIT_ENTRY_OFFSET)
            # Valid BUY LIMIT must be <= current ask
            if entry > ask:
                append_trade_log({
                    "event": "LIMIT_SKIP_INVALID_PRICE",
                    "book_time": book_time,
                    "side": side,
                    "entry": entry,
                    "ask": ask,
                    "note": "BUY_LIMIT entry must be <= ask",
                })
                continue

            sl = (wall_px - STOP_BUFFER)
            risk_per_unit = entry - sl
            if risk_per_unit <= 0:
                continue
            tp = entry + TP_R * risk_per_unit
            vol = calc_volume_for_cash_risk(MT5_SYMBOL, "long", entry, sl, RISK_CASH)

            if vol <= 0:
                continue

            ok, res = send_limit_order(
                MT5_SYMBOL, "long", vol,
                entry=entry, sl=sl, tp=tp,
                comment=f"WBv3 LIM L {ASSET_TAG}"
            )

            append_trade_log({
                "event": "LIMIT_PLACE",
                "book_time": book_time,
                "side": "long",
                "wall": wall_px,
                "strength": strength,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "vol": vol,
                "ok": ok,
                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
            })

        else:
            entry = wall_px - float(LIMIT_ENTRY_OFFSET)
            # Valid SELL LIMIT must be >= current bid
            if entry < bid:
                append_trade_log({
                    "event": "LIMIT_SKIP_INVALID_PRICE",
                    "book_time": book_time,
                    "side": side,
                    "entry": entry,
                    "bid": bid,
                    "note": "SELL_LIMIT entry must be >= bid",
                })
                continue

            sl = (wall_px + STOP_BUFFER)
            risk_per_unit = sl - entry
            if risk_per_unit <= 0:
                continue
            tp = entry - TP_R * risk_per_unit
            vol = calc_volume_for_cash_risk(MT5_SYMBOL, "short", entry, sl, RISK_CASH)

            if vol <= 0:
                continue

            ok, res = send_limit_order(
                MT5_SYMBOL, "short", vol,
                entry=entry, sl=sl, tp=tp,
                comment=f"WBv3 LIM S {ASSET_TAG}"
            )

            append_trade_log({
                "event": "LIMIT_PLACE",
                "book_time": book_time,
                "side": "short",
                "wall": wall_px,
                "strength": strength,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "vol": vol,
                "ok": ok,
                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
            })

        # Update cap based on latest pending
        pending_after = get_pending_orders_on_symbol(MT5_SYMBOL)
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)
        cap_left = MAX_OPEN_POSITIONS - (len(open_positions) + len(pending_after))
        placed += 1

# =========================
# Books logger (JSONL + walls CSV) + limit refresh
# =========================
def maybe_run_books_logger(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    day = day_yyyymmdd(now)

    order_path = OUT_ORDER_DIR / f"orderbook_{day}.jsonl"
    pos_path   = OUT_POS_DIR   / f"positionbook_{day}.jsonl"
    walls_csv  = OUT_WALLS_DIR / f"walls_{day}.csv"

    if state.last_saved_book_time is None:
        last_obj = read_last_jsonl_line(order_path)
        if last_obj:
            t_last, _ = get_book_time_and_price(last_obj)
            if t_last:
                state.last_saved_book_time = t_last

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
    if now < wake_dt:
        return

    deadline = now + timedelta(seconds=BOOK_GRACE_SECONDS)
    attempts = 0

    while datetime.now(timezone.utc) <= deadline:
        attempts += 1
        try:
            ob, pb = fetch_books()
            t_ob, p_ob = get_book_time_and_price(ob)

            if not t_ob:
                time.sleep(BOOK_RETRY_EVERY_SECONDS)
                continue

            if t_ob == state.last_saved_book_time:
                time.sleep(BOOK_RETRY_EVERY_SECONDS)
                continue

            # NEW snapshot
            append_jsonl(order_path, ob)
            append_jsonl(pos_path, pb)
            state.last_saved_book_time = t_ob

            walls_row = compute_best_walls_from_orderbook_snapshot(
                ob=ob,
                range_dollars=BOOK_RANGE_DOLLARS,
                total_min=WALL_TOTAL_MIN,
                imb_min=WALL_IMB_MIN,
            )
            append_walls_csv(walls_csv, walls_row)

            append_trade_log({
                "event": "BOOKS_SAVED",
                "book_time": t_ob,
                "ref_price": p_ob,
                "buy_wall": walls_row.get("buy_wall_price"),
                "sell_wall": walls_row.get("sell_wall_price"),
                "attempts": attempts,
            })

            # LIMIT refresh is tied to NEW books snapshot
            refresh_limit_orders_on_new_books(walls_row=walls_row, book_time=t_ob)

            # schedule next wake
            try:
                last_dt = parse_oanda_time(state.last_saved_book_time)
                next_expected = last_dt + timedelta(seconds=BOOK_STEP_SECONDS)
                state.next_books_wake_utc = fmt_z_time(next_expected + timedelta(seconds=5))
            except Exception:
                state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=BOOK_STEP_SECONDS))

            print(f"[BOOKS] NEW {t_ob} ref={p_ob} BUY={walls_row.get('buy_wall_price')} SELL={walls_row.get('sell_wall_price')}")
            return

        except Exception as e:
            append_trade_log({"event": "BOOKS_ERR", "err": str(e)})
            time.sleep(BOOK_RETRY_EVERY_SECONDS)

    state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=10))
    print("[BOOKS] WARN: no new snapshot within grace")

# =========================
# Candles logger (JSONL)
# =========================
def maybe_run_candles_logger(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    day = day_yyyymmdd(now)
    out_path = OUT_CAND_DIR / f"{INSTRUMENT}_{GRANULARITY}_{day}.jsonl"

    if state.last_saved_candle_time is None:
        state.last_saved_candle_time = read_last_candle_time(out_path)

    if state.next_candle_wake_utc is None:
        boundary = next_boundary(now, CANDLE_STEP_SECONDS)
        wake = boundary + timedelta(seconds=CANDLE_WAKE_DELAY_SEC)
        state.next_candle_wake_utc = fmt_z_time(wake)

    wake_dt = parse_z_time(state.next_candle_wake_utc)
    if now < wake_dt:
        return

    last_dt = parse_z_time(state.last_saved_candle_time) if state.last_saved_candle_time else None

    fetch_to = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    fetch_from = (last_dt - timedelta(minutes=1)) if last_dt else (fetch_to - timedelta(seconds=CANDLE_STEP_SECONDS))

    if fetch_to <= fetch_from:
        fetch_from = fetch_to - timedelta(minutes=25)

    deadline = datetime.now(timezone.utc) + timedelta(seconds=CANDLE_GRACE_SECONDS)
    attempts = 0

    while datetime.now(timezone.utc) <= deadline:
        attempts += 1
        try:
            raw = fetch_candles(fmt_z_time(fetch_from), fmt_z_time(fetch_to))
            candles = [c for c in normalize_candles(raw) if c.get("complete")]

            if not candles:
                time.sleep(CANDLE_RETRY_EVERY)
                continue

            new = []
            for c in candles:
                try:
                    c_dt = parse_z_time(c["time"])
                except Exception:
                    continue
                if last_dt is None or c_dt > last_dt:
                    new.append(c)

            if not new:
                time.sleep(CANDLE_RETRY_EVERY)
                continue

            for c in new:
                append_jsonl(out_path, c)

            state.last_saved_candle_time = new[-1]["time"]

            append_trade_log({
                "event": "CANDLES_SAVED",
                "n_new": len(new),
                "newest": state.last_saved_candle_time,
                "attempts": attempts,
            })

            boundary = next_boundary(datetime.now(timezone.utc), CANDLE_STEP_SECONDS)
            state.next_candle_wake_utc = fmt_z_time(boundary + timedelta(seconds=CANDLE_WAKE_DELAY_SEC))

            print(f"[CANDLES] +{len(new)} newest={state.last_saved_candle_time}")
            return

        except Exception as e:
            append_trade_log({"event": "CANDLES_ERR", "err": str(e)})
            time.sleep(CANDLE_RETRY_EVERY)

    state.next_candle_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=10))
    print("[CANDLES] WARN: no new candles within grace")

# =========================
# Load candles (EMA + ATR) across yday/today
# =========================
def load_recent_candles_for_ema(day: str) -> pd.DataFrame:
    path = OUT_CAND_DIR / f"{INSTRUMENT}_{GRANULARITY}_{day}.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows = []
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

    df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
    if len(df):
        df["ema"] = compute_ema(df["c"], EMA_SPAN)
        df["atr"] = compute_atr(df, ATR_TRAIL_PERIOD)
    else:
        df["ema"] = np.nan
        df["atr"] = np.nan
    return df

def load_recent_candles_for_ema_multi(days: List[str], keep_last: int = 3000) -> pd.DataFrame:
    dfs = []
    for d in days:
        df = load_recent_candles_for_ema(d)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True).sort_values("dt").reset_index(drop=True)
    if keep_last and len(out) > keep_last:
        out = out.iloc[-keep_last:].reset_index(drop=True)
    out["ema"] = compute_ema(out["c"], EMA_SPAN)
    out["atr"] = compute_atr(out, ATR_TRAIL_PERIOD)
    return out

# =========================
# Strategy runner (Script A parity for walls regime)
# =========================
def maybe_process_new_candles_and_trade(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    today = day_yyyymmdd(now)
    yday = day_yyyymmdd(now - timedelta(days=1))

    candles = load_recent_candles_for_ema_multi([yday, today], keep_last=3000)
    if candles.empty:
        return
    if len(candles) < max(EMA_SPAN + 5, ATR_TRAIL_PERIOD + 5, 25):
        return

    # determine new rows
    if state.last_processed_candle_time is None:
        if state.last_saved_candle_time:
            state.last_processed_candle_time = state.last_saved_candle_time
            new_rows = candles.loc[candles["time"] > state.last_processed_candle_time]
        else:
            new_rows = candles.iloc[-1:]
    else:
        new_rows = candles.loc[candles["time"] > state.last_processed_candle_time]

    if new_rows.empty:
        return

    append_trade_log({
        "event": "REPLAY_BATCH",
        "note": f"processing {len(new_rows)} new candles since {state.last_processed_candle_time}",
    })

    for _, row in new_rows.iterrows():
        candle_time = str(row["time"])
        candle_dt: datetime = row["dt"]
        h = float(row["h"]); l = float(row["l"]); c = float(row["c"])
        ema = float(row["ema"]) if not pd.isna(row["ema"]) else float("nan")
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else float("nan")

        state.last_processed_candle_time = candle_time

        # Load walls file for candle day (Script A parity)
        candle_day = candle_dt.strftime("%Y%m%d")
        walls_path = OUT_WALLS_DIR / f"walls_{candle_day}.csv"

        # fallback to yesterday's walls if early in day
        if not walls_path.exists():
            yday_str = day_yyyymmdd(candle_dt - timedelta(days=1))
            y_walls_path = OUT_WALLS_DIR / f"walls_{yday_str}.csv"
            if y_walls_path.exists():
                walls_path = y_walls_path
            else:
                append_trade_log({
                    "event": "STRAT_SKIP_NO_WALL_FILE",
                    "candle_time": candle_time,
                    "walls_file": str(walls_path),
                })
                continue

        walls = load_walls_csv(walls_path)
        w = latest_wall_row(walls, candle_dt)  # critical: wall_dt <= candle_dt
        if w is None:
            append_trade_log({
                "event": "STRAT_SKIP_NO_WALL_ROW",
                "candle_time": candle_time,
                "candle_dt": fmt_z_time(candle_dt),
            })
            continue

        wall_time = str(w["time"])
        ref_price = float(w["ref_price"]) if pd.notna(w["ref_price"]) else None
        buy_wall  = float(w["buy_wall_price"]) if pd.notna(w.get("buy_wall_price")) else None
        sell_wall = float(w["sell_wall_price"]) if pd.notna(w.get("sell_wall_price")) else None

        # Regime reset
        if state.last_wall_time is None or wall_time != state.last_wall_time:
            state.retests_buy = 0
            state.retests_sell = 0
            state.last_wall_time = wall_time

        state.tick_n += 1
        if state.tick_n % 20 == 0:
            append_trade_log({
                "event": "STRAT_TICK",
                "candle_time": candle_time,
                "close": c,
                "ema": ema,
                "atr": atr,
                "ref_price": ref_price,
                "buy_wall": buy_wall,
                "sell_wall": sell_wall,
                "retests_buy": state.retests_buy,
                "retests_sell": state.retests_sell,
                "wall_time": wall_time,
            })

        # --- Manage positions: time exit + trailing ---
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)

        # ensure meta
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
                if ok:
                    state.pos_meta.pop(ticket, None)

        # refresh open positions after closes
        open_positions = get_open_positions_on_symbol(MT5_SYMBOL)

        # trailing logic (BE @ +1R then ATR)
        if ENABLE_TRAILING and open_positions and (not np.isnan(atr)) and atr > 0:
            tick = mt5.symbol_info_tick(MT5_SYMBOL)
            if tick is not None:
                for p in list(open_positions):
                    ticket = str(getattr(p, "ticket", ""))
                    meta = state.pos_meta.get(ticket)

                    if (meta is None) or (float(meta.get("entry", 0.0)) <= 0) or (float(meta.get("risk_per_unit", 0.0)) <= 0):
                        # rebuild
                        entry_px = float(getattr(p, "price_open", 0.0))
                        sl_px = float(getattr(p, "sl", 0.0))
                        side = "long" if p.type == mt5.POSITION_TYPE_BUY else "short"
                        risk_per_unit = abs(entry_px - sl_px) if (entry_px > 0 and sl_px > 0) else 0.0
                        opened_epoch = getattr(p, "time", None)
                        opened_dt = datetime.fromtimestamp(int(opened_epoch), tz=timezone.utc) if opened_epoch else datetime.now(timezone.utc)
                        prev_meta = meta if isinstance(meta, dict) else {}

                        state.pos_meta[ticket] = meta = {
                            **prev_meta,
                            "opened_time_utc": prev_meta.get("opened_time_utc", fmt_z_time(opened_dt)),
                            "side": side,
                            "entry": entry_px,
                            "initial_sl": sl_px,
                            "risk_per_unit": risk_per_unit,
                            "max_exit_candle_time": prev_meta.get(
                                "max_exit_candle_time",
                                fmt_z_time(opened_dt + timedelta(minutes=MAX_HOLD_MINUTES)),
                            ),
                        }

                    side = meta.get("side")
                    entry = float(meta.get("entry", 0.0))
                    risk_per_unit = float(meta.get("risk_per_unit", 0.0))
                    if entry <= 0 or risk_per_unit <= 0:
                        continue

                    price_now = float(tick.bid) if side == "long" else float(tick.ask)
                    pnl_per_unit = (price_now - entry) if side == "long" else (entry - price_now)
                    r_mult = pnl_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

                    if r_mult < TRAIL_START_R:
                        continue

                    be_sl = (entry + BE_BUFFER) if side == "long" else (entry - BE_BUFFER)
                    atr_sl = (price_now - ATR_TRAIL_MULT * atr) if side == "long" else (price_now + ATR_TRAIL_MULT * atr)

                    cur_sl = float(getattr(p, "sl", 0.0))
                    cur_tp = float(getattr(p, "tp", 0.0))

                    if side == "long":
                        candidate = max(cur_sl if cur_sl > 0 else -1e18, be_sl, atr_sl)
                        if candidate >= price_now:
                            continue
                        new_sl = candidate
                    else:
                        candidate = min(cur_sl if cur_sl > 0 else 1e18, be_sl, atr_sl)
                        if candidate <= price_now:
                            continue
                        new_sl = candidate

                    new_sl = norm_price(new_sl)
                    if cur_sl > 0 and abs(new_sl - cur_sl) < (SYMBOL_INFO.point * 2):
                        continue

                    ok, res = modify_position_sl_tp(p, new_sl, cur_tp)
                    append_trade_log({
                        "event": "TRAIL_UPDATE",
                        "candle_time": candle_time,
                        "ticket": ticket,
                        "side": side,
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

        # --- Capacity check across OPEN + PENDING ---
        pending_orders = get_pending_orders_on_symbol(MT5_SYMBOL)
        total_exposure = len(open_positions) + len(pending_orders)
        if total_exposure >= MAX_OPEN_POSITIONS:
            continue

        # --- Retest counting ---
        if buy_wall is not None and l <= buy_wall + TOUCH_DIST:
            state.retests_buy += 1
        if sell_wall is not None and h >= sell_wall - TOUCH_DIST:
            state.retests_sell += 1

        if ref_price is None:
            continue

        def within_mwd(level: float) -> bool:
            return abs(ref_price - level) <= MAX_WALL_DISTANCE

        # --- Market entries strict rules ---
        entries_this_candle = 0

        def can_market_enter() -> bool:
            nonlocal entries_this_candle
            if entries_this_candle >= MAX_ENTRIES_PER_CANDLE:
                return False
            if state.last_entry_candle_time == candle_time:
                return False
            # Must also respect exposure cap
            poss_now = get_open_positions_on_symbol(MT5_SYMBOL)
            pend_now = get_pending_orders_on_symbol(MT5_SYMBOL)
            if (len(poss_now) + len(pend_now)) >= MAX_OPEN_POSITIONS:
                return False
            return True

        def mark_market_entered():
            nonlocal entries_this_candle
            entries_this_candle += 1
            state.last_entry_candle_time = candle_time

        # SHORT breakout
        if ALLOW_SHORTS and buy_wall is not None:
            mwd_ok = within_mwd(buy_wall)
            ret_ok = state.retests_buy >= RETESTS_REQUIRED
            broken_down = (c <= buy_wall - BREAK_BUFFER)
            trend_ok = (np.isnan(ema)) or ((c < ema) and ((ema - c) >= MIN_EMA_DIST))

            if (ret_ok or broken_down):
                append_trade_log({
                    "event": "SHORT_CHECK",
                    "candle_time": candle_time,
                    "wall_time": wall_time,
                    "close": c,
                    "ema": ema,
                    "ref_price": ref_price,
                    "buy_wall": buy_wall,
                    "retests_buy": state.retests_buy,
                    "mwd_ok": mwd_ok,
                    "ret_ok": ret_ok,
                    "broken_down": broken_down,
                    "trend_ok": trend_ok,
                })

            if mwd_ok and ret_ok and broken_down and trend_ok and can_market_enter():
                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                entry = float(tick.bid) if tick else c
                entry = norm_price(entry)
                sl = norm_price(buy_wall + STOP_BUFFER)

                risk_per_unit = sl - entry
                if risk_per_unit > 0:
                    tp = norm_price(entry - TP_R * risk_per_unit)
                    vol = calc_volume_for_cash_risk(MT5_SYMBOL, "short", entry, sl, RISK_CASH)

                    append_trade_log({
                        "event": "SIGNAL_SHORT_MKT",
                        "candle_time": candle_time,
                        "entry_est": entry,
                        "sl": sl,
                        "tp": tp,
                        "volume": vol,
                        "open_n": len(open_positions),
                        "pending_n": len(pending_orders),
                    })

                    if not SIGNALS_ONLY and vol > 0:
                        ok, res = send_market_order(
                            MT5_SYMBOL, "short", vol, sl, tp,
                            comment=f"WBv3 MKT S {ASSET_TAG}"
                        )
                        append_trade_log({
                            "event": "ORDER_SHORT_MKT",
                            "candle_time": candle_time,
                            "ok": ok,
                            "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                            "vol": vol,
                        })
                        if ok:
                            mark_market_entered()
                            state.retests_buy = 0

                continue  # same as your style: after short path, go next candle

        # LONG breakout
        if ALLOW_LONGS and sell_wall is not None:
            mwd_ok = within_mwd(sell_wall)
            ret_ok = state.retests_sell >= RETESTS_REQUIRED
            broken_up = (c >= sell_wall + BREAK_BUFFER)
            trend_ok = (np.isnan(ema)) or ((c > ema) and ((c - ema) >= MIN_EMA_DIST))

            if (ret_ok or broken_up):
                append_trade_log({
                    "event": "LONG_CHECK",
                    "candle_time": candle_time,
                    "wall_time": wall_time,
                    "close": c,
                    "ema": ema,
                    "ref_price": ref_price,
                    "sell_wall": sell_wall,
                    "retests_sell": state.retests_sell,
                    "mwd_ok": mwd_ok,
                    "ret_ok": ret_ok,
                    "broken_up": broken_up,
                    "trend_ok": trend_ok,
                })

            if mwd_ok and ret_ok and broken_up and trend_ok and can_market_enter():
                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                entry = float(tick.ask) if tick else c
                entry = norm_price(entry)
                sl = norm_price(sell_wall - STOP_BUFFER)

                risk_per_unit = entry - sl
                if risk_per_unit > 0:
                    tp = norm_price(entry + TP_R * risk_per_unit)
                    vol = calc_volume_for_cash_risk(MT5_SYMBOL, "long", entry, sl, RISK_CASH)

                    append_trade_log({
                        "event": "SIGNAL_LONG_MKT",
                        "candle_time": candle_time,
                        "entry_est": entry,
                        "sl": sl,
                        "tp": tp,
                        "volume": vol,
                        "open_n": len(open_positions),
                        "pending_n": len(pending_orders),
                    })

                    if not SIGNALS_ONLY and vol > 0:
                        ok, res = send_market_order(
                            MT5_SYMBOL, "long", vol, sl, tp,
                            comment=f"WBv3 MKT L {ASSET_TAG}"
                        )
                        append_trade_log({
                            "event": "ORDER_LONG_MKT",
                            "candle_time": candle_time,
                            "ok": ok,
                            "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                            "vol": vol,
                        })
                        if ok:
                            mark_market_entered()
                            state.retests_sell = 0

                continue

# =========================
# Main
# =========================
def main():
    if "PASTE_YOUR_OANDA_TOKEN_HERE" in OANDA_TOKEN:
        raise RuntimeError("Set OANDA_TOKEN at the top of the script.")

    print("\n=== WBv3 EMA + TRAIL + LIMITS (Script A parity) ===")
    print(f"[ROOT] {BASE_DIR}")
    print(f"[ASSET] {ASSET_TAG}  [STRAT] {STRATEGY_TAG}")
    print(f"[OANDA] api={OANDA_API_URL} instrument={INSTRUMENT}")
    print(f"[MT5]   server={MT5_SERVER} symbol={MT5_SYMBOL} magic={MAGIC}")
    print(f"[RISK]  cash={RISK_CASH} max_hold={MAX_HOLD_MINUTES} max_open={MAX_OPEN_POSITIONS} max_entries_per_candle={MAX_ENTRIES_PER_CANDLE}")
    print(f"[BREAK] EMA={EMA_SPAN} MIN_EMA_DIST={MIN_EMA_DIST} RET={RETESTS_REQUIRED} TOUCH={TOUCH_DIST} BREAK={BREAK_BUFFER} STOP={STOP_BUFFER} TP_R={TP_R} MWD={MAX_WALL_DISTANCE}")
    print(f"[TRAIL] enabled={ENABLE_TRAILING} start_R={TRAIL_START_R} BE_buf={BE_BUFFER} ATR_p={ATR_TRAIL_PERIOD} ATR_mult={ATR_TRAIL_MULT}")
    print(f"[LIM]   enabled={ENABLE_LIMITS} both_sides={PLACE_BOTH_SIDES_LIMITS} offset={LIMIT_ENTRY_OFFSET} max_dist_mkt={LIMIT_MAX_DIST_FROM_MKT}\n")

    init_mt5()
    state = BotState()

    prev = load_state()
    if prev:
        apply_state_to_botstate(state, prev)
        if getattr(state, "instrument", "") and state.instrument != INSTRUMENT:
            append_trade_log({
                "event": "STATE_RESET_INSTRUMENT_MISMATCH",
                "prev_instrument": state.instrument,
                "new_instrument": INSTRUMENT,
            })
            state = BotState()

        append_trade_log({
            "event": "STATE_RESTORED",
            "note": f"restored last_processed_candle_time={state.last_processed_candle_time}",
        })
    else:
        append_trade_log({"event": "STATE_NEW", "note": "no state.json found; starting fresh"})

    if state.pos_meta is None or not isinstance(state.pos_meta, dict):
        state.pos_meta = {}

    if not TRADES_JSONL.exists():
        append_trade_log({"event": "INIT", "note": "bot started"})

    try:
        while True:
            maybe_run_books_logger(state)
            maybe_run_candles_logger(state)
            maybe_process_new_candles_and_trade(state)

            state.instrument = INSTRUMENT
            s = asdict(state)
            s["saved_at_utc"] = fmt_z_time(datetime.now(timezone.utc))
            save_state(s)

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt")
    finally:
        shutdown_mt5()
        print("[MT5] shutdown OK")

if __name__ == "__main__":
    main()
