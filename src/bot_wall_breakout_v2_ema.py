#!/usr/bin/env python
"""
bot_wall_breakout_v2_ema.py
===========================

ONE SCRIPT that replicates what your backtest project did:

DATA LOGGING (same behaviors as backtest loggers)
- OANDA orderBook + positionBook snapshots -> JSONL (append-only)
  input/raw/orderbook/orderbook_YYYYMMDD.jsonl
  input/raw/positionbook/positionbook_YYYYMMDD.jsonl
  *dedupe by orderBook.time*

- OANDA candles M1 (mid) -> JSONL (append-only)
  input/raw/candles/XAU_USD_M1_YYYYMMDD.jsonl
  *dedupe by candle time*

- Walls extraction -> CSV (append-only)
  output/reports/walls/walls_YYYYMMDD.csv
  using the SAME compute_best_walls_from_orderbook_snapshot()

STRATEGY (same as your grid/backtest v2 logic)
- Uses OANDA candle closes
- Uses latest wall snapshot where wall_dt <= candle_dt (regime)
- Retest counting within regime; reset on wall_time change
- Breakout/Breakdown after RETESTS_REQUIRED
- EMA filter + MIN_EMA_DIST
- One position at a time
- SL beyond wall (STOP_BUFFER)
- TP = entry ± TP_R * risk_per_unit
- Time exit after MAX_HOLD_MINUTES (on candle close time)

EXECUTION
- Trades placed on MT5 (ICMarkets demo etc.)
- Risk sizing uses MT5 order_calc_profit so cash-risk aligns to RISK_CASH
- SL/TP broker-managed; time exit closes at market

PORTABLE PATHS
- Uses script directory as "project root" (so you can move the .py anywhere)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
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
MT5_LOGIN         = 52652352
MT5_PASSWORD      = "0e6IM0LjE$l47R"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\ALGO-OrderBook_XAU\terminal64.exe"

LOCAL_TZ   = pytz_timezone("Europe/London")
MT5_SYMBOL = "XAUUSD"   # adjust if broker uses suffix
SYMBOL_INFO = None      # filled by init_mt5()

# =========================
# OANDA CONFIG (EDIT)
# =========================
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"  # practice
OANDA_TOKEN   = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
INSTRUMENT    = "XAU_USD"  # OANDA instrument name

# =========================
# BOOKS LOGGER CONFIG (same idea)
# =========================
BOOK_STEP_SECONDS        = 20 * 60   # 1200 (OANDA books cadence)
BOOK_GRACE_SECONDS       = 180       # retry window
BOOK_RETRY_EVERY_SECONDS = 10

BOOK_RANGE_DOLLARS = 25.0
WALL_TOTAL_MIN     = 0.08
WALL_IMB_MIN       = 0.06

# =========================
# CANDLES LOGGER CONFIG (same idea)
# =========================
GRANULARITY           = "M1"
PRICE_TYPE            = "M"     # midpoint
CANDLE_STEP_SECONDS   = 20 * 60
CANDLE_WAKE_DELAY_SEC = 8
CANDLE_GRACE_SECONDS  = 60
CANDLE_RETRY_EVERY    = 10

# =========================
# STRATEGY CONFIG
# =========================
EMA_SPAN          = 21
MIN_EMA_DIST      = 0.5
RETESTS_REQUIRED  = 5
TOUCH_DIST        = 1.0
BREAK_BUFFER      = 0.2
STOP_BUFFER       = 1.0
TP_R              = 2.5
MAX_WALL_DISTANCE = 12.0

# =========================
# EXECUTION / RISK
# =========================
RISK_CASH        = 1000.0
MAX_HOLD_MINUTES = 180
ALLOW_LONGS      = True
ALLOW_SHORTS     = True

SIGNALS_ONLY     = False  # set True to only log signals

MAGIC            = 2209001
DEVIATION        = 20

# Main loop tick
LOOP_SLEEP_SECONDS = 1.0

# =========================
# PORTABLE PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent  # save everything relative to this script folder

OUT_ORDER_DIR = BASE_DIR / "input" / "raw" / "orderbook"
OUT_POS_DIR   = BASE_DIR / "input" / "raw" / "positionbook"
OUT_CAND_DIR  = BASE_DIR / "input" / "raw" / "candles"
OUT_WALLS_DIR = BASE_DIR / "output" / "reports" / "walls"
OUT_LIVE_DIR  = BASE_DIR / "output" / "live" / "wall_breakout" / "v2_ema"

for d in [OUT_ORDER_DIR, OUT_POS_DIR, OUT_CAND_DIR, OUT_WALLS_DIR, OUT_LIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRADES_JSONL = OUT_LIVE_DIR / "trades.jsonl"
STATE_JSON = OUT_LIVE_DIR / "state.json"

# =========================
# Helpers (same as backtest style)
# =========================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def parse_oanda_time(s: str) -> datetime:
    # "2025-12-15T15:20:00Z"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def parse_z_time(s: str) -> datetime:
    # "2025-12-15T16:15:00.000000000Z" or "2025-12-15T16:15:00Z"
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
    """
    EXACT same logic as your backtest logger_books.py
    """
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

def load_walls_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["wall_dt"] = df["time"].apply(parse_z_time)  # walls time is ...Z
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

def get_supported_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    # Many brokers expose a single "default" filling mode here
    fm = getattr(info, "filling_mode", None)

    # If it’s one of the known modes, use it
    if fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        return int(fm)

    # Safe fallback (most commonly accepted)
    return mt5.ORDER_FILLING_IOC

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
    params = {
        "granularity": GRANULARITY,
        "from": time_from,
        "to": time_to,
        "price": PRICE_TYPE,
    }
    return fetch_json(f"/instruments/{INSTRUMENT}/candles", timeout=45, params=params)

# =========================
# Cadence helpers (same concept)
# =========================
def next_boundary(now: datetime, step_seconds: int) -> datetime:
    epoch = int(now.timestamp())
    next_epoch = ((epoch // step_seconds) + 1) * step_seconds
    return datetime.fromtimestamp(next_epoch, tz=timezone.utc).replace(second=0, microsecond=0)

# =========================
# MT5 execution helpers
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
    print(f"[MT5] Connected login={acc.login} balance={acc.balance:.2f} equity={acc.equity:.2f}")
    print(f"[MT5] {MT5_SYMBOL} digits={info.digits} point={info.point} vol_min={info.volume_min} step={info.volume_step}")

def norm_price(x: float) -> float:
    return float(round(x, SYMBOL_INFO.digits))

def shutdown_mt5() -> None:
    mt5.shutdown()

def get_open_position_on_symbol(symbol: str) -> Optional[mt5.TradePosition]:
    poss = mt5.positions_get(symbol=symbol)
    if not poss:
        return None
    for p in poss:
        if int(getattr(p, "magic", 0)) == int(MAGIC):
            return p
    return None

def clamp_to_step(val: float, vmin: float, vmax: float, step: float) -> float:
    val = max(vmin, min(vmax, val))
    if step <= 0:
        return val
    n = round((val - vmin) / step)
    return vmin + n * step

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

    # Try preferred filling mode first, then fallback
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
            last = mt5.last_error()
            continue

        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, res

        # If unsupported filling mode, try next
        if "Unsupported filling mode" in str(res):
            continue

        # Any other failure: return immediately (so you see real reason)
        return False, res

    return False, "All filling modes rejected (IOC/RETURN/FOK)"

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
        "type_filling": get_supported_filling_mode(pos.symbol),
    }
    res = mt5.order_send(req)
    if res is None:
        return False, f"order_send None: {mt5.last_error()}"
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        return False, res
    return True, res

# =========================
# Live logging
# =========================
def append_trade_log(row: Dict[str, Any]) -> None:
    """
    Append-only JSONL log (one event per line).
    Adds a schema_version and ensures everything is JSON-serializable.
    """
    obj = dict(row)

    # Optional: add a version tag so you can evolve log fields later
    obj.setdefault("schema_version", 1)

    # Ensure dt_utc exists for every log row (if caller forgot)
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

def apply_state_to_botstate(state: BotState, data: Dict[str, Any]) -> None:
    """
    Safely copy persisted values into BotState (ignore unknown keys).
    """
    for k, v in data.items():
        if hasattr(state, k):
            setattr(state, k, v)


# =========================
# Bot State
# =========================
@dataclass
class BotState:
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

    # position tracking (for time exit)
    entry_candle_time: Optional[str] = None
    max_exit_candle_time: Optional[str] = None
    side: Optional[str] = None
    sl: Optional[float] = None
    tp: Optional[float] = None

    tick_n: int = 0

# =========================
# Books cadence runner (dedupe + aligned)
# =========================
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
            t_last, _ = get_book_time_and_price(last_obj)
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
            wake = now  # bootstrap immediately
        state.next_books_wake_utc = fmt_z_time(wake)

    wake_dt = parse_z_time(state.next_books_wake_utc)

    if now < wake_dt:
        return

    # when waking: retry in grace window until NEW snapshot time
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
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "book_time": t_ob,
                "ref_price": p_ob,
                "buy_wall": walls_row.get("buy_wall_price"),
                "sell_wall": walls_row.get("sell_wall_price"),
                "attempts": attempts,
            })

            # schedule next wake based on cadence
            try:
                last_dt = parse_oanda_time(state.last_saved_book_time)
                next_expected = last_dt + timedelta(seconds=BOOK_STEP_SECONDS)
                state.next_books_wake_utc = fmt_z_time(next_expected + timedelta(seconds=5))
            except Exception:
                state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=BOOK_STEP_SECONDS))

            print(f"[BOOKS] NEW {t_ob} ref={p_ob} BUY={walls_row.get('buy_wall_price')} SELL={walls_row.get('sell_wall_price')}")
            return

        except Exception as e:
            append_trade_log({"event": "BOOKS_ERR", "dt_utc": fmt_z_time(datetime.now(timezone.utc)), "err": str(e)})
            time.sleep(BOOK_RETRY_EVERY_SECONDS)

    # missed grace
    state.next_books_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=10))
    print("[BOOKS] WARN: no new snapshot within grace")

# =========================
# Candles cadence runner (dedupe + aligned)
# =========================
def maybe_run_candles_logger(state: BotState) -> None:
    now = datetime.now(timezone.utc)
    day = day_yyyymmdd(now)
    out_path = OUT_CAND_DIR / f"{INSTRUMENT}_{GRANULARITY}_{day}.jsonl"

    # init last_saved_candle_time from file if missing
    if state.last_saved_candle_time is None:
        t_last = read_last_candle_time(out_path)
        state.last_saved_candle_time = t_last

    # schedule next wake if missing
    if state.next_candle_wake_utc is None:
        boundary = next_boundary(now, CANDLE_STEP_SECONDS)
        wake = boundary + timedelta(seconds=CANDLE_WAKE_DELAY_SEC)
        state.next_candle_wake_utc = fmt_z_time(wake)

    wake_dt = parse_z_time(state.next_candle_wake_utc)
    if now < wake_dt:
        return

    last_dt = parse_z_time(state.last_saved_candle_time) if state.last_saved_candle_time else None

    # fetch window
    fetch_to = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if last_dt:
        fetch_from = last_dt - timedelta(minutes=1)
    else:
        fetch_from = fetch_to - timedelta(seconds=CANDLE_STEP_SECONDS)

    if fetch_to <= fetch_from:
        fetch_from = fetch_to - timedelta(minutes=25)

    deadline = datetime.now(timezone.utc) + timedelta(seconds=CANDLE_GRACE_SECONDS)
    attempts = 0

    while datetime.now(timezone.utc) <= deadline:
        attempts += 1
        try:
            raw = fetch_candles(fmt_z_time(fetch_from), fmt_z_time(fetch_to))
            candles = normalize_candles(raw)
            candles = [c for c in candles if c.get("complete")]

            if not candles:
                time.sleep(CANDLE_RETRY_EVERY)
                continue

            # keep only > last_dt
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

            # append
            for c in new:
                append_jsonl(out_path, c)

            state.last_saved_candle_time = new[-1]["time"]

            append_trade_log({
                "event": "CANDLES_SAVED",
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "n_new": len(new),
                "newest": state.last_saved_candle_time,
                "attempts": attempts,
            })

            # schedule next wake
            boundary = next_boundary(datetime.now(timezone.utc), CANDLE_STEP_SECONDS)
            state.next_candle_wake_utc = fmt_z_time(boundary + timedelta(seconds=CANDLE_WAKE_DELAY_SEC))

            print(f"[CANDLES] +{len(new)} newest={state.last_saved_candle_time}")
            return

        except Exception as e:
            append_trade_log({"event": "CANDLES_ERR", "dt_utc": fmt_z_time(datetime.now(timezone.utc)), "err": str(e)})
            time.sleep(CANDLE_RETRY_EVERY)

    # missed grace -> try again soon
    state.next_candle_wake_utc = fmt_z_time(datetime.now(timezone.utc) + timedelta(seconds=10))
    print("[CANDLES] WARN: no new candles within grace")

# =========================
# Strategy runner (process NEW candle close)
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
    else:
        df["ema"] = np.nan
    return df

def load_recent_candles_for_ema_multi(days: List[str], keep_last: int = 2000) -> pd.DataFrame:
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
    # recompute EMA on the combined series (important)
    out["ema"] = compute_ema(out["c"], EMA_SPAN)
    return out

def maybe_process_new_candles_and_trade(state: BotState) -> None:
    """
    Critical fix:
    - process ALL new candles since last_processed_candle_time (sequentially),
      so retests/breakouts behave like the backtest.
    - Use the candle's own day to load the correct walls file.
    """
    # Load *today* candles file (fine), but process ALL new rows inside it.
    # (Optional improvement below: include yesterday for EMA continuity)
    now = datetime.now(timezone.utc)
    today = day_yyyymmdd(now)
    yday = day_yyyymmdd(now - timedelta(days=1))

    candles = load_recent_candles_for_ema_multi([yday, today], keep_last=3000)

    if candles.empty:
        return
    if len(candles) < max(EMA_SPAN + 5, 10):
        return

    # Determine which rows are NEW
    if state.last_processed_candle_time is None:
        # On first run: if we have a last saved candle, start from it (process after it).
        # If we don't, bootstrap with just the last candle to avoid replaying the whole file.
        if state.last_saved_candle_time:
            state.last_processed_candle_time = state.last_saved_candle_time
            mask = candles["time"] > state.last_processed_candle_time
            new_rows = candles.loc[mask]
        else:
            new_rows = candles.iloc[-1:]
    else:
        mask = candles["time"] > state.last_processed_candle_time
        new_rows = candles.loc[mask]

    if new_rows.empty:
        return

    append_trade_log({
        "event": "REPLAY_BATCH",
        "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
        "note": f"processing {len(new_rows)} new candles since {state.last_processed_candle_time}",
    })

    # IMPORTANT: loop candle-by-candle like backtest
    for _, row in new_rows.iterrows():
        candle_time = str(row["time"])
        candle_dt: datetime = row["dt"]
        h = float(row["h"]); l = float(row["l"]); c = float(row["c"])
        ema = float(row["ema"]) if not pd.isna(row["ema"]) else float("nan")

        # Update processed marker immediately (so if we crash mid-loop, we don't reprocess earlier ones)
        state.last_processed_candle_time = candle_time

        # Load walls for the candle's day (not "today")
        candle_day = candle_dt.strftime("%Y%m%d")
        walls_path = OUT_WALLS_DIR / f"walls_{candle_day}.csv"

        if not walls_path.exists():
            # fallback to yesterday's walls if today's hasn't started yet
            yday_str = day_yyyymmdd(candle_dt - timedelta(days=1))
            y_walls_path = OUT_WALLS_DIR / f"walls_{yday_str}.csv"

            if y_walls_path.exists():
                walls_path = y_walls_path
            else:
                append_trade_log({
                    "event": "STRAT_SKIP_NO_WALL_FILE",
                    "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                    "candle_time": candle_time,
                    "walls_file": str(walls_path),
                })
                continue

        walls = load_walls_csv(walls_path)
        w = latest_wall_row(walls, candle_dt)
        if w is None:
            append_trade_log({
                "event": "STRAT_SKIP_NO_WALL_ROW",
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "candle_time": candle_time,
                "candle_dt": fmt_z_time(candle_dt),
            })
            continue

        wall_time = str(w["time"])
        ref_price = float(w["ref_price"]) if pd.notna(w["ref_price"]) else None
        buy_wall  = float(w["buy_wall_price"]) if pd.notna(w.get("buy_wall_price")) else None
        sell_wall = float(w["sell_wall_price"]) if pd.notna(w.get("sell_wall_price")) else None

        # Regime reset (same as backtest)
        if state.last_wall_time is None or wall_time != state.last_wall_time:
            state.retests_buy = 0
            state.retests_sell = 0
            state.last_wall_time = wall_time

        # Heartbeat every 20 processed candles
        state.tick_n += 1

        if state.tick_n % 20 == 0:
            append_trade_log({
                "event": "STRAT_TICK",
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "candle_time": candle_time,
                "close": c,
                "ema": ema,
                "ref_price": ref_price,
                "buy_wall": buy_wall,
                "sell_wall": sell_wall,
                "retests_buy": state.retests_buy,
                "retests_sell": state.retests_sell,
                "wall_time": wall_time,
            })

        # Manage open position first (same behavior)
        pos = get_open_position_on_symbol(MT5_SYMBOL)
        if pos is not None:
            if state.max_exit_candle_time:
                max_exit_dt = parse_z_time(state.max_exit_candle_time)
                if candle_dt >= max_exit_dt:
                    ok, res = close_position_market(pos, "max_hold")
                    append_trade_log({
                        "event": "TIME_EXIT",
                        "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                        "candle_time": candle_time,
                        "ticket": getattr(pos, "ticket", None),
                        "ok": ok,
                        "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                    })
                    if ok:
                        state.entry_candle_time = None
                        state.max_exit_candle_time = None
                        state.side = None
                        state.sl = None
                        state.tp = None
            # While in a position, do not evaluate new entries
            continue

        # Retest counting (now happens for EVERY new candle)
        if buy_wall is not None and l <= buy_wall + TOUCH_DIST:
            state.retests_buy += 1
        if sell_wall is not None and h >= sell_wall - TOUCH_DIST:
            state.retests_sell += 1

        if ref_price is None:
            continue

        def within_mwd(level: float) -> bool:
            return abs(ref_price - level) <= MAX_WALL_DISTANCE

        def log_short_check(reason: str, **extra):
            append_trade_log({
                "event": "SHORT_CHECK",
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "candle_time": candle_time,
                "wall_time": wall_time,
                "close": c,
                "ema": ema,
                "ref_price": ref_price,
                "buy_wall": buy_wall,
                "retests_buy": state.retests_buy,
                "retests_required": RETESTS_REQUIRED,
                "break_buffer": BREAK_BUFFER,
                "min_ema_dist": MIN_EMA_DIST,
                "max_wall_distance": MAX_WALL_DISTANCE,
                "reason": reason,
                **extra
            })

        def log_long_check(reason: str, **extra):
            append_trade_log({
                "event": "LONG_CHECK",
                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                "candle_time": candle_time,
                "wall_time": wall_time,
                "close": c,
                "ema": ema,
                "ref_price": ref_price,
                "sell_wall": sell_wall,
                "retests_sell": state.retests_sell,
                "retests_required": RETESTS_REQUIRED,
                "break_buffer": BREAK_BUFFER,
                "min_ema_dist": MIN_EMA_DIST,
                "max_wall_distance": MAX_WALL_DISTANCE,
                "reason": reason,
                **extra
            })

        # SHORT (breakdown)
        if ALLOW_SHORTS:
            if buy_wall is None:
                pass
            else:
                mwd_ok = within_mwd(buy_wall)
                ret_ok = state.retests_buy >= RETESTS_REQUIRED
                broken_down = (c <= buy_wall - BREAK_BUFFER)
                trend_ok = (np.isnan(ema)) or ((c < ema) and ((ema - c) >= MIN_EMA_DIST))

                # log the decision whenever we are "close" to qualifying:
                # (either enough retests OR price is near break level)
                if ret_ok or broken_down:
                    log_short_check(
                        "EVAL",
                        mwd_ok=mwd_ok,
                        ret_ok=ret_ok,
                        broken_down=broken_down,
                        trend_ok=trend_ok,
                        dist_ref_to_wall=abs(ref_price - buy_wall) if (ref_price is not None) else None,
                    )

                if mwd_ok and ret_ok and broken_down and trend_ok:
                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
                    entry = float(tick.bid) if tick else c

                    entry = norm_price(entry)
                    sl = norm_price(buy_wall + STOP_BUFFER)

                    risk_per_unit = sl - entry
                    if risk_per_unit > 0:
                        tp = norm_price(entry - TP_R * risk_per_unit)
                        vol = calc_volume_for_cash_risk(MT5_SYMBOL, "short", entry, sl, RISK_CASH)

                        append_trade_log({
                            "event": "SIGNAL_SHORT",
                            "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                            "candle_time": candle_time,
                            "close": c,
                            "ema": ema,
                            "retests_buy": state.retests_buy,
                            "ref_price": ref_price,
                            "buy_wall": buy_wall,
                            "entry_est": entry,
                            "sl": sl,
                            "tp": tp,
                            "volume": vol,
                            "wall_time": wall_time,
                        })

                        if not SIGNALS_ONLY and vol > 0:
                            ok, res = send_market_order(MT5_SYMBOL, "short", vol, sl, tp,
                                                        comment=f"WBv2EMA S EMA{EMA_SPAN}")

                            append_trade_log({
                                "event": "ORDER_SHORT",
                                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                                "ok": ok,
                                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                                "vol": vol
                            })

                            if ok:
                                state.entry_candle_time = candle_time
                                state.max_exit_candle_time = fmt_z_time(candle_dt + timedelta(minutes=MAX_HOLD_MINUTES))
                                state.side = "short"
                                state.sl = sl
                                state.tp = tp
                                state.retests_buy = 0
                    continue

        # LONG (breakout)
        if ALLOW_LONGS:
            if sell_wall is None:
                pass
            else:
                mwd_ok = within_mwd(sell_wall)
                ret_ok = state.retests_sell >= RETESTS_REQUIRED
                broken_up = (c >= sell_wall + BREAK_BUFFER)
                trend_ok = (np.isnan(ema)) or ((c > ema) and ((c - ema) >= MIN_EMA_DIST))

                # log the decision whenever we are "close" to qualifying:
                if ret_ok or broken_up:
                    log_long_check(
                        "EVAL",
                        mwd_ok=mwd_ok,
                        ret_ok=ret_ok,
                        broken_up=broken_up,
                        trend_ok=trend_ok,
                        dist_ref_to_wall=abs(ref_price - sell_wall) if (ref_price is not None) else None,
                    )

                if mwd_ok and ret_ok and broken_up and trend_ok:
                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
                    entry = float(tick.ask) if tick else c

                    entry = norm_price(entry)
                    sl = norm_price(sell_wall - STOP_BUFFER)

                    risk_per_unit = entry - sl
                    if risk_per_unit > 0:
                        tp = norm_price(entry + TP_R * risk_per_unit)
                        vol = calc_volume_for_cash_risk(MT5_SYMBOL, "long", entry, sl, RISK_CASH)

                        append_trade_log({
                            "event": "SIGNAL_LONG",
                            "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                            "candle_time": candle_time,
                            "close": c,
                            "ema": ema,
                            "retests_sell": state.retests_sell,
                            "ref_price": ref_price,
                            "sell_wall": sell_wall,
                            "entry_est": entry,
                            "sl": sl,
                            "tp": tp,
                            "volume": vol,
                            "wall_time": wall_time,
                        })

                        if not SIGNALS_ONLY and vol > 0:
                            ok, res = send_market_order(MT5_SYMBOL, "long", vol, sl, tp,
                                                        comment=f"WBv2EMA L EMA{EMA_SPAN}")

                            append_trade_log({
                                "event": "ORDER_LONG",
                                "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
                                "ok": ok,
                                "res": (res._asdict() if hasattr(res, "_asdict") else str(res)),
                                "vol": vol
                            })

                            if ok:
                                state.entry_candle_time = candle_time
                                state.max_exit_candle_time = fmt_z_time(candle_dt + timedelta(minutes=MAX_HOLD_MINUTES))
                                state.side = "long"
                                state.sl = sl
                                state.tp = tp
                                state.retests_sell = 0
                    continue


# =========================
# Main
# =========================
def main():
    if "PASTE_YOUR_OANDA_TOKEN_HERE" in OANDA_TOKEN:
        raise RuntimeError("Set OANDA_TOKEN at the top of the script.")

    print("\n=== WBv2 EMA LIVE BOT (Backtest-Replica Data Flow) ===")
    print(f"[ROOT] {BASE_DIR}")
    print(f"[OANDA] api={OANDA_API_URL} instrument={INSTRUMENT}")
    print(f"[BOOKS] step={BOOK_STEP_SECONDS}s grace={BOOK_GRACE_SECONDS}s retry={BOOK_RETRY_EVERY_SECONDS}s range=±{BOOK_RANGE_DOLLARS} total_min={WALL_TOTAL_MIN} imb_min={WALL_IMB_MIN}")
    print(f"[CAND]  step={CANDLE_STEP_SECONDS}s wake_delay={CANDLE_WAKE_DELAY_SEC}s grace={CANDLE_GRACE_SECONDS}s retry={CANDLE_RETRY_EVERY}s")
    print(f"[STRAT] EMA={EMA_SPAN} MIN_DIST={MIN_EMA_DIST} RET={RETESTS_REQUIRED} TOUCH={TOUCH_DIST} BREAK={BREAK_BUFFER} STOP={STOP_BUFFER} TP_R={TP_R} MWD={MAX_WALL_DISTANCE}")
    print(f"[EXEC]  MT5={MT5_SERVER} {MT5_SYMBOL} risk_cash={RISK_CASH} max_hold={MAX_HOLD_MINUTES} signals_only={SIGNALS_ONLY}\n")

    init_mt5()
    state = BotState()

    # Restore previous bot state (so we replay missed candles)
    prev = load_state()
    if prev:
        apply_state_to_botstate(state, prev)
        append_trade_log({
            "event": "STATE_RESTORED",
            "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
            "note": f"restored last_processed_candle_time={state.last_processed_candle_time}",
        })
    else:
        append_trade_log({
            "event": "STATE_NEW",
            "dt_utc": fmt_z_time(datetime.now(timezone.utc)),
            "note": "no state.json found; starting fresh",
        })

    # create live log header
    if not TRADES_JSONL.exists():
        append_trade_log({"event": "INIT", "note": "bot started"})

    try:
        while True:
            # 1) Run books logger (aligned + deduped + walls csv)
            maybe_run_books_logger(state)

            # 2) Run candles logger (aligned + deduped)
            maybe_run_candles_logger(state)

            # 3) Process newest candle for strategy + possible MT5 order
            maybe_process_new_candles_and_trade(state)

            # save state every loop
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
