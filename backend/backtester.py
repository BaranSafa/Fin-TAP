"""
backtester.py  —  Fin-TAP v3.0 Draft

Backtest engine: her horizon periyodunda RSI + trend sinyali kullanan
stratejiyi tarihsel veriyle simüle eder.
→ P&L eğrisi, win rate, max drawdown, Sharpe ratio vs buy-and-hold

Entegrasyon: bu dosyayı backend/backtester.py olarak kopyala.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    from .data_manager import get_processed_data
except ImportError:
    from data_manager import get_processed_data


# ──────────────────────────────────────────────────────────────────────────────
#  Yardımcı: teknik sinyal üret
# ──────────────────────────────────────────────────────────────────────────────

def _compute_signal(df: pd.DataFrame, idx: int) -> str:
    """
    idx pozisyonunda RSI(14) + SMA(20/50) trend sinyali hesapla.
    Döndürür: 'long' | 'short' | 'neutral'
    """
    window = df.iloc[max(0, idx - 60): idx + 1]
    if len(window) < 15:
        return "neutral"

    closes = window["Close"].values.astype(float)

    # RSI(14)
    diffs = np.diff(closes[-15:])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_g = np.mean(gains) if gains.any() else 0.0
    avg_l = np.mean(losses) if losses.any() else 1e-9
    rsi = 100 - 100 / (1 + avg_g / avg_l)

    # SMA trend
    sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
    sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else closes[-1]
    cur   = closes[-1]

    bullish = rsi < 55 and cur > sma20 and sma20 > sma50
    bearish = rsi > 65 or cur < sma20 < sma50

    if bullish:
        return "long"
    if bearish:
        return "short"
    return "neutral"


# ──────────────────────────────────────────────────────────────────────────────
#  Ana fonksiyon
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    horizon: int = 14,
    lookback_days: int = 365,
    start_capital: float = 1000.0,
    allow_short: bool = False,
) -> Optional[dict]:
    """
    Parametreler
    ------------
    ticker         : hisse/kripto sembolü
    horizon        : sinyal başına kaç günlük pozisyon (7/14/30)
    lookback_days  : kaç günlük geçmişe bak
    start_capital  : başlangıç sermayesi (USD)
    allow_short    : short sinyallerde açığa satış yap mı

    Döndürür: dict ile grafik ve istatistik verileri | None (hata)
    """
    df = get_processed_data(ticker)
    if df is None or df.empty:
        return None

    # lookback_days kadar son veriyi al
    df = df.tail(lookback_days + horizon + 60).copy()
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])

    if len(df) < horizon * 3:
        return None

    closes = df["Close"].values.astype(float)
    dates  = [d.strftime("%Y-%m-%d") for d in df.index]
    n      = len(closes)

    # ── Strateji simülasyonu ─────────────────────────────────────────────────
    portfolio   = start_capital   # strateji portföy değeri
    bah_shares  = start_capital / closes[0]   # buy-and-hold hisse sayısı

    portfolio_values = []   # her gün portföy değeri
    bah_values       = []   # buy-and-hold değeri
    trade_dates      = []   # işlem günleri
    trades           = []   # {date, signal, entry, exit, pct_return}

    in_position = False
    position_type = None   # 'long' | 'short'
    entry_price   = 0.0
    entry_idx     = 0
    current_val   = portfolio

    step = max(1, horizon)
    check_indices = list(range(60, n - horizon, step))

    for i in check_indices:
        signal = _compute_signal(df, i)
        entry  = closes[i]
        exit_i = min(i + horizon, n - 1)
        exit_p = closes[exit_i]

        raw_ret = (exit_p - entry) / entry   # gerçek getiri

        if signal == "long":
            trade_ret = raw_ret
            won = raw_ret > 0
        elif signal == "short" and allow_short:
            trade_ret = -raw_ret
            won = raw_ret < 0
        else:
            trade_ret = 0.0
            won = None

        current_val *= (1 + trade_ret)

        trades.append({
            "date":       dates[i],
            "signal":     signal,
            "entry":      round(float(entry), 2),
            "exit":       round(float(exit_p), 2),
            "pct_return": round(trade_ret * 100, 2),
            "won":        won,
        })

    # ── Günlük değer serisi (grafik için) ────────────────────────────────────
    # Her işlem arasında linear interpolasyon
    port_series = [start_capital]
    bah_series  = [start_capital]
    date_series = [dates[60]]

    running = start_capital
    trade_map = {t["date"]: t["pct_return"] for t in trades}

    for idx in range(61, n):
        d = dates[idx]
        if d in trade_map:
            running *= (1 + trade_map[d] / 100)
        port_series.append(round(running, 2))
        bah_series.append(round(closes[idx] / closes[60] * start_capital, 2))
        date_series.append(d)

    # ── İstatistikler ────────────────────────────────────────────────────────
    active_trades = [t for t in trades if t["won"] is not None]
    win_trades    = [t for t in active_trades if t["won"]]
    win_rate      = len(win_trades) / len(active_trades) * 100 if active_trades else 0.0

    returns_arr = np.array([t["pct_return"] for t in active_trades]) / 100
    sharpe = 0.0
    if len(returns_arr) > 1 and returns_arr.std() > 0:
        sharpe = float((returns_arr.mean() / returns_arr.std()) * np.sqrt(252 / horizon))

    # Max drawdown
    peak = start_capital
    max_dd = 0.0
    running2 = start_capital
    for t in trades:
        running2 *= (1 + t["pct_return"] / 100)
        if running2 > peak:
            peak = running2
        dd = (peak - running2) / peak * 100
        if dd > max_dd:
            max_dd = dd

    final_port = port_series[-1]
    final_bah  = bah_series[-1]
    total_ret  = (final_port - start_capital) / start_capital * 100
    bah_ret    = (final_bah  - start_capital) / start_capital * 100

    return {
        "ticker":         ticker,
        "horizon":        horizon,
        "lookback_days":  lookback_days,
        "start_capital":  start_capital,
        # Chart data
        "dates":          date_series[::2],          # her iki günde bir (veri küçültme)
        "portfolio":      port_series[::2],
        "buy_and_hold":   bah_series[::2],
        # Stats
        "total_return":   round(total_ret, 2),
        "bah_return":     round(bah_ret, 2),
        "win_rate":       round(win_rate, 1),
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   round(max_dd, 2),
        "trade_count":    len(active_trades),
        "win_count":      len(win_trades),
        "final_value":    round(final_port, 2),
        # Son 10 işlem (tablo için)
        "recent_trades":  trades[-10:][::-1],
    }
