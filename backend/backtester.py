"""
backtester.py  —  Fin-TAP v3.0
================================
Backtest motoru: RSI + hareketli ortalama (SMA) sinyalini geçmiş veriye uygular
ve strateji performansını ölçer.

Ne döndürür?
  - P&L (kâr/zarar) eğrisi
  - Win rate (kazanan işlem oranı)
  - Max drawdown (en derin düşüş yüzdesi)
  - Sharpe ratio (risk-ayarlı getiri)
  - Buy-and-Hold karşılaştırması

Strateji mantığı:
  Geçmiş veri üzerinde her "horizon" günde bir sinyal üretilir.
  'long'  → al ve horizon gün sonra sat
  'short' → allow_short=True ise açığa sat (aksi hâlde işlem yapma)
  'neutral' → bu periyotta işlem yok

Finans terimleri sözlüğü (hocana anlatırken):
  RSI (Relative Strength Index) → 0-100 arası momentum göstergesi;
        <30 aşırı satım (ucuz), >70 aşırı alım (pahalı) işareti.
  SMA (Simple Moving Average)   → son N günün kapanış ortalaması.
  Drawdown                      → tepe noktasından mevcut değere düşüş oranı.
  Sharpe Ratio                  → birim risk başına ne kadar kazandık?
                                  >1 iyi, >2 çok iyi kabul edilir.
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
    idx pozisyonundaki gün için RSI(14) + SMA(20/50) trend sinyali hesapla.

    Karar mantığı:
      Bullish (long):  RSI < 55 VE fiyat > SMA20 VE SMA20 > SMA50 → trend yukarı
      Bearish (short): RSI > 65 VEYA fiyat < SMA20 < SMA50 → trend aşağı
      Geri kalanlar: neutral

    Döndürür: 'long' | 'short' | 'neutral'
    """
    # Bu güne kadar olan son 60 günlük pencereyi al
    window = df.iloc[max(0, idx - 60): idx + 1]
    if len(window) < 15:
        return "neutral"

    closes = window["Close"].values.astype(float)

    # RSI(14) hesapla — son 14 günün kazanç/kayıp ortalaması
    diffs  = np.diff(closes[-15:])
    gains  = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_g  = np.mean(gains)  if gains.any()  else 0.0
    avg_l  = np.mean(losses) if losses.any() else 1e-9   # sıfıra bölmeyi engelle
    rsi    = 100 - 100 / (1 + avg_g / avg_l)

    # Son 20 ve 50 günün basit ortalaması
    sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
    sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else closes[-1]
    cur   = closes[-1]

    # Trendin yönünü belirle
    bullish = rsi < 55 and cur > sma20 and sma20 > sma50   # yukarı trend
    bearish = rsi > 65 or cur < sma20 < sma50              # aşağı trend

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
    Geçmiş veri üzerinde strateji simülasyonu yap ve sonuçları döndür.

    Parametreler
    ------------
    ticker         : hisse/kripto sembolü (örn. 'AAPL', 'BTC-USD')
    horizon        : sinyal başına kaç günlük pozisyon tut (7 / 14 / 30)
    lookback_days  : kaç günlük geçmişe bak (varsayılan: 1 yıl)
    start_capital  : başlangıç sermayesi USD cinsinden
    allow_short    : True → short sinyallerde de işlem yap, False → sadece long

    Döndürür: istatistik ve grafik verilerini içeren dict | None (veri yoksa)
    """
    df = get_processed_data(ticker)
    if df is None or df.empty:
        return None

    # Test için yeterli geçmiş veri olması gerekiyor
    df = df.tail(lookback_days + horizon + 60).copy()
    required_cols = [c for c in ("Close", "High", "Low", "Volume") if c in df.columns]
    if "Close" not in required_cols:
        return None
    df = df.dropna(subset=required_cols)

    if len(df) < horizon * 3:
        return None

    closes = df["Close"].values.astype(float)
    dates  = [d.strftime("%Y-%m-%d") for d in df.index]
    n      = len(closes)

    # ── Strateji simülasyonu ─────────────────────────────────────────────────
    current_val = start_capital   # strateji portföy değeri (her işlemde güncellenir)
    trades      = []              # tüm işlem kayıtları

    # Her horizon günde bir sinyal üret (ilk 60 gün ısınma periyodu)
    step          = max(1, horizon)
    check_indices = list(range(60, n - horizon, step))

    for i in check_indices:
        signal = _compute_signal(df, i)
        entry  = closes[i]
        exit_i = min(i + horizon, n - 1)   # horizon gün sonrasında çık
        exit_p = closes[exit_i]

        # Gerçekleşen ham getiri (%)
        raw_ret = (exit_p - entry) / entry

        if signal == "long":
            trade_ret = raw_ret           # alım yaptık → fiyat artışı kâr
            won = raw_ret > 0
        elif signal == "short" and allow_short:
            trade_ret = -raw_ret          # açığa satış → fiyat düşüşü kâr
            won = raw_ret < 0
        else:
            trade_ret = 0.0               # neutral veya short izni yok → bekle
            won = None

        current_val *= (1 + trade_ret)   # bileşik büyüme

        trades.append({
            "date":       dates[i],
            "signal":     signal,
            "entry":      round(float(entry), 2),
            "exit":       round(float(exit_p), 2),
            "pct_return": round(trade_ret * 100, 2),
            "won":        won,
        })

    # ── Günlük değer serisi (grafik için) ────────────────────────────────────
    # Strateji portföyü ve Buy-and-Hold portföyü günlük olarak izlenir
    port_series = [start_capital]
    bah_series  = [start_capital]
    date_series = [dates[60]]

    running   = start_capital
    # İşlem günlerini hızlı aramak için sözlük
    trade_map = {t["date"]: t["pct_return"] for t in trades}

    for idx in range(61, n):
        d = dates[idx]
        if d in trade_map:
            running *= (1 + trade_map[d] / 100)
        port_series.append(round(running, 2))
        # Buy-and-Hold: başlangıçtaki fiyata göre orantılı büyüme
        bah_series.append(round(closes[idx] / closes[60] * start_capital, 2))
        date_series.append(d)

    # ── İstatistikler ────────────────────────────────────────────────────────
    # Sadece gerçekten işlem yapılan günler (neutral hariç)
    active_trades = [t for t in trades if t["won"] is not None]
    win_trades    = [t for t in active_trades if t["won"]]
    win_rate      = len(win_trades) / len(active_trades) * 100 if active_trades else 0.0

    # Sharpe Ratio: (ortalama getiri / getiri std sapması) × √(yıllık işlem sayısı)
    returns_arr = np.array([t["pct_return"] for t in active_trades]) / 100
    sharpe = 0.0
    if len(returns_arr) > 1 and returns_arr.std() > 0:
        sharpe = float((returns_arr.mean() / returns_arr.std()) * np.sqrt(252 / horizon))

    # Max Drawdown: tepe değerden en derin düşüş yüzdesi
    peak     = start_capital
    max_dd   = 0.0
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
        # Grafik verisi — her 2 günde bir nokta alınır (veri boyutunu küçültmek için)
        "dates":          date_series[::2],
        "portfolio":      port_series[::2],
        "buy_and_hold":   bah_series[::2],
        # İstatistikler
        "total_return":   round(total_ret, 2),    # strateji toplam getirisi (%)
        "bah_return":     round(bah_ret, 2),      # buy-and-hold toplam getirisi (%)
        "win_rate":       round(win_rate, 1),      # kazanan işlem oranı (%)
        "sharpe":         round(sharpe, 2),        # Sharpe ratio
        "max_drawdown":   round(max_dd, 2),        # maksimum düşüş (%)
        "trade_count":    len(active_trades),
        "win_count":      len(win_trades),
        "final_value":    round(final_port, 2),    # son portföy değeri (USD)
        # Son 10 işlem tablosu için (ters sırayla: en yeni önce)
        "recent_trades":  trades[-10:][::-1],
    }
