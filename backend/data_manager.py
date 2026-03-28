"""
data_manager.py  —  Fin-TAP Backend

RENDER SORUNU ÇÖZÜMÜ:
  yfinance 0.2.50+ "Impersonating chrome136 is not supported" hatası veriyor.
  requirements.txt'te yfinance==0.2.44 sabitlendik → bu hata yok.

  Ek olarak 3 katmanlı indirme stratejisi:
  1. yfinance (requests session ile custom headers)
  2. Yahoo Finance v8 API (requests ile doğrudan)
  3. Yahoo Finance v7 CSV API (son çare)

DATA LEAKAGE ÇÖZÜMÜ:
  Tüm feature'lar price-relative → raw fiyat yok → flat line yok.
  Hedef: log(Close[t+1] / Close[t])
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import time, traceback, json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

import requests

# yfinance — sabit 0.2.44 versiyonu
import yfinance as yf

# Opsiyonel önbellekleme
try:
    import requests_cache
    requests_cache.install_cache(
        "yf_cache", backend="sqlite", expire_after=timedelta(hours=4)
    )
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _ewm(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────
# KATMAN 1: yfinance (0.2.44 stabil)
# ─────────────────────────────────────────────────────────────
def _try_yfinance(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        # yfinance 0.2.44'te requests session ile custom header geçebiliriz
        session = requests.Session()
        session.headers.update(_HEADERS)

        t = yf.Ticker(ticker, session=session)
        df = t.history(start=start, auto_adjust=True)

        if df is not None and not df.empty:
            df.columns = df.columns.str.lower()
            print(f"[data] {ticker}: yfinance OK ({len(df)} satır)")
            return df
    except Exception as e:
        print(f"[data] {ticker}: yfinance hata: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# KATMAN 2: Yahoo Finance v8 API (doğrudan requests)
# ─────────────────────────────────────────────────────────────
def _try_yahoo_v8(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        end_ts   = int(datetime.now().timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={start_ts}&period2={end_ts}&includePrePost=false"
        )

        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            print(f"[data] {ticker}: v8 API boş sonuç")
            return None

        r         = result[0]
        timestamps= r["timestamp"]
        q         = r["indicators"]["quote"][0]
        adj       = r["indicators"].get("adjclose", [{}])[0]

        dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").normalize()

        df = pd.DataFrame({
            "open":   q.get("open",   [None]*len(timestamps)),
            "high":   q.get("high",   [None]*len(timestamps)),
            "low":    q.get("low",    [None]*len(timestamps)),
            "close":  adj.get("adjclose", q.get("close", [None]*len(timestamps))),
            "volume": q.get("volume", [None]*len(timestamps)),
        }, index=dates)

        df = df.dropna()
        if df.empty:
            return None

        print(f"[data] {ticker}: Yahoo v8 API OK ({len(df)} satır)")
        return df

    except Exception as e:
        print(f"[data] {ticker}: v8 API hata: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# KATMAN 3: Yahoo Finance v7 CSV (son çare)
# ─────────────────────────────────────────────────────────────
def _try_yahoo_csv(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        end_ts   = int(datetime.now().timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
        )

        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.lower()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.rename(columns={"adj close": "close"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        if df.empty:
            return None

        print(f"[data] {ticker}: Yahoo CSV API OK ({len(df)} satır)")
        return df

    except Exception as e:
        print(f"[data] {ticker}: CSV API hata: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# ANA İNDİRME FONKSİYONU (3 katmanlı fallback)
# ─────────────────────────────────────────────────────────────
def _download(ticker: str, start: str, retries: int = 2) -> Optional[pd.DataFrame]:
    """3 yöntem dene, birincisi çalışırsa geri kalanını atla."""
    for attempt in range(retries):
        # 1. yfinance
        df = _try_yfinance(ticker, start)
        if df is not None and len(df) > 50:
            return df

        # 2. Yahoo v8 API
        df = _try_yahoo_v8(ticker, start)
        if df is not None and len(df) > 50:
            return df

        # 3. Yahoo CSV
        df = _try_yahoo_csv(ticker, start)
        if df is not None and len(df) > 50:
            return df

        if attempt < retries - 1:
            print(f"[data] {ticker}: tüm yöntemler başarısız, {2}s bekleyip tekrar...")
            time.sleep(2)

    print(f"[data] {ticker}: veri indirilemedi (3 yöntem de başarısız)")
    return None


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────
def get_processed_data(ticker: str, start_date: str = "2018-01-01") -> Optional[pd.DataFrame]:
    """
    OHLCV indir + göreceli feature'ları hesapla.
    Tüm feature'lar price-relative → data leakage yok → flat line yok.
    """
    try:
        df_raw = _download(ticker, start_date)
        if df_raw is None or df_raw.empty:
            return None

        # Sütun adlarını normalize et
        df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]
        df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

        # Gerekli sütunlar
        needed = {"open", "high", "low", "close", "volume"}
        missing = needed - set(df_raw.columns)
        if missing:
            print(f"[data] {ticker}: eksik sütunlar: {missing}")
            return None

        for col in needed:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        df_raw.dropna(subset=list(needed), inplace=True)

        if len(df_raw) < 120:
            print(f"[data] {ticker}: yetersiz satır ({len(df_raw)})")
            return None

        c  = df_raw["close"].astype(float)
        h  = df_raw["high"].astype(float)
        lo = df_raw["low"].astype(float)
        v  = df_raw["volume"].astype(float)
        op = df_raw["open"].astype(float)

        out = pd.DataFrame(index=df_raw.index)

        # ── Log-returns (leakage yok) ────────────────────────────────────
        lr = np.log(c / c.shift(1))
        out["lr_1"]  = lr
        out["lr_2"]  = lr.shift(1)
        out["lr_3"]  = lr.shift(2)
        out["lr_5"]  = np.log(c / c.shift(5))  / 5
        out["lr_10"] = np.log(c / c.shift(10)) / 10
        out["lr_20"] = np.log(c / c.shift(20)) / 20

        # ── RSI ──────────────────────────────────────────────────────────
        for period in [7, 14, 21]:
            d = c.diff()
            g = d.where(d > 0, 0.0).rolling(period).mean()
            l = (-d.where(d < 0, 0.0)).rolling(period).mean()
            out[f"rsi_{period}"] = (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50)
        out["rsi_diff"] = out["rsi_14"] - out["rsi_7"]
        out["rsi_mom"]  = out["rsi_14"].diff(5)

        # ── MACD (price-relative) ────────────────────────────────────────
        macd = (_ewm(c, 12) - _ewm(c, 26)) / c.replace(0, np.nan)
        sig  = _ewm(macd, 9)
        out["macd"]      = macd
        out["macd_sig"]  = sig
        out["macd_hist"] = macd - sig
        out["macd_mom"]  = macd.diff(3)

        # ── Bollinger ────────────────────────────────────────────────────
        s20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
        bbu = s20 + 2*sd20; bbl = s20 - 2*sd20
        out["bb_pct"]   = ((c - bbl) / (bbu - bbl).replace(0, np.nan)).clip(0, 1)
        out["bb_width"] = (bbu - bbl) / s20.replace(0, np.nan)

        # ── SMA/EMA uzaklıkları ──────────────────────────────────────────
        for w in [5, 10, 20, 50, 100]:
            sm = c.rolling(w).mean()
            out[f"dist_sma{w}"] = (c - sm) / sm.replace(0, np.nan)
        for span in [9, 21]:
            em = _ewm(c, span)
            out[f"dist_ema{span}"] = (c - em) / em.replace(0, np.nan)

        # ── Volatilite ───────────────────────────────────────────────────
        for w in [5, 10, 20]:
            out[f"vol_{w}d"] = lr.rolling(w).std()
        out["rvol_20"]   = lr.rolling(20).std() * np.sqrt(252)
        out["vol_ratio"] = out["vol_10d"] / out["vol_20d"].replace(0, np.nan)

        # ── ATR ──────────────────────────────────────────────────────────
        pc = c.shift(1)
        tr = pd.concat([h-lo, (h-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        out["atr_pct"]   = atr14 / c.replace(0, np.nan)
        out["atr_trend"] = atr14 / atr14.rolling(14).mean().replace(0, np.nan)

        # ── Stochastic ───────────────────────────────────────────────────
        ll14 = lo.rolling(14).min(); hh14 = h.rolling(14).max()
        stk  = (100*(c-ll14)/(hh14-ll14).replace(0, np.nan)).fillna(50)
        out["stoch_k"]    = stk
        out["stoch_d"]    = stk.rolling(3).mean()
        out["stoch_diff"] = stk - out["stoch_d"]

        # ── Williams %R ──────────────────────────────────────────────────
        out["willr"] = (-100*(hh14-c)/(hh14-ll14).replace(0, np.nan)).fillna(-50)

        # ── CCI ──────────────────────────────────────────────────────────
        tp    = (h + lo + c) / 3
        tp_ma = tp.rolling(20).mean()
        tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        out["cci"] = ((tp-tp_ma) / (0.015*tp_md.replace(0, np.nan))).clip(-300, 300).fillna(0)

        # ── ADX ──────────────────────────────────────────────────────────
        pdm = h.diff().clip(lower=0); mdm = (-lo.diff()).clip(lower=0)
        tr14 = tr.rolling(14).mean().replace(0, np.nan)
        out["adx_plus"]  = (100*pdm.rolling(14).mean()/tr14).fillna(0)
        out["adx_minus"] = (100*mdm.rolling(14).mean()/tr14).fillna(0)
        out["adx_diff"]  = out["adx_plus"] - out["adx_minus"]

        # ── ROC ──────────────────────────────────────────────────────────
        for w in [3, 5, 10, 20]:
            out[f"roc_{w}"] = c.pct_change(w) * 100

        # ── Hacim ────────────────────────────────────────────────────────
        vsma = v.rolling(14).mean()
        out["v_ratio"] = (v / vsma.replace(0, np.nan)).clip(0, 10)
        out["v_trend"] = (vsma / v.rolling(50).mean().replace(0, np.nan)).clip(0, 5)
        out["pv_corr"] = (lr * (v/vsma.replace(0, np.nan))).clip(-5, 5)

        # ── Candle pattern ───────────────────────────────────────────────
        out["hl_pct"]     = (h - lo) / c.replace(0, np.nan)
        out["open_close"] = (c - op) / c.replace(0, np.nan)

        # ── 52 haftalık ──────────────────────────────────────────────────
        out["dist_52w_high"] = (c - c.rolling(252).max()) / c.replace(0, np.nan)
        out["dist_52w_low"]  = (c - c.rolling(252).min()) / c.replace(0, np.nan)

        # ── Eğim ─────────────────────────────────────────────────────────
        out["sma20_slope"] = s20.diff(5) / s20.shift(5).replace(0, np.nan)
        s50 = c.rolling(50).mean()
        out["sma50_slope"] = s50.diff(5) / s50.shift(5).replace(0, np.nan)

        # HEDEF
        out["target_lr"] = lr.shift(-1)

        # Chart için raw fiyat (feature değil)
        out["Close"]  = c
        out["High"]   = h
        out["Low"]    = lo
        out["Volume"] = v

        out.dropna(inplace=True)

        if out.empty or len(out) < 60:
            print(f"[data] {ticker}: dropna sonrası yetersiz ({len(out)})")
            return None

        print(f"[data] {ticker}: {len(out)} satır, {len(out.columns)} kolon OK")
        return out

    except Exception as e:
        print(f"[data] {ticker} HATA: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# Feature grupları
# ─────────────────────────────────────────────────────────────
FEATURE_GROUPS: dict = {
    "Returns":    ["lr_1","lr_2","lr_3","lr_5","lr_10","lr_20"],
    "RSI":        ["rsi_7","rsi_14","rsi_21","rsi_diff","rsi_mom"],
    "MACD":       ["macd","macd_sig","macd_hist","macd_mom"],
    "Bollinger":  ["bb_pct","bb_width"],
    "SMA":        ["dist_sma5","dist_sma10","dist_sma20","dist_sma50",
                   "dist_sma100","dist_ema9","dist_ema21"],
    "EMA":        ["dist_ema9","dist_ema21"],
    "Volatility": ["vol_5d","vol_10d","vol_20d","rvol_20","vol_ratio"],
    "ATR":        ["atr_pct","atr_trend"],
    "Stoch":      ["stoch_k","stoch_d","stoch_diff"],
    "Williams":   ["willr"],
    "CCI":        ["cci"],
    "ADX":        ["adx_plus","adx_minus","adx_diff"],
    "Momentum":   ["roc_3","roc_5","roc_10","roc_20"],
    "Volume":     ["v_ratio","v_trend","pv_corr"],
    "Pattern":    ["hl_pct","open_close"],
    "Distance":   ["dist_52w_high","dist_52w_low"],
    "Trend":      ["sma20_slope","sma50_slope"],
}

DEFAULT_GROUPS = ["Returns","RSI","MACD","Bollinger","SMA","Volatility","ATR","Momentum"]
