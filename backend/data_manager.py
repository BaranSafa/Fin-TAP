"""
data_manager.py  —  Fin-TAP Backend
======================================
Bu modülün görevi: Yahoo Finance'tan ham fiyat verisi çekmek ve
makine öğrenmesi modelleri için teknik gösterge (feature) hesaplamak.

── VERİ AKIŞI ──────────────────────────────────────────────────────────────
  get_processed_data(ticker)
       ↓
  1. Cache kontrolü → taze ise hemen döndür
       ↓ (bayat ise)
  2. _download_raw() → 3 katmanlı indirme dener:
       a) yfinance (curl_cffi ile bot koruması aşılır)
       b) Yahoo Finance v8 API (REST)
       c) Yahoo Finance v7 CSV indir
       ↓ (başarılı)
  3. _features() → ~50 teknik gösterge hesapla
       ↓
  4. Cache'e kaydet, DataFrame döndür

── OTOMATİK GÜNCELLEME (IN-MEMORY CACHE) ──────────────────────────────────
  Borsa açıkken:    1 saat cache → her saatte yeni veri
  Borsa kapalıyken: 6 saat cache → gereksiz API çağrısı yok
  Render restart   → cache sıfırlanır → ilk istekte taze veri gelir
  force_refresh=True → cache'i atla, hemen güncelle

── requests_cache SQLite KALDIRILDI ────────────────────────────────────────
  Render free tier disk persist etmiyor → eski veri sunuyordu

── TEKNİK GÖSTERGELER SÖZLÜĞÜ ─────────────────────────────────────────────
  lr_N   : N günlük logaritmik getiri (fiyat değişiminin daha stabil ölçümü)
  RSI    : Relative Strength Index — momentum göstergesi (0-100)
  MACD   : Moving Average Convergence/Divergence — trend takip göstergesi
  BB     : Bollinger Bantları — fiyatın hareketli ortalamaya göre konumu
  SMA/EMA: Basit / Üstel Hareketli Ortalama
  ATR    : Average True Range — volatilite (oynaklık) ölçümü
  Stoch  : Stochastic Oscillator — aşırı alım/satım tespiti
  CCI    : Commodity Channel Index — trend gücü
  ADX    : Average Directional Index — trend kuvveti
  ROC    : Rate of Change — fiyat momentum hızı
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import time, traceback
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ── IN-MEMORY CACHE ──────────────────────────────────────────────────────────
_MEM_CACHE: dict = {}   # {ticker: {"df": DataFrame, "at": datetime}}


def _market_open() -> bool:
    """
    NYSE/NASDAQ şu an açık mı? (America/New_York saat dilimi — yaz/kış saati dahil)
    Hafta sonu → False. 09:30-16:00 ET aralığında → True.
    """
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:       # hafta sonu
            return False
        t = now_et.hour * 60 + now_et.minute
        return 570 <= t < 960           # 09:30 = 570 dk, 16:00 = 960 dk
    except Exception:
        # zoneinfo yoksa (Python < 3.9) UTC fallback
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False
        return 14 <= now.hour < 21      # 09:30–16:00 EST ≈ 14:30–21:00 UTC


def _ttl() -> int:
    """Cache geçerlilik süresi: borsa açıkken 1 saat, kapalıyken 6 saat."""
    return 3600 if _market_open() else 6 * 3600


def cache_clear(ticker: str | None = None):
    if ticker:
        _MEM_CACHE.pop(ticker, None)
    else:
        _MEM_CACHE.clear()
    print(f"[cache] {'Tümü' if not ticker else ticker} temizlendi")


def cache_status() -> dict:
    now = datetime.utcnow()
    return {
        t: {
            "age_min":   round((now - e["at"]).total_seconds() / 60, 1),
            "rows":      len(e["df"]),
            "fresh":     (now - e["at"]).total_seconds() < _ttl(),
            "last_date": str(e["df"].index[-1].date()),
        }
        for t, e in _MEM_CACHE.items()
    }


# ── YARDIMCI ─────────────────────────────────────────────────────────────────
_HDRS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*",
}


def _ewm(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


# ── 3 KATMANLI İNDİRME ───────────────────────────────────────────────────────
def _yf(ticker: str, start: str) -> Optional[pd.DataFrame]:
    """
    curl_cffi ile yfinance — chrome impersonation hatası yok.
    curl_cffi==0.7.4 + yfinance>=0.2.50 gerekli (requirements.txt).
    """
    try:
        # curl_cffi session — Yahoo Finance'ın bot engelini geçer
        try:
            from curl_cffi import requests as cffi_requests
            sess = cffi_requests.Session(impersonate="chrome")
        except Exception:
            # curl_cffi yoksa normal requests session dene
            sess = requests.Session()
            sess.headers.update(_HDRS)

        t  = yf.Ticker(ticker, session=sess)
        df = t.history(start=start, auto_adjust=True)

        if df is not None and not df.empty:
            df.columns = df.columns.str.lower()
            print(f"[data] {ticker}: yfinance OK ({len(df)}r, son:{df.index[-1].date()})")
            return df
    except Exception as e:
        print(f"[data] {ticker}: yfinance err: {e}")
    return None


def _v8(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        s  = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        e  = int(datetime.utcnow().timestamp())
        r  = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={s}&period2={e}",
            headers=_HDRS, timeout=20)
        r.raise_for_status()
        res = r.json().get("chart", {}).get("result", [])
        if not res:
            return None
        d  = res[0]; ts = d["timestamp"]
        q  = d["indicators"]["quote"][0]
        ac = d["indicators"].get("adjclose", [{}])[0]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize()
        df = pd.DataFrame({
            "open": q.get("open"), "high": q.get("high"),
            "low":  q.get("low"),
            "close": ac.get("adjclose") or q.get("close"),
            "volume": q.get("volume"),
        }, index=idx).dropna()
        if not df.empty:
            print(f"[data] {ticker}: v8 OK ({len(df)}r, son:{df.index[-1].date()})")
        return df or None
    except Exception as e:
        print(f"[data] {ticker}: v8 err: {e}")
    return None


def _csv(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        s = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        e = int(datetime.utcnow().timestamp())
        r = requests.get(
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={s}&period2={e}&interval=1d&events=history",
            headers=_HDRS, timeout=20)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.lower()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "close"})
        df = df[["open","high","low","close","volume"]].dropna()
        if not df.empty:
            print(f"[data] {ticker}: csv OK ({len(df)}r, son:{df.index[-1].date()})")
        return df or None
    except Exception as e:
        print(f"[data] {ticker}: csv err: {e}")
    return None


def _download_raw(ticker: str, start: str) -> Optional[pd.DataFrame]:
    """
    Üç farklı yöntemi sırayla dener; biri başarılı olursa hemen döndürür.
    Hepsi başarısız olursa 2 saniye bekleyip bir kez daha dener.
    50'den az satır gelen veriyi geçersiz kabul eder (ML için yetersiz).
    """
    for attempt in range(2):
        for fn in (_yf, _v8, _csv):
            df = fn(ticker, start)
            if df is not None and len(df) > 50:
                return df
        if attempt == 0:
            print(f"[data] {ticker}: tüm yöntemler başarısız, 2s bekleniyor...")
            time.sleep(2)
    return None


# ── FEATURE HESAPLAMA ─────────────────────────────────────────────────────────
def _features(df_raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Ham OHLCV verisinden ~50 teknik gösterge hesaplar.
    ML modeli bu göstergeleri "özellik" olarak kullanır.

    Çıktı sütunları:
      lr_1..lr_20   : 1-20 günlük logaritmik getiriler
      rsi_7/14/21   : farklı periyotlarda RSI
      macd*         : MACD, sinyal çizgisi, histogram
      bb_*          : Bollinger Bant pozisyonu ve genişliği
      dist_sma*/ema*: fiyatın hareketli ortalamalardan uzaklığı
      vol_*         : gerçekleşmiş volatilite (5/10/20 gün)
      atr_*         : Average True Range
      stoch_*       : Stochastic Oscillator
      willr         : Williams %R
      cci           : Commodity Channel Index
      adx_*         : Directional Movement göstergeleri
      roc_*         : Rate of Change momentum
      v_*           : hacim göstergeleri
      target_lr     : ertesi gün logaritmik getiri (modelin tahmin hedefi)
    """
    try:
        df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
        df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

        needed = {"open","high","low","close","volume"}
        if not needed.issubset(set(df_raw.columns)):
            return None
        for col in needed:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        df_raw.dropna(subset=list(needed), inplace=True)
        if len(df_raw) < 120:
            return None

        c = df_raw["close"].astype(float)
        h = df_raw["high"].astype(float)
        lo = df_raw["low"].astype(float)
        v  = df_raw["volume"].astype(float)
        op = df_raw["open"].astype(float)
        out = pd.DataFrame(index=df_raw.index)

        lr = np.log(c / c.shift(1))
        out["lr_1"]  = lr;          out["lr_2"]  = lr.shift(1)
        out["lr_3"]  = lr.shift(2); out["lr_5"]  = np.log(c/c.shift(5))/5
        out["lr_10"] = np.log(c/c.shift(10))/10
        out["lr_20"] = np.log(c/c.shift(20))/20

        for p in [7, 14, 21]:
            d = c.diff(); g = d.where(d>0,0.).rolling(p).mean()
            l = (-d.where(d<0,0.)).rolling(p).mean()
            out[f"rsi_{p}"] = (100 - 100/(1+g/l.replace(0,np.nan))).fillna(50)
        out["rsi_diff"] = out["rsi_14"] - out["rsi_7"]
        out["rsi_mom"]  = out["rsi_14"].diff(5)

        macd = (_ewm(c,12)-_ewm(c,26))/c.replace(0,np.nan); sig = _ewm(macd,9)
        out["macd"] = macd; out["macd_sig"] = sig
        out["macd_hist"] = macd-sig; out["macd_mom"] = macd.diff(3)

        s20=c.rolling(20).mean(); sd20=c.rolling(20).std()
        bbu=s20+2*sd20; bbl=s20-2*sd20
        out["bb_pct"]   = ((c-bbl)/(bbu-bbl).replace(0,np.nan)).clip(0,1)
        out["bb_width"] = (bbu-bbl)/s20.replace(0,np.nan)

        for w in [5,10,20,50,100]:
            sm=c.rolling(w).mean(); out[f"dist_sma{w}"]=(c-sm)/sm.replace(0,np.nan)
        for sp in [9,21]:
            em=_ewm(c,sp); out[f"dist_ema{sp}"]=(c-em)/em.replace(0,np.nan)

        for w in [5,10,20]: out[f"vol_{w}d"]=lr.rolling(w).std()
        out["rvol_20"]  = lr.rolling(20).std()*np.sqrt(252)
        out["vol_ratio"]= out["vol_10d"]/out["vol_20d"].replace(0,np.nan)

        pc=c.shift(1); tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
        atr14=tr.rolling(14).mean()
        out["atr_pct"]  =atr14/c.replace(0,np.nan)
        out["atr_trend"]=atr14/atr14.rolling(14).mean().replace(0,np.nan)

        ll14=lo.rolling(14).min(); hh14=h.rolling(14).max()
        stk=(100*(c-ll14)/(hh14-ll14).replace(0,np.nan)).fillna(50)
        out["stoch_k"]=stk; out["stoch_d"]=stk.rolling(3).mean()
        out["stoch_diff"]=stk-out["stoch_d"]
        out["willr"]=(-100*(hh14-c)/(hh14-ll14).replace(0,np.nan)).fillna(-50)

        tp=(h+lo+c)/3; tp_ma=tp.rolling(20).mean()
        tp_md=tp.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)
        out["cci"]=((tp-tp_ma)/(0.015*tp_md.replace(0,np.nan))).clip(-300,300).fillna(0)

        pdm=h.diff().clip(lower=0); mdm=(-lo.diff()).clip(lower=0)
        tr14=tr.rolling(14).mean().replace(0,np.nan)
        out["adx_plus"] =(100*pdm.rolling(14).mean()/tr14).fillna(0)
        out["adx_minus"]=(100*mdm.rolling(14).mean()/tr14).fillna(0)
        out["adx_diff"] =out["adx_plus"]-out["adx_minus"]

        for w in [3,5,10,20]: out[f"roc_{w}"]=c.pct_change(w)*100

        vsma=v.rolling(14).mean()
        out["v_ratio"]=(v/vsma.replace(0,np.nan)).clip(0,10)
        out["v_trend"]=(vsma/v.rolling(50).mean().replace(0,np.nan)).clip(0,5)
        out["pv_corr"]=(lr*(v/vsma.replace(0,np.nan))).clip(-5,5)

        out["hl_pct"]    =(h-lo)/c.replace(0,np.nan)
        out["open_close"]=(c-op)/c.replace(0,np.nan)
        out["dist_52w_high"]=(c-c.rolling(252).max())/c.replace(0,np.nan)
        out["dist_52w_low"] =(c-c.rolling(252).min())/c.replace(0,np.nan)

        out["sma20_slope"]=s20.diff(5)/s20.shift(5).replace(0,np.nan)
        s50=c.rolling(50).mean()
        out["sma50_slope"]=s50.diff(5)/s50.shift(5).replace(0,np.nan)

        out["target_lr"]=lr.shift(-1)
        out["Close"]=c; out["High"]=h; out["Low"]=lo; out["Volume"]=v

        out.dropna(inplace=True)
        if out.empty or len(out) < 60:
            return None
        print(f"[data] {ticker}: {len(out)} satır, son: {out.index[-1].date()} OK")
        return out
    except Exception as e:
        print(f"[data] {ticker} feature err: {e}"); traceback.print_exc(); return None


# ── ANA FONKSİYON ────────────────────────────────────────────────────────────
def get_processed_data(
    ticker: str,
    start_date: str = "2018-01-01",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Güncel veri döndürür.
    - Borsa açık → 1 saat cache
    - Borsa kapalı → 6 saat cache
    - force_refresh=True → cache'i atla
    """
    now = datetime.utcnow()
    ttl = _ttl()

    if not force_refresh and ticker in _MEM_CACHE:
        entry = _MEM_CACHE[ticker]
        age   = (now - entry["at"]).total_seconds()
        if age < ttl:
            print(f"[data] {ticker}: cache HIT ({age/60:.0f}dk, son:{entry['df'].index[-1].date()})")
            return entry["df"]
        print(f"[data] {ticker}: cache STALE ({age/3600:.1f}s), yenileniyor...")

    df_raw = _download_raw(ticker, start_date)
    if df_raw is None:
        # İndirme başarısız — eski cache daha iyi
        if ticker in _MEM_CACHE:
            old = _MEM_CACHE[ticker]
            print(f"[data] {ticker}: indirme başarısız, eski cache ({(now-old['at']).total_seconds()/3600:.1f}s) kullanılıyor")
            return old["df"]
        return None

    df = _features(df_raw, ticker)
    if df is None:
        return None

    _MEM_CACHE[ticker] = {"df": df, "at": now}
    return df


# ── FEATURE GRUPLARI ─────────────────────────────────────────────────────────
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

DEFAULT_GROUPS = [
    "Returns","RSI","MACD","Bollinger","SMA","Volatility","ATR","Momentum"
]
