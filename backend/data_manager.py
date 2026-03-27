"""
data_manager.py  —  Fin-TAP Backend
Düzeltilen TEMEL SORUN (FLAT LINE):
  Önceki versiyonda Close, lag_1 gibi raw fiyat değerleri feature olarak
  kullanılıyordu. Model y = Close[t+1] tahmin ederken trivially
  Close[t] ≈ Close[t+1] öğreniyordu → BACKTEST DÜZLÜK.

  ÇÖZÜM: Tüm feature'lar göreceli/normalise edildi (raw fiyat YOK).
  Hedef: log(Close[t+1] / Close[t])  →  return tahmin edip sonra
         fiyata çeviriyoruz: price = base_price * exp(predicted_return)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def _ewm(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def get_processed_data(ticker: str, start_date: str = "2017-01-01") -> pd.DataFrame | None:
    try:
        df_raw = yf.download(
            ticker, start=start_date,
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if df_raw is None or df_raw.empty:
            print(f"[data] {ticker}: boş veri.")
            return None

        # MultiIndex flatten
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [str(c[0]).strip() for c in df_raw.columns]
        else:
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = df_raw.columns.str.lower()
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]
        df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df_raw.columns)):
            print(f"[data] {ticker}: eksik sütunlar: {needed - set(df_raw.columns)}")
            return None

        for col in needed:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        df_raw.dropna(subset=list(needed), inplace=True)

        if len(df_raw) < 120:
            print(f"[data] {ticker}: yetersiz satır ({len(df_raw)}).")
            return None

        c  = df_raw["close"].astype(float)
        h  = df_raw["high"].astype(float)
        lo = df_raw["low"].astype(float)
        v  = df_raw["volume"].astype(float)
        op = df_raw["open"].astype(float)

        out = pd.DataFrame(index=df_raw.index)

        # ── Log-returns (gecikmeli) ──────────────────────────────────────
        lr = np.log(c / c.shift(1))
        out["lr_1"]  = lr
        out["lr_2"]  = lr.shift(1)
        out["lr_3"]  = lr.shift(2)
        out["lr_5"]  = np.log(c / c.shift(5))  / 5
        out["lr_10"] = np.log(c / c.shift(10)) / 10
        out["lr_20"] = np.log(c / c.shift(20)) / 20

        # ── RSI ──────────────────────────────────────────────────────────
        for period in [7, 14, 21]:
            delta = c.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs    = gain / loss.replace(0, np.nan)
            out[f"rsi_{period}"] = (100 - 100 / (1 + rs)).fillna(50)

        out["rsi_diff_7_14"] = out["rsi_14"] - out["rsi_7"]
        out["rsi_mom"]       = out["rsi_14"].diff(5)

        # ── MACD (price-relative → no leakage) ───────────────────────────
        ema12 = _ewm(c, 12)
        ema26 = _ewm(c, 26)
        macd  = (ema12 - ema26) / c.replace(0, np.nan)
        sig   = _ewm(macd, 9)
        out["macd"]      = macd
        out["macd_sig"]  = sig
        out["macd_hist"] = macd - sig
        out["macd_mom"]  = macd.diff(3)

        # ── Bollinger (göreceli) ──────────────────────────────────────────
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_u  = sma20 + 2 * std20
        bb_l  = sma20 - 2 * std20
        denom = (bb_u - bb_l).replace(0, np.nan)
        out["bb_pct"]   = ((c - bb_l) / denom).clip(0, 1)
        out["bb_width"] = ((bb_u - bb_l) / sma20.replace(0, np.nan))

        # ── SMA uzaklıkları (price-relative) ─────────────────────────────
        for w in [5, 10, 20, 50, 100]:
            sma = c.rolling(w).mean()
            out[f"dist_sma{w}"] = (c - sma) / sma.replace(0, np.nan)

        for span in [9, 21, 50]:
            ema = _ewm(c, span)
            out[f"dist_ema{span}"] = (c - ema) / ema.replace(0, np.nan)

        # ── Volatilite ────────────────────────────────────────────────────
        for w in [5, 10, 20]:
            out[f"vol_{w}d"] = lr.rolling(w).std()
        out["rvol_20"] = lr.rolling(20).std() * np.sqrt(252)
        # Volatilite değişim hızı
        out["vol_ratio"] = out["vol_10d"] / out["vol_20d"].replace(0, np.nan)

        # ── ATR (price-relative) ──────────────────────────────────────────
        prev_c = c.shift(1)
        tr = pd.concat([
            h - lo,
            (h - prev_c).abs(),
            (lo - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        out["atr_pct"]   = atr14 / c.replace(0, np.nan)
        out["atr_trend"] = atr14 / atr14.rolling(14).mean().replace(0, np.nan)

        # ── Stochastic ────────────────────────────────────────────────────
        ll14 = lo.rolling(14).min()
        hh14 = h.rolling(14).max()
        stk  = (100 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)).fillna(50)
        out["stoch_k"]    = stk
        out["stoch_d"]    = stk.rolling(3).mean()
        out["stoch_diff"] = stk - out["stoch_d"]

        # ── Williams %R ───────────────────────────────────────────────────
        out["willr"] = (-100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)).fillna(-50)

        # ── CCI ───────────────────────────────────────────────────────────
        tp    = (h + lo + c) / 3
        tp_ma = tp.rolling(20).mean()
        tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        out["cci"] = ((tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan))).clip(-300, 300).fillna(0)

        # ── ADX ───────────────────────────────────────────────────────────
        pdm  = h.diff().clip(lower=0)
        mdm  = (-lo.diff()).clip(lower=0)
        tr14 = tr.rolling(14).mean().replace(0, np.nan)
        out["adx_plus"]  = (100 * pdm.rolling(14).mean() / tr14).fillna(0)
        out["adx_minus"] = (100 * mdm.rolling(14).mean() / tr14).fillna(0)
        out["adx_diff"]  = out["adx_plus"] - out["adx_minus"]

        # ── ROC ───────────────────────────────────────────────────────────
        for w in [3, 5, 10, 20]:
            out[f"roc_{w}"] = c.pct_change(w) * 100

        # ── Hacim (göreceli) ──────────────────────────────────────────────
        vsma14 = v.rolling(14).mean()
        vsma50 = v.rolling(50).mean()
        out["v_ratio"]  = (v / vsma14.replace(0, np.nan)).clip(0, 10)
        out["v_trend"]  = (vsma14 / vsma50.replace(0, np.nan)).clip(0, 5)
        # Fiyat-hacim baskısı (OBV türevi)
        out["pv_corr"]  = (lr * (v / vsma14.replace(0, np.nan))).clip(-5, 5)

        # ── Fiyat pattern (candle) ────────────────────────────────────────
        out["hl_pct"]     = (h - lo) / c.replace(0, np.nan)
        out["open_close"] = (c - op) / c.replace(0, np.nan)
        out["upper_wick"] = (h - c.clip(upper=h)) / c.replace(0, np.nan)
        out["lower_wick"] = (c.clip(lower=lo) - lo) / c.replace(0, np.nan)

        # ── 52 haftalık uzaklık ───────────────────────────────────────────
        out["dist_52w_high"] = (c - c.rolling(252).max()) / c.replace(0, np.nan)
        out["dist_52w_low"]  = (c - c.rolling(252).min()) / c.replace(0, np.nan)

        # ── SMA eğimi ─────────────────────────────────────────────────────
        sma50 = c.rolling(50).mean()
        out["sma50_slope"] = sma50.diff(5) / sma50.shift(5).replace(0, np.nan)
        out["sma20_slope"] = sma20.diff(5) / sma20.shift(5).replace(0, np.nan)

        # ── HEDEF ─────────────────────────────────────────────────────────
        out["target_lr"] = lr.shift(-1)   # sonraki günün log-return'ü

        # ── Raw fiyat (chart için, feature DEĞİL) ─────────────────────────
        out["Close"]  = c
        out["High"]   = h
        out["Low"]    = lo
        out["Volume"] = v

        # NaN temizliği
        out.dropna(inplace=True)

        if out.empty or len(out) < 60:
            print(f"[data] {ticker}: dropna sonrası yetersiz ({len(out)}).")
            return None

        print(f"[data] {ticker}: {len(out)} satır, {len(out.columns)} kolon.")
        return out

    except Exception as e:
        import traceback
        print(f"[data] {ticker} HATA: {e}")
        traceback.print_exc()
        return None


# Feature grupları (UI için — Close/High/Low/Volume/target_lr hariç)
FEATURE_COLS = [
    # returns
    "lr_1","lr_2","lr_3","lr_5","lr_10","lr_20",
    # rsi
    "rsi_7","rsi_14","rsi_21","rsi_diff_7_14","rsi_mom",
    # macd
    "macd","macd_sig","macd_hist","macd_mom",
    # bollinger
    "bb_pct","bb_width",
    # sma/ema
    "dist_sma5","dist_sma10","dist_sma20","dist_sma50","dist_sma100",
    "dist_ema9","dist_ema21","dist_ema50",
    # volatility
    "vol_5d","vol_10d","vol_20d","rvol_20","vol_ratio",
    # atr
    "atr_pct","atr_trend",
    # stoch
    "stoch_k","stoch_d","stoch_diff",
    # willr
    "willr",
    # cci
    "cci",
    # adx
    "adx_plus","adx_minus","adx_diff",
    # roc
    "roc_3","roc_5","roc_10","roc_20",
    # volume
    "v_ratio","v_trend","pv_corr",
    # pattern
    "hl_pct","open_close","upper_wick","lower_wick",
    # distance
    "dist_52w_high","dist_52w_low",
    # slope
    "sma50_slope","sma20_slope",
]

FEATURE_GROUPS = {
    "Returns":    ["lr_1","lr_2","lr_3","lr_5","lr_10","lr_20"],
    "RSI":        ["rsi_7","rsi_14","rsi_21","rsi_diff_7_14","rsi_mom"],
    "MACD":       ["macd","macd_sig","macd_hist","macd_mom"],
    "Bollinger":  ["bb_pct","bb_width"],
    "SMA":        ["dist_sma5","dist_sma10","dist_sma20","dist_sma50","dist_sma100",
                   "dist_ema9","dist_ema21","dist_ema50"],
    "Volatility": ["vol_5d","vol_10d","vol_20d","rvol_20","vol_ratio"],
    "ATR":        ["atr_pct","atr_trend"],
    "Stoch":      ["stoch_k","stoch_d","stoch_diff"],
    "Williams":   ["willr"],
    "CCI":        ["cci"],
    "ADX":        ["adx_plus","adx_minus","adx_diff"],
    "Momentum":   ["roc_3","roc_5","roc_10","roc_20"],
    "Volume":     ["v_ratio","v_trend","pv_corr"],
    "Pattern":    ["hl_pct","open_close","upper_wick","lower_wick"],
    "Distance":   ["dist_52w_high","dist_52w_low"],
    "Trend":      ["sma50_slope","sma20_slope"],
}

DEFAULT_GROUPS = [
    "Returns","RSI","MACD","Bollinger","SMA","Volatility","ATR","Momentum"
]
