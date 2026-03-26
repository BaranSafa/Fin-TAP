"""
data_manager.py  –  Fin-TAP Backend
Sıfırdan yazılmış, tüm edge-case'ler kapatılmış veri pipeline.

Teşhis edilen sorunlar:
1. yfinance MultiIndex tuple kolonlar → string flatten
2. Stoch / RSI sıfıra bölme → inf / NaN
3. volume int64 → float cast
4. dropna NaN kontrolü sonrası bile bazı kolonlar NaN kalıyor
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


# ──────────────────────────────────────────────────────────
#  YARDIMCI: Güvenli EWM (pandas ewm bazen uyarı verir)
# ──────────────────────────────────────────────────────────
def _safe_ewm(series, span):
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def get_processed_data(ticker: str, start_date: str = "2018-01-01") -> pd.DataFrame | None:
    """
    Verilen ticker için ham OHLCV verisini indirir,
    teknik indikatörleri hesaplar ve temiz bir DataFrame döndürür.

    Returns:
        pd.DataFrame  – başarı
        None          – herhangi bir hata
    """
    try:
        # ── 1. YFinance'dan veri çek ─────────────────────────────────────
        df = yf.download(
            ticker,
            start=start_date,
            end=datetime.now().strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True,   # split/dividend düzeltmesi
        )

        if df is None or df.empty:
            print(f"[data_manager] {ticker}: Boş veri.")
            return None

        # ── 2. MultiIndex flatten (yfinance 0.2.x+) ─────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).strip() for col in df.columns]
        else:
            df.columns = [str(c).strip() for c in df.columns]

        # Küçük harf
        df.columns = df.columns.str.lower()

        # Duplicate sütun/index temizliği
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        df = df[~df.index.duplicated(keep='last')]

        # ── 3. Gerekli sütunlar kontrolü ─────────────────────────────────
        needed = {'open', 'high', 'low', 'close', 'volume'}
        missing = needed - set(df.columns)
        if missing:
            print(f"[data_manager] {ticker}: Eksik sütunlar → {missing}")
            return None

        # ── 4. Tip düzeltmeleri ──────────────────────────────────────────
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=list(needed), inplace=True)

        if len(df) < 100:
            print(f"[data_manager] {ticker}: Yetersiz satır ({len(df)})")
            return None

        # ── 5. Çalışma serileri ──────────────────────────────────────────
        close  = df['close'].astype(float)
        high   = df['high'].astype(float)
        low    = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # ── 6. Lag / Return özellikleri ──────────────────────────────────
        df['lag_1']   = close.shift(1)
        df['lag_2']   = close.shift(2)
        df['lag_5']   = close.shift(5)
        df['lag_10']  = close.shift(10)

        df['returns']  = close.pct_change()
        df['ret_3']    = close.pct_change(3)
        df['ret_5']    = close.pct_change(5)
        df['ret_10']   = close.pct_change(10)

        # ── 7. Hareketli Ortalamalar ─────────────────────────────────────
        df['SMA_5']  = close.rolling(5).mean()
        df['SMA_14'] = close.rolling(14).mean()
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()

        df['EMA_9']  = _safe_ewm(close, 9)
        df['EMA_14'] = _safe_ewm(close, 14)
        df['EMA_21'] = _safe_ewm(close, 21)

        # Fiyatın SMA'ya uzaklığı (normalise)
        df['price_sma14_ratio'] = close / df['SMA_14'].replace(0, np.nan)
        df['price_sma50_ratio'] = close / df['SMA_50'].replace(0, np.nan)

        # ── 8. Volatilite ────────────────────────────────────────────────
        df['volatility_10']  = close.rolling(10).std()
        df['volatility_20']  = close.rolling(20).std()
        df['log_returns']    = np.log(close / close.shift(1))
        df['realized_vol']   = df['log_returns'].rolling(20).std() * np.sqrt(252)

        # ── 9. Bollinger Bands ───────────────────────────────────────────
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        df['BB_upper']  = sma20 + 2 * std20
        df['BB_middle'] = sma20
        df['BB_lower']  = sma20 - 2 * std20
        # %B: fiyatın bantlar içindeki konumu [0-1]
        bb_range = (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
        df['BB_pct']  = (close - df['BB_lower']) / bb_range
        # Bant genişliği
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma20.replace(0, np.nan)

        # ── 10. MACD ─────────────────────────────────────────────────────
        ema12 = _safe_ewm(close, 12)
        ema26 = _safe_ewm(close, 26)
        df['MACD']        = ema12 - ema26
        df['MACD_signal'] = _safe_ewm(df['MACD'], 9)
        df['MACD_hist']   = df['MACD'] - df['MACD_signal']

        # ── 11. RSI ──────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df['RSI_14'] = (100 - (100 / (1 + rs))).fillna(50)

        # RSI(7) – kısa vadeli momentum
        gain7 = delta.where(delta > 0, 0.0).rolling(7).mean()
        loss7 = (-delta.where(delta < 0, 0.0)).rolling(7).mean()
        rs7   = gain7 / loss7.replace(0, np.nan)
        df['RSI_7'] = (100 - (100 / (1 + rs7))).fillna(50)

        # ── 12. Stochastic ───────────────────────────────────────────────
        ll14 = low.rolling(14).min()
        hh14 = high.rolling(14).max()
        denom = (hh14 - ll14).replace(0, np.nan)
        df['STOCH_K'] = (100 * (close - ll14) / denom).fillna(50)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()

        # ── 13. ATR (Average True Range) ─────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        # Normalised ATR
        df['ATR_pct'] = df['ATR_14'] / close.replace(0, np.nan)

        # ── 14. OBV (On-Balance Volume) ──────────────────────────────────
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV']        = obv
        df['OBV_SMA_14'] = obv.rolling(14).mean()
        df['OBV_ratio']  = obv / obv.rolling(14).mean().replace(0, np.nan)

        # ── 15. ADX (Average Directional Index – basit) ──────────────────
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr_roll  = tr.rolling(14).mean().replace(0, np.nan)
        df['ADX_plus']  = 100 * plus_dm.rolling(14).mean()  / tr_roll
        df['ADX_minus'] = 100 * minus_dm.rolling(14).mean() / tr_roll
        df['ADX_14']    = (df['ADX_plus'] - df['ADX_minus']).abs()

        # ── 16. Momentum / ROC ───────────────────────────────────────────
        df['MOM_5']  = close.diff(5)
        df['MOM_10'] = close.diff(10)
        df['ROC_5']  = close.pct_change(5) * 100
        df['ROC_10'] = close.pct_change(10) * 100

        # ── 17. CCI (Commodity Channel Index) ────────────────────────────
        typical = (high + low + close) / 3
        cci_ma  = typical.rolling(20).mean()
        cci_md  = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI_20'] = ((typical - cci_ma) / (0.015 * cci_md.replace(0, np.nan))).fillna(0)

        # ── 18. Williams %R ──────────────────────────────────────────────
        hh14w = high.rolling(14).max()
        ll14w = low.rolling(14).min()
        dw    = (hh14w - ll14w).replace(0, np.nan)
        df['WILLIAMS_R'] = (-100 * (hh14w - close) / dw).fillna(-50)

        # ── 19. Hacim özellikleri ────────────────────────────────────────
        df['volume_sma_14']   = volume.rolling(14).mean()
        df['volume_ratio']    = volume / df['volume_sma_14'].replace(0, np.nan)
        df['price_x_volume']  = (close * volume) / 1e9   # normalise

        # ── 20. Fiyat pattern ────────────────────────────────────────────
        df['high_low_pct'] = (high - low) / close.replace(0, np.nan)
        df['close_open']   = (close - df['open']) / close.replace(0, np.nan)

        # ── NaN temizliği ─────────────────────────────────────────────────
        df.dropna(inplace=True)

        if df.empty:
            print(f"[data_manager] {ticker}: dropna sonrası boş.")
            return None

        # ── Sütun adlarını kısmen büyük harfe geri çevir ─────────────────
        df.rename(columns={
            'open':   'Open',
            'high':   'High',
            'low':    'Low',
            'close':  'Close',
            'volume': 'Volume',
        }, inplace=True)

        print(f"[data_manager] {ticker}: {len(df)} satır, {len(df.columns)} feature hazır.")
        return df

    except Exception as e:
        import traceback
        print(f"[data_manager] {ticker} HATA: {e}")
        traceback.print_exc()
        return None
