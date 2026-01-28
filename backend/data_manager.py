import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def get_processed_data(ticker, start_date="2020-01-01"): # Tarihi biraz yaklaştırdım, daha hızlı çalışır
    try:
        # Bugüne kadar olan veriyi çek
        end_date = datetime.now()
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            if not df.columns.is_unique:
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
            
        df.columns = df.columns.str.lower()
        
        # --- SADECE PANDAS İLE HESAPLAMALAR ---
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        df['returns'] = close.pct_change()
        df['lag_1'] = close.shift(1)
        df['lag_2'] = close.shift(2)
        df['SMA_14'] = close.rolling(14).mean()
        df['SMA_50'] = close.rolling(50).mean()
        df['EMA_14'] = close.ewm(span=14).mean()
        df['volatility'] = close.rolling(20).std()
        
        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BBU_20_2.0'] = sma20 + 2*std20
        df['BBM_20_2.0'] = sma20
        df['BBL_20_2.0'] = sma20 - 2*std20
        
        # Momentum
        df['MOM_10'] = close.diff(10)
        
        # MACD
        exp12 = close.ewm(span=12).mean()
        exp26 = close.ewm(span=26).mean()
        df['MACD_12_26_9'] = exp12 - exp26
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Stoch
        ll = low.rolling(14).min()
        hh = high.rolling(14).max()
        df['STOCHk_14_3_3'] = 100 * ((close - ll) / (hh - ll))
        df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(3).mean()
        
        # ATR
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        df['ATRr_14'] = tr.rolling(14).mean()
        
        # OBV
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # ADX (Basit)
        df['ADX_14'] = (high - low) / close
        df['DMP_14'] = high.diff()
        df['DMN_14'] = -low.diff()
        df['ADXR_14_2'] = df['ADX_14'].rolling(14).mean()

        # DİKKAT: Burada 'Target' oluşturmuyoruz ve 'dropna' yapmıyoruz.
        # Sadece indikatörlerin başındaki NaN'ları temizliyoruz (ilk 50 gün)
        # Ama SON GÜNLERİ (Bugünü) KORUYORUZ.
        df.dropna(inplace=True)
        
        df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        return df
    except Exception as e:
        print(f"Data error: {e}")
        return None