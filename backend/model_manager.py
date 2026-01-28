from .dynamic_trainer import train_and_predict_dynamic
from .data_manager import get_processed_data

def get_suggestion_metrics(ticker):
    try:
        # 1. Tahmin Al
        future_price, _ = train_and_predict_dynamic(ticker, 'LINEAR', ['RSI', 'MACD'])
        
        if future_price is None: return None
        
        # 2. Veri Al
        df = get_processed_data(ticker)
        if df is None or df.empty: return None

        last_price = df['Close'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]
        
        # 3. Hesapla
        potential_gain_pct = ((future_price / last_price) - 1) * 100
        
        # 4. Mantık ve Renkler (DÜZLEŞTİRİLMİŞ YAPI)
        score = 0
        
        if potential_gain_pct > 1: score += 2
        if potential_gain_pct > 3: score += 1
        
        if rsi < 30: score += 2
        elif rsi < 50: score += 1
        elif rsi > 70: score -= 3
        
        # Varsayılan Değerler (Gri/Nötr)
        recommendation = "NÖTR"
        risk_level = "ORTA"
        
        # CSS Sınıfları (Doğrudan değişken olarak)
        css_border = "border-gray-500"
        css_title = "text-gray-400"
        css_badge = "bg-gray-800 text-gray-300 border-gray-600"
        css_price = "text-gray-400"
        css_bar = "bg-gray-500"
        css_percent = "text-gray-400"
        
        # Renk Mantığı
        if score >= 3:
            recommendation = "GÜÇLÜ AL"
            risk_level = "DÜŞÜK" if rsi < 40 else "ORTA"
            css_border = "border-green-500"
            css_title = "text-green-400"
            css_badge = "bg-green-900 text-green-300 border-green-700"
            css_price = "text-green-400"
            css_bar = "bg-green-500"
            css_percent = "text-green-400"
            
        elif score >= 1:
            recommendation = "AL"
            risk_level = "ORTA"
            css_border = "border-cyan-500"
            css_title = "text-cyan-400"
            css_badge = "bg-cyan-900 text-cyan-300 border-cyan-700"
            css_price = "text-cyan-400"
            css_bar = "bg-cyan-500"
            css_percent = "text-cyan-400"
            
        elif score <= -1:
            recommendation = "SAT"
            risk_level = "YÜKSEK"
            css_border = "border-red-500"
            css_title = "text-red-400"
            css_badge = "bg-red-900 text-red-300 border-red-700"
            css_price = "text-red-400"
            css_bar = "bg-red-500"
            css_percent = "text-red-400"
        
        # RSI Rengi
        rsi_color = "text-gray-300"
        if rsi > 70: rsi_color = "text-red-400"
        elif rsi < 30: rsi_color = "text-green-400"

        # Düz sözlük döndür (Nested object yok)
        return {
            "ticker": ticker,
            "last_price": round(last_price, 2),
            "predicted_price": round(future_price, 2),
            "potential_gain_pct": round(potential_gain_pct, 2),
            "rsi": round(rsi, 2),
            "recommendation": recommendation,
            "risk_level": risk_level,
            "rsi_color": rsi_color,
            # CSS sınıfları ayrı ayrı
            "css_border": css_border,
            "css_title": css_title,
            "css_badge": css_badge,
            "css_price": css_price,
            "css_bar": css_bar,
            "css_percent": css_percent
        }

    except Exception as e:
        print(f"Suggestion error {ticker}: {e}")
        return None