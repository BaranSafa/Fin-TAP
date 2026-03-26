"""
model_manager.py  –  Fin-TAP Backend
AI öneri metrikleri hesaplama modülü.
"""

try:
    from .dynamic_trainer import train_and_predict_dynamic
    from .data_manager import get_processed_data
except ImportError:
    from dynamic_trainer import train_and_predict_dynamic
    from data_manager import get_processed_data


def get_suggestion_metrics(ticker: str) -> dict | None:
    """
    Ticker için AI tabanlı öneri metrikleri üretir.
    compare_stocks API endpoint'i tarafından kullanılır.

    Returns: dict veya None
    """
    try:
        # Hızlı bir model ile tahmin al (LINEAR = Ridge, hızlı)
        future_preds, _ = train_and_predict_dynamic(
            ticker, 'LINEAR', ['RSI', 'MACD', 'Bollinger', 'Lag']
        )

        if future_preds is None or len(future_preds) == 0:
            print(f"[model_manager] {ticker}: tahmin döndü None")
            return None

        # 14. günün tahmini (en uçtaki projeksiyon)
        future_price = float(future_preds[-1])

        # Güncel piyasa verisi
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return None

        last_price = float(df['Close'].iloc[-1])
        rsi        = float(df['RSI_14'].iloc[-1])

        if last_price == 0:
            return None

        # Getiri potansiyeli
        potential_gain_pct = ((future_price / last_price) - 1) * 100

        # ── Skor sistemi ──────────────────────────────────────────────────
        score = 0

        # Getiri
        if potential_gain_pct > 5:  score += 3
        elif potential_gain_pct > 2: score += 2
        elif potential_gain_pct > 0: score += 1
        elif potential_gain_pct < -2: score -= 2

        # RSI momentumu
        if rsi < 30:    score += 3   # aşırı satım → fırsat
        elif rsi < 45:  score += 1
        elif rsi > 70:  score -= 3   # aşırı alım → risk
        elif rsi > 60:  score -= 1

        # MACD sinyali
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd_val = float(df['MACD'].iloc[-1])
            macd_sig = float(df['MACD_signal'].iloc[-1])
            if macd_val > macd_sig:  score += 1
            else:                    score -= 1

        # Bollinger %B konumu
        if 'BB_pct' in df.columns:
            bb_pct = float(df['BB_pct'].iloc[-1])
            if bb_pct < 0.2:   score += 1   # alt bant yakını
            elif bb_pct > 0.8: score -= 1   # üst bant yakını

        # ── Öneri etiketleri ─────────────────────────────────────────────
        if score >= 4:
            recommendation = "STRONG BUY"
            risk_level     = "LOW"
            signal_color   = "green"
        elif score >= 2:
            recommendation = "BUY"
            risk_level     = "LOW-MED"
            signal_color   = "green"
        elif score >= 0:
            recommendation = "HOLD"
            risk_level     = "MEDIUM"
            signal_color   = "amber"
        elif score >= -2:
            recommendation = "CAUTION"
            risk_level     = "HIGH"
            signal_color   = "amber"
        else:
            recommendation = "SELL"
            risk_level     = "HIGH"
            signal_color   = "red"

        # RSI rengi (frontend için)
        rsi_color = "text"
        if rsi > 70:   rsi_color = "red"
        elif rsi < 30: rsi_color = "green"

        return {
            "ticker":             ticker,
            "last_price":         round(last_price, 2),
            "predicted_price":    round(future_price, 2),
            "potential_gain_pct": round(potential_gain_pct, 2),
            "rsi":                round(rsi, 2),
            "recommendation":     recommendation,
            "risk_level":         risk_level,
            "signal_color":       signal_color,
            "rsi_color":          rsi_color,
            "score":              score,
        }

    except Exception as e:
        import traceback
        print(f"[model_manager] {ticker} HATA: {e}")
        traceback.print_exc()
        return None
