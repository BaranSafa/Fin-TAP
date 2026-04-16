"""model_manager.py  —  Fin-TAP Backend"""
from __future__ import annotations
import traceback

try:
    from .dynamic_trainer import train_and_predict_dynamic
    from .data_manager    import get_processed_data
except ImportError:
    from dynamic_trainer import train_and_predict_dynamic
    from data_manager    import get_processed_data


def get_suggestion_metrics(ticker: str) -> dict | None:
    try:
        future_preds, _ = train_and_predict_dynamic(
            ticker, "LINEAR", ["RSI", "MACD", "Bollinger", "Volatility"]
        )
        if not future_preds:
            return None

        future_price = float(future_preds[-1])
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return None

        last_price = float(df["Close"].iloc[-1])
        rsi14      = float(df["rsi_14"].iloc[-1])
        if last_price == 0:
            return None

        gain_pct = ((future_price / last_price) - 1) * 100

        score = 0
        if gain_pct > 5:    score += 3
        elif gain_pct > 2:  score += 2
        elif gain_pct > 0:  score += 1
        elif gain_pct < -2: score -= 2

        if rsi14 < 30:      score += 3
        elif rsi14 < 45:    score += 1
        elif rsi14 > 70:    score -= 3
        elif rsi14 > 60:    score -= 1

        if "macd" in df.columns and "macd_sig" in df.columns:
            score += 1 if float(df["macd"].iloc[-1]) > float(df["macd_sig"].iloc[-1]) else -1

        if "bb_pct" in df.columns:
            bp = float(df["bb_pct"].iloc[-1])
            if bp < 0.2:   score += 1
            elif bp > 0.8: score -= 1

        if score >= 4:      rec, risk, color = "STRONG BUY",  "LOW",    "green"
        elif score >= 2:    rec, risk, color = "BUY",         "LOW-MED","green"
        elif score >= 0:    rec, risk, color = "HOLD",        "MEDIUM", "amber"
        elif score >= -2:   rec, risk, color = "CAUTION",     "HIGH",   "amber"
        else:               rec, risk, color = "SELL",        "HIGH",   "red"

        rsi_color = "text"
        if rsi14 > 70:   rsi_color = "red"
        elif rsi14 < 30: rsi_color = "green"

        return {
            "ticker": ticker, "last_price": round(last_price, 2),
            "predicted_price": round(future_price, 2),
            "potential_gain_pct": round(gain_pct, 2),
            "rsi": round(rsi14, 2), "recommendation": rec,
            "risk_level": risk, "signal_color": color,
            "rsi_color": rsi_color, "score": score,
        }
    except Exception as e:
        print(f"[model_manager] {ticker}: {e}")
        traceback.print_exc()
        return None
