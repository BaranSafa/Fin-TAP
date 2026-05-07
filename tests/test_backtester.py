import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify

import backend.backtester as backtester


def test_backtest_allow_short_returns_json_safe_bools(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=160, freq="D")
    close = np.linspace(200, 80, 160) + np.sin(np.arange(160)) * 3
    df = pd.DataFrame(
        {
            "Close": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    monkeypatch.setattr(backtester, "get_processed_data", lambda ticker: df)

    result = backtester.run_backtest("AAPL", allow_short=True)

    assert result["trade_count"] > 0
    assert any(t["signal"] == "short" and isinstance(t["won"], bool) for t in result["recent_trades"])

    app = Flask(__name__)
    with app.app_context():
        payload = jsonify(result).get_data(as_text=True)
    assert json.loads(payload)["ticker"] == "AAPL"
