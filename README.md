# Fin-TAP 🚀

Fin-TAP (Financial Trend Analysis & Prediction Platform) is a comprehensive Full-Stack Web and REST API application that analyzes price trends of financial assets using machine learning models, offers backtesting capabilities, provides paper trading simulation, and features AI-powered stock screening tools.

The application is built on a modular, flexible architecture featuring a **Flask** backend and a modern JavaScript/Tailwind-based **Frontend**.

---

## 🌟 Core Features

* 🔮 **Dynamic AI Prediction Engine:** Generates 7, 14, 30, and 90-day price projections using advanced machine learning models such as `LINEAR`, `RANDOM_FOREST`, `EXTRA_TREES`, `GRADIENT_BOOST`, `XGBOOST`, `LIGHTGBM`, and `LSTM`.
* 📊 **Advanced Strategy Backtester:** Simulates RSI + SMA trend-following strategies using historical market data. It computes professional performance metrics including strategy returns, Buy-and-Hold returns, Win Rate, Maximum Drawdown, and Sharpe Ratio. Supports Short Selling simulations.
* 📈 **Simulated Paper Trading:** Integrated with real-time asset price feeds, allowing users to practice trading strategies and manage a virtual portfolio with risk-free starter capital.
* 🧠 **AI Insights & Portfolio Analysis:**
    * **AI Analyst:** Evaluates current technical indicators (`RSI`, `SMA`, `Volatility`) via deterministic logic to determine asset risk levels (`LOW`, `MEDIUM`, `HIGH`) and directional confidence.
    * **AI Portfolio Insight:** Automatically scans the user's watchlist to highlight top-performing assets and high-risk exposures.
    * **AI Stock Screener:** Ranks and filters tracked instruments tailored to specific investor profiles: Balanced, Low Risk, or Momentum.
* 🛡️ **Robust Production Security:** Armed with full CSRF protection (`Flask-WTF`), HTTPOnly/Secure/SameSite cookie compliance, strong cryptographic password hashing (`Werkzeug`), Open Redirect defense mechanisms, and secure HTTP headers (`X-Frame-Options`, `X-Content-Type-Options`).
* 💸 **Stripe Token Economy:** Implements a token consumption architecture for running ML predictions. Built-in Stripe Checkout and Subscription models with fully automated renewal webhook synchronization.
* 📡 **Developer REST API v1:** Exposes dedicated endpoints guarded by Bearer Token (API Key) authentication and daily rate-limiting, serving clean OHLCV history, market prices, and news sentiment feeds.
* 🔔 **Smart Price Alerts:** User-configured asset boundary parameters that trigger automated real-time HTML/text email updates via `Flask-Mail` upon breakout thresholds.
* 📰 **News Sentiment Analysis:** Parses Yahoo Finance RSS headlines through the VADER Sentiment Intensity Analyzer to derive real-time market sentiment flags (`BULLISH`, `BEARISH`, `NEUTRAL`).

---

## 🛠️ Technology Stack

* **Backend Framework:** Python 3.10+, Flask, Flask-Login, Flask-SQLAlchemy (Supports SQLite & PostgreSQL)
* **Data & Machine Learning:** Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, yfinance, curl_cffi, VADER Sentiment
* **Security & Rate Limiting:** Flask-Limiter, Flask-WTF (CSRF)
* **Payment Integration:** Stripe Python SDK
* **Testing Suite:** Pytest, Pytest-Flask

---

## 📦 Installation & Configuration

### 1. Clone the Repository and Navigate to the Directory
```bash
git clone [https://github.com/your_username/Fin-TAP.git](https://github.com/your_username/Fin-TAP.git)
cd Fin-TAP