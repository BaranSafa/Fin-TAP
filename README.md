# Fin-TAP — AI-Powered Financial Analysis Platform

**Graduation Project — Financial Technology & Applied AI Platform**

---

## 1. Abstract

Fin-TAP is a full-stack, AI-driven financial forecasting and analysis platform built with Flask (Python) on the backend and a custom-designed responsive web frontend. The platform allows registered users to generate machine-learning-based price forecasts for 31 stocks and 8 cryptocurrencies, compare assets side-by-side, track a personal watchlist, set price alerts, simulate trades with a virtual $10,000 paper-trading portfolio, backtest a rule-based trading strategy against historical data, and consume a developer-facing REST API.

The core forecasting engine trains one of seven selectable machine learning models (Ridge Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, and an LSTM neural network) on-demand, using over 50 engineered technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, CCI, ADX, volume ratios, and more) derived from live OHLCV market data fetched from Yahoo Finance.

Beyond price prediction, the project implements an original **AI Analyst module** (`backend/ai_analyst.py`) consisting of three independently coded, from-scratch artificial intelligence models that analyze the same technical indicators from complementary statistical perspectives and combine their outputs into a single ensemble decision:

1. **Naive Bayes Sentiment Classifier** — a probabilistic text classifier implemented manually with Bayes' theorem and Laplace smoothing, trained on a hand-curated financial vocabulary to classify market context as positive, negative, or neutral.
2. **TF-IDF + Logistic Regression** — a manually implemented Term Frequency–Inverse Document Frequency vectorizer feeding a logistic regression classifier trained via gradient descent (with learning-rate decay) to estimate the probability of a bullish or bearish signal.
3. **Rule-Based Expert System** — an IF-THEN production-rule engine that evaluates more than 25 technical-analysis rules (RSI thresholds, MACD crossovers, Bollinger Band position, trend slope, volatility, volume spikes, 52-week price position, etc.), each carrying a signed weight, and aggregates them into a weighted BUY/SELL/HOLD recommendation.

The three models' outputs are merged using a weighted-voting ensemble (Expert System 50%, Naive Bayes 25%, TF-IDF Logistic Regression 25%) to produce a final recommendation, a confidence score, and a human-readable analysis report — entirely with code written for this project, without relying on any third-party LLM API.

The platform also includes a token-based monetization system (Stripe integration for one-time token packs and monthly subscriptions), user authentication with secure password hashing and a password-reset flow, an admin dashboard, rate limiting, and CSRF protection, making it a production-oriented demonstration of combining classical machine learning, hand-built statistical AI, and modern web engineering practices.

---

## 2. Key Features

| Module                         | Description                                                                                   |
|---------------------------------|------------------------------------------------------------------------------------------------|
| AI Price Forecasting            | 7 selectable ML models, 7/14/30/90-day horizons, dynamic feature-group selection               |
| **AI Analyst (3-model ensemble)** | Naive Bayes, TF-IDF + Logistic Regression, and a 25+ rule Expert System, combined via weighted voting |
| Asset Comparison                | Side-by-side AI forecasts and technical scoring for two assets                                 |
| News Sentiment Analysis         | Yahoo Finance RSS + VADER sentiment scoring per ticker                                         |
| Candlestick & Indicator Charts  | OHLCV candlesticks with toggleable SMA/Bollinger/RSI overlays                                  |
| Watchlist / Portfolio           | Per-user favorite-instrument tracking with live price and RSI signal                           |
| Price Alerts                    | Threshold-based alerts with scheduled email notifications                                      |
| Paper Trading                   | $10,000 virtual portfolio simulator with buy/sell execution and P&L tracking                   |
| Backtesting Engine              | RSI + SMA trend-signal strategy simulated against historical data, with Sharpe ratio & drawdown |
| Correlation Matrix              | Pairwise Pearson correlation heatmap across all tracked instruments                            |
| Developer REST API              | Bearer-token authenticated public API (`/api/v1/...`) with daily rate limits                   |
| Token Economy & Payments        | Stripe Checkout for token packs and subscriptions, webhook-driven balance updates              |
| Authentication & Security       | Hashed passwords, CSRF protection, rate limiting, secure cookies, password-reset via email      |
| Admin Dashboard                 | Revenue, user activity, prediction volume, and cache-health monitoring                         |

---

## 3. Project Structure

```
Fin-TAP/
├── app.py                     # Main Flask application (routes, auth, payments, API)
├── models.py                  # SQLAlchemy database models
├── train.py                   # Ticker universe definition / setup script
├── run.py                     # Local development server entry point
├── Procfile                   # Production server command (gunicorn)
├── requirements.txt           # Python dependencies
├── backend/
│   ├── data_manager.py        # Market data fetching + 50+ technical indicator engineering
│   ├── dynamic_trainer.py     # On-demand ML model training & rolling-window forecasting
│   ├── model_manager.py       # BUY/SELL/HOLD scoring used by the comparison page
│   ├── backtester.py          # Historical strategy backtesting engine
│   └── ai_analyst.py          # AI Analyst: Naive Bayes, TF-IDF+LR, Expert System, Ensemble
└── frontend/
    ├── templates/              # Jinja2 HTML templates
    └── static/js/              # Client-side JavaScript (charts, forms, AJAX)
```

---

## 4. Required Libraries

All dependencies are listed in `requirements.txt`:

```
flask
flask-login
flask-sqlalchemy
flask-cors
flask-limiter
flask-wtf
flask-mail
werkzeug
gunicorn
psycopg2-binary
yfinance>=0.2.50
curl_cffi==0.7.4
pandas
numpy
scikit-learn
xgboost
lightgbm
requests
stripe
feedparser
vaderSentiment
```

**Notes:**
- `tensorflow` is required only if the **LSTM** model option is used. It is imported lazily and the application degrades gracefully (LSTM option disabled) if it is not installed.
- The **AI Analyst module** (`backend/ai_analyst.py`) uses only Python's standard library (`math`, `re`) — no additional packages are required for it.
- `psycopg2-binary` is only needed for PostgreSQL; the app falls back to SQLite automatically if no `DATABASE_URL` environment variable is set.

---

## 5. Installation & Setup

### 5.1 Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- (Optional) A virtual environment tool such as `venv`

### 5.2 Step-by-Step Installation

```bash
# 1. Extract the archive and navigate into the project folder
cd Fin-TAP

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install TensorFlow only if you want to use the LSTM model
pip install tensorflow

# 5. (Optional) Initialize the ticker registry placeholder files
python train.py
```

### 5.3 Environment Variables (Optional)

The application runs out-of-the-box with safe defaults (SQLite database, auto-generated secret key). For full functionality, the following environment variables can be set:

| Variable              | Purpose                                              | Required?            |
|------------------------|-------------------------------------------------------|------------------------|
| `SECRET_KEY`           | Flask session signing key                             | Recommended for production |
| `DATABASE_URL`         | PostgreSQL connection string                          | Optional (defaults to SQLite) |
| `ADMIN_SECRET`         | Header secret to access `/db-test`, `/db-kur`         | Optional |
| `ADMIN_EMAIL`          | Comma-separated emails granted `/admin` access        | Optional |
| `MAIL_SERVER` / `MAIL_USERNAME` / `MAIL_PASSWORD` | SMTP credentials for password-reset & price-alert emails | Optional (dev-mode link shown if absent) |
| `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` | Enables token-purchase checkout & webhook  | Optional |
| `CRON_SECRET`          | Protects scheduled `/api/refresh` and `/api/alerts/check` endpoints | Optional |
| `ALLOWED_ORIGINS`      | CORS allow-list for production deployments             | Optional |

If these are not set, the app will run in a safe local/demo mode (no real payments, no real emails sent — reset links are printed/displayed instead).

### 5.4 Running the Project Locally

```bash
python run.py
```

Then open a browser and navigate to:

```
http://127.0.0.1:5000
```

The first visit will automatically create the SQLite database (`instance/fintap.db`) and required tables.

### 5.5 Running in Production (Optional)

```bash
gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
```

(This is also defined in the included `Procfile` for platforms such as Render or Heroku.)

---

## 6. Using the Platform

1. **Register** a free account at `/register` — every new user receives 5 free prediction tokens.
2. **Run an AI Forecast** at `/predict`: choose a ticker, an ML model, a forecast horizon (7/14/30/90 days), and a set of technical-indicator feature groups, then click "Run Analysis."
3. **View the AI Analyst report**: after a forecast completes, click **"AI Derin Analiz (3 Model)"** to run the three-model ensemble (Naive Bayes, TF-IDF + Logistic Regression, Expert System) and view the combined BUY/SELL/HOLD recommendation with rule-by-rule justification.
4. **Explore other modules**: Compare assets, manage your Watchlist, set Price Alerts, try Paper Trading, run the Backtester, view the Correlation Matrix, or generate a Developer API key from the respective sidebar pages.

---

## 7. Data Source Note

This project does **not** ship a static dataset. All market data (OHLCV price history) is fetched live and on-demand from Yahoo Finance via the `yfinance` library (with a `curl_cffi`/REST fallback chain implemented in `backend/data_manager.py`). No internet connection is required to read the source code, but a connection is needed to fetch live prices when running the application. An in-memory cache (1-hour TTL while markets are open, 6-hour TTL otherwise) minimizes redundant network requests during a session.

No large datasets are included in this archive, in line with the submission guidelines.

---

## 8. Disclaimer

This project was developed for academic/educational purposes as part of a graduation project. The AI forecasts, technical signals, and the AI Analyst's BUY/SELL/HOLD recommendations are **not financial advice** and should not be used for real investment decisions.
