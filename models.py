from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

# Tüm modeller bu db nesnesi üzerinden tanımlanır; app.py'de db.init_app(app) ile bağlanır.
db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(255), unique=True, nullable=False, index=True) 
    password   = db.Column(db.String(512), nullable=False)   
    name       = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    wallet       = db.relationship("Wallet",      backref="user", uselist=False)   
    transactions = db.relationship("Transaction", backref="user", lazy=True)
    predictions  = db.relationship("Prediction",  backref="user", lazy=True)
    watchlist    = db.relationship("Watchlist",   backref="user", lazy=True)
    price_alerts = db.relationship("PriceAlert",  backref="user", lazy=True)
    api_keys     = db.relationship("ApiKey",      backref="user", lazy=True)


class Wallet(db.Model):
    """
    Token cüzdanı — her kullanıcının kaç tahmini hakkı kaldığını tutar.
    Yeni kayıtta 5 token verilir (app.py'deki register route'unda).
    """
    __tablename__ = "wallet"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, index=True)
    balance      = db.Column(db.Integer, default=5)           
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


class Transaction(db.Model):
    """
    Stripe ödeme kaydı — kullanıcı token satın aldığında burada saklanır.
    Ödeme başarılı olduğunda Wallet.balance bu tablodaki tokens_added kadar artar.
    """
    __tablename__ = "transaction"
    __table_args__ = (
        db.Index("ix_transaction_user_id", "user_id"),  
    )

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"))
    amount_paid  = db.Column(db.Float,   nullable=False)  
    tokens_added = db.Column(db.Integer, nullable=False)   
    date         = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):

    __tablename__ = "prediction"
    __table_args__ = (
        db.Index("ix_prediction_user_id", "user_id"),
    )

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("user.id"))
    symbol           = db.Column(db.String(20),  nullable=False) 
    model_type       = db.Column(db.String(50))                    
    predicted_result = db.Column(db.String(100))                    
    accuracy_pct     = db.Column(db.Float, nullable=True)           
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)


class Watchlist(db.Model):
    __tablename__ = "watchlist"
    __table_args__ = (
        db.Index("ix_watchlist_user_id", "user_id"),
        db.UniqueConstraint("user_id", "symbol", name="uq_watchlist_user_symbol"),
    )

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol     = db.Column(db.String(20), nullable=False)
    added_at   = db.Column(db.DateTime, default=datetime.utcnow)


class PriceAlert(db.Model):
    """
    Fiyat alarmı: bir hisse belirli fiyatın üstüne/altına indiğinde email gönder.
    direction: 'above' → fiyat hedefin üzerine çıktığında tetikle
               'below' → fiyat hedefin altına düştüğünde tetikle
    status:    'active' | 'triggered' | 'cancelled'
    """
    __tablename__ = "price_alert"
    __table_args__ = (
        db.Index("ix_price_alert_user_id", "user_id"),
        db.Index("ix_price_alert_active",  "status"),   
    )

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol       = db.Column(db.String(20),  nullable=False)
    target_price = db.Column(db.Float,       nullable=False)
    direction    = db.Column(db.String(10),  nullable=False)   
    note         = db.Column(db.String(200), nullable=True)   
    status       = db.Column(db.String(20),  default="active") 
    created_at   = db.Column(db.DateTime,    default=datetime.utcnow)
    triggered_at = db.Column(db.DateTime,    nullable=True)    


PAPER_STARTING_CASH = 10_000.0


class ApiKey(db.Model):
    __tablename__ = "api_key"
    __table_args__ = (
        db.Index("ix_api_key_user_id", "user_id"),
        db.Index("ix_api_key_hash",    "key_hash", unique=True),  # hash benzersiz olmalı
    )

    id             = db.Column(db.Integer,     primary_key=True)
    user_id        = db.Column(db.Integer,     db.ForeignKey("user.id"), nullable=False)
    name           = db.Column(db.String(80),  nullable=False)               # kullanıcının verdiği isim
    key_prefix     = db.Column(db.String(20),  nullable=False)               # fintap_sk_XXXX...
    key_hash       = db.Column(db.String(64),  nullable=False, unique=True)  # SHA-256 hash
    is_active      = db.Column(db.Boolean,     default=True)
    requests_today = db.Column(db.Integer,     default=0)    # günlük istek sayacı
    last_used_at   = db.Column(db.DateTime,    nullable=True)
    created_at     = db.Column(db.DateTime,    default=datetime.utcnow)


class PaperPortfolio(db.Model):

    __tablename__ = "paper_portfolio"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False, index=True)
    cash       = db.Column(db.Float,   default=PAPER_STARTING_CASH, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_at   = db.Column(db.DateTime, nullable=True)


class PaperPosition(db.Model):

    __tablename__ = "paper_position"
    __table_args__ = (
        db.Index("ix_paper_position_user_id", "user_id"),
        db.UniqueConstraint("user_id", "symbol", name="uq_paper_pos_user_sym"),
    )

    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol    = db.Column(db.String(20), nullable=False)
    quantity  = db.Column(db.Float,  nullable=False, default=0.0)  # kesirli hisse desteklenir
    avg_cost  = db.Column(db.Float,  nullable=False, default=0.0)  # ortalama alış fiyatı
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)


class PaperTrade(db.Model):

    __tablename__ = "paper_trade"
    __table_args__ = (
        db.Index("ix_paper_trade_user_id", "user_id"),
    )

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol      = db.Column(db.String(20), nullable=False)
    action      = db.Column(db.String(10), nullable=False)   # 'buy' | 'sell'
    quantity    = db.Column(db.Float,  nullable=False)
    price       = db.Column(db.Float,  nullable=False)       # işlem anındaki fiyat
    total       = db.Column(db.Float,  nullable=False)       # qty * price
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)
