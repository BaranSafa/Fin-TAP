"""
models.py  —  Fin-TAP Database Models

DÜZELTİLEN KRİTİK SORUN:
  Werkzeug 3.x scrypt algoritması kullanıyor → hash = 162 karakter
  Eski: password = db.Column(db.String(150))  ← 150 < 162 → TRUNCATION → kayıt başarısız
  Yeni: password = db.Column(db.String(512))  ← her algoritma için yeterli

V0.7 EKLEMELERİ:
  - DB index'leri eklendi (user_id foreign key'leri)
  - Watchlist modeli eklendi (favori hisse takibi)
  - Prediction.accuracy_pct eklendi (gerçek fiyat geldikten sonra doldurulur)
"""
from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

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


class Wallet(db.Model):
    __tablename__ = "wallet"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, index=True)
    balance      = db.Column(db.Integer, default=5)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


class Transaction(db.Model):
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
    accuracy_pct     = db.Column(db.Float, nullable=True)   # Gerçek fiyatla karşılaştırıldıktan sonra dolar
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)


class Watchlist(db.Model):
    """Kullanıcının takip listesindeki hisseler."""
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
    direction: 'above' | 'below'
    status: 'active' | 'triggered' | 'cancelled'
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
    direction    = db.Column(db.String(10),  nullable=False)   # 'above' | 'below'
    note         = db.Column(db.String(200), nullable=True)    # kullanıcı notu
    status       = db.Column(db.String(20),  default="active") # 'active' | 'triggered' | 'cancelled'
    created_at   = db.Column(db.DateTime,    default=datetime.utcnow)
    triggered_at = db.Column(db.DateTime,    nullable=True)


PAPER_STARTING_CASH = 10_000.0


class PaperPortfolio(db.Model):
    """
    Her kullanıcı için sanal trading hesabı.
    cash: mevcut nakit bakiyesi (başlangıç: 10.000 USD)
    """
    __tablename__ = "paper_portfolio"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False, index=True)
    cash       = db.Column(db.Float,   default=PAPER_STARTING_CASH, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_at   = db.Column(db.DateTime, nullable=True)


class PaperPosition(db.Model):
    """Açık sanal pozisyon — bir kullanıcının elinde tuttuğu hisse."""
    __tablename__ = "paper_position"
    __table_args__ = (
        db.Index("ix_paper_position_user_id", "user_id"),
        db.UniqueConstraint("user_id", "symbol", name="uq_paper_pos_user_sym"),
    )

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol     = db.Column(db.String(20), nullable=False)
    quantity   = db.Column(db.Float,  nullable=False, default=0.0)   # fractional shares
    avg_cost   = db.Column(db.Float,  nullable=False, default=0.0)   # ortalama alış fiyatı
    opened_at  = db.Column(db.DateTime, default=datetime.utcnow)


class PaperTrade(db.Model):
    """Gerçekleşmiş sanal işlem kaydı."""
    __tablename__ = "paper_trade"
    __table_args__ = (
        db.Index("ix_paper_trade_user_id", "user_id"),
    )

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symbol     = db.Column(db.String(20), nullable=False)
    action     = db.Column(db.String(10), nullable=False)   # 'buy' | 'sell'
    quantity   = db.Column(db.Float,  nullable=False)
    price      = db.Column(db.Float,  nullable=False)       # işlem fiyatı
    total      = db.Column(db.Float,  nullable=False)       # qty * price
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)
