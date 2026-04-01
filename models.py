"""
models.py  —  Fin-TAP Database Models

DÜZELTİLEN KRİTİK SORUN:
  Werkzeug 3.x scrypt algoritması kullanıyor → hash = 162 karakter
  Eski: password = db.Column(db.String(150))  ← 150 < 162 → TRUNCATION → kayıt başarısız
  Yeni: password = db.Column(db.String(512))  ← her algoritma için yeterli
"""
from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(255), unique=True, nullable=False)
    password   = db.Column(db.String(512), nullable=False)   # 150 → 512 (scrypt hash)
    name       = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    wallet       = db.relationship("Wallet",      backref="user", uselist=False)
    transactions = db.relationship("Transaction", backref="user", lazy=True)
    predictions  = db.relationship("Prediction",  backref="user", lazy=True)


class Wallet(db.Model):
    __tablename__ = "wallet"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True)
    balance      = db.Column(db.Integer, default=5)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


class Transaction(db.Model):
    __tablename__ = "transaction"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"))
    amount_paid  = db.Column(db.Float,   nullable=False)
    tokens_added = db.Column(db.Integer, nullable=False)
    date         = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    __tablename__ = "prediction"

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("user.id"))
    symbol           = db.Column(db.String(20),  nullable=False)
    model_type       = db.Column(db.String(50))
    predicted_result = db.Column(db.String(100))
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
