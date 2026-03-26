from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

# db nesnesini burada oluşturuyoruz
db = SQLAlchemy()

# 1. KULLANICI TABLOSU
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # İlişkiler
    wallet = db.relationship('Wallet', backref='user', uselist=False)
    transactions = db.relationship('Transaction', backref='user', lazy=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

# 2. CÜZDAN (WALLET) TABLOSU
class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    balance = db.Column(db.Integer, default=5)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# 3. İŞLEM GEÇMİŞİ TABLOSU
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    amount_paid = db.Column(db.Float, nullable=False)
    tokens_added = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# 4. TAHMİN GEÇMİŞİ TABLOSU (Eksik olan buydu!)
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    symbol = db.Column(db.String(20), nullable=False)
    model_type = db.Column(db.String(50))
    predicted_result = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)