from werkzeug.security import generate_password_hash

from app import app, db
from models import User, Wallet


def test_ai_chat_answers_backtest_question():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        user = User.query.filter_by(email="ai-chat-test@example.com").first()
        if not user:
            user = User(
                email="ai-chat-test@example.com",
                name="AI Chat Test",
                password=generate_password_hash("Pass12345"),
            )
            db.session.add(user)
            db.session.flush()
            db.session.add(Wallet(user_id=user.id, balance=5))
            db.session.commit()

    with app.test_client() as client:
        client.post("/login", data={"email": "ai-chat-test@example.com", "password": "Pass12345"})
        response = client.post(
            "/api/ai/chat",
            json={"message": "Allow short selling nedir?", "page": "/backtest"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert "Backtest" in data["answer"]
    assert data["quick_links"][0]["url"] == "/backtest"
