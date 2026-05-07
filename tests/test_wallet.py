def test_wallet_page(client):
    response = client.get('/wallet')

    assert response.status_code in [200, 302]