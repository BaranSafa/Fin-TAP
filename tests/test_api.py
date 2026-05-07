def test_prediction_api(client):
    response = client.get('/api/predict/AAPL')

    assert response.status_code in [200, 404]