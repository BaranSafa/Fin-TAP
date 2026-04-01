document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-button');
    const resultCard = document.getElementById('result-card');
    const predictionText = document.getElementById('future-prediction-text');
    const modelNameText = document.getElementById('result-model-name');
    const chartCanvas = document.getElementById('predictionChart');
    let chartInstance = null;

    if (predictBtn) {
        predictBtn.addEventListener('click', async () => {
            // 1. Verileri Topla
            const ticker = document.getElementById('ticker-input').value;
            const model = document.getElementById('model-input').value;
            
            // Seçili özellikleri (checkbox) al
            const features = [];
            document.querySelectorAll('#feature-checkboxes input:checked').forEach(cb => {
                features.push(cb.value);
            });

            // 2. Butonu "Yükleniyor" yap
            const originalText = predictBtn.innerText;
            predictBtn.innerText = "Analiz Yapılıyor...";
            predictBtn.disabled = true;
            predictBtn.classList.add('opacity-50', 'cursor-not-allowed');

            try {
                // 3. API'ye İstek At (Doğru Adres: /api/predict_run)
                const response = await fetch('/api/predict_run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker, model, features })
                });

                // HTML dönerse hata ver (Senin aldığın hatayı yakalar)
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") === -1) {
                    throw new Error("Sunucu JSON yerine HTML döndürdü. Backend hatası olabilir.");
                }

                const data = await response.json();

                if (data.error) {
                    alert("Hata: " + data.error);
                } else {
                    // 4. Sonuçları Göster
                    resultCard.classList.remove('hidden');
                    predictionText.innerText = `$${data.prediction}`;
                    modelNameText.innerText = model;

                    // Grafiği Çiz
                    drawChart(data.chart_data);

                    // Bakiyeyi anlık güncelle (sayfa yenilemeye gerek yok)
                    if (data.balance !== undefined) {
                        const tokenCount = document.querySelector('.token-count');
                        if (tokenCount) {
                            tokenCount.textContent = data.balance;
                        }
                        // Bakiye 0 veya az ise uyarı bandını göster
                        const zeroBanner = document.getElementById('zero-token-banner');
                        const lowBanner  = document.getElementById('low-token-banner');
                        if (zeroBanner) zeroBanner.style.display = data.balance === 0 ? 'flex' : 'none';
                        if (lowBanner)  lowBanner.style.display  = (data.balance > 0 && data.balance <= 2) ? 'flex' : 'none';
                    }
                }

            } catch (err) {
                console.error(err);
                alert("Bir hata oluştu: " + err.message);
            } finally {
                // Butonu eski haline getir
                predictBtn.innerText = originalText;
                predictBtn.disabled = false;
                predictBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        });
    }

    function drawChart(chartData) {
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        const ctx = chartCanvas.getContext('2d');
        
        // Verileri hazırla
        const labels = chartData.dates;
        const actualPrices = chartData.actual_prices;
        const predictedPrices = chartData.predicted_prices;
        
        // Grafik verilerini birleştir (Geçmiş + Gelecek gibi düşünebiliriz ama 
        // burada basitçe test verisi ve model tahmini kıyaslaması yapıyoruz)
        
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Gerçek Fiyat',
                        data: actualPrices,
                        borderColor: '#10b981', // Yeşil
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Model Tahmini',
                        data: predictedPrices,
                        borderColor: '#f59e0b', // Turuncu
                        borderDash: [5, 5],
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: { color: '#9ca3af' }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#6b7280', maxTicksLimit: 8 }
                    },
                    y: {
                        grid: { color: '#374151' },
                        ticks: { color: '#6b7280' }
                    }
                }
            }
        });
    }
});