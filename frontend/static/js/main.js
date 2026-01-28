document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-button');
    const resultCard = document.getElementById('result-card');
    let chartInstance = null;

    predictBtn.addEventListener('click', async () => {
        const ticker = document.getElementById('ticker-input').value;
        const model = document.getElementById('model-input').value;
        
        // 1. Seçili Checkbox'ları Topla
        const checkboxes = document.querySelectorAll('#feature-checkboxes input:checked');
        const features = Array.from(checkboxes).map(cb => cb.value).join(',');

        if (!ticker) {
            alert("Lütfen bir hisse senedi seçin.");
            return;
        }

        // 2. Arayüzü Hazırla (Yükleniyor...)
        predictBtn.disabled = true;
        predictBtn.innerText = "Analiz Yapılıyor...";
        resultCard.classList.add('hidden');

        try {
            // 3. API İsteği Gönder
            const url = `/api/get_data/${ticker}?model=${model}&features=${features}`;
            const response = await fetch(url);
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || "Sunucu hatası");

            // --- VERİ İŞLEME VE TARİH HESAPLAMA ---

            // A. Tarihleri Hazırla
            // Backend'den gelen son tarihi alıp üzerine 14 gün ekleyeceğiz
            const lastDateStr = data.last_known_date;
            const lastDateObj = new Date(lastDateStr);
            const futureDates = [];
            
            for (let i = 1; i <= 14; i++) {
                const d = new Date(lastDateObj);
                d.setDate(d.getDate() + i); // Her döngüde 1 gün ekle
                futureDates.push(d.toISOString().split('T')[0]); // YYYY-MM-DD formatı
            }

            // Grafik için tüm etiketler: Geçmiş Tarihler + Gelecek Tarihler
            const allLabels = [...data.validation_data.dates, ...futureDates];

            // B. Fiyat Verilerini Hazırla
            const actualPrices = data.validation_data.actual_prices;     // Gerçek (Yeşil)
            const validationPrices = data.validation_data.predicted_prices; // Test (Turuncu)
            const futurePredictions = data.future_prediction_7day;       // Gelecek (Kırmızı Liste)

            // C. Grafikte Çizgileri Birleştirme Mantığı
            // Kırmızı çizginin havada asılı kalmaması için, yeşil çizginin son noktasından başlaması lazım.
            
            // 1. Yeşil Çizgi Verisi (Gelecek kısımlar boş/null olacak)
            const datasetActual = [...actualPrices, ...new Array(14).fill(null)];

            // 2. Turuncu Çizgi Verisi (Gelecek kısımlar boş/null olacak)
            const datasetValidation = [...validationPrices, ...new Array(14).fill(null)];

            // 3. Kırmızı Çizgi Verisi (Geçmiş kısımlar boş/null olacak)
            // Başlangıç noktası olarak son gerçek fiyatı ekliyoruz, sonra tahminleri ekliyoruz.
            const datasetFuture = new Array(actualPrices.length - 1).fill(null);
            datasetFuture.push(actualPrices[actualPrices.length - 1]); // Bağlantı noktası
            datasetFuture.push(...futurePredictions); // Tahminleri ekle

            // --- SONUÇLARI GÖSTER ---

            // Metin Güncelleme (14. Günün Tahmini)
            const finalPrice = futurePredictions[futurePredictions.length - 1];
            document.getElementById('future-prediction-text').innerText = `$${finalPrice.toFixed(2)}`;
            document.getElementById('result-model-name').innerText = model;
            
            // Kartı Göster
            resultCard.classList.remove('hidden');

            // --- GRAFİĞİ ÇİZ ---
            const ctx = document.getElementById('predictionChart').getContext('2d');
            
            if (chartInstance) {
                chartInstance.destroy(); // Eski grafiği temizle
            }

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Gerçek Geçmiş Fiyat',
                            data: datasetActual,
                            borderColor: '#10b981', // Yeşil
                            backgroundColor: '#10b981',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'Modelin Geçmiş Testi',
                            data: datasetValidation,
                            borderColor: 'rgba(245, 158, 11, 0.7)', // Turuncu
                            borderWidth: 1,
                            borderDash: [5, 5], // Kesikli çizgi
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'Gelecek 14 Gün Tahmini',
                            data: datasetFuture,
                            borderColor: '#ef4444', // Kırmızı
                            backgroundColor: 'rgba(239, 68, 68, 0.1)', // Altı hafif boyalı
                            borderWidth: 3,
                            pointRadius: 3, // Noktalar görünsün
                            pointHoverRadius: 6,
                            fill: true,
                            tension: 0.4 // Hafif kavisli estetik çizgi
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
                            labels: { color: '#9ca3af' } // Gri yazı
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#9ca3af' },
                            grid: { color: '#374151' } // Koyu gri ızgara
                        },
                        y: {
                            ticks: { color: '#9ca3af' },
                            grid: { color: '#374151' }
                        }
                    }
                }
            });

        } catch (err) {
            console.error(err);
            alert("Hata oluştu: " + err.message);
        } finally {
            // İşlem bitince butonu aç
            predictBtn.disabled = false;
            predictBtn.innerText = "ANALİZ BAŞLAT";
        }
    });
});