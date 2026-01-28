document.addEventListener('DOMContentLoaded', async () => {
    const ctx = document.getElementById('marketChart').getContext('2d');
    const tickerSelect = document.getElementById('dashboard-ticker-select');
    const chartTitle = document.getElementById('chart-title');
    const marketCardsContainer = document.getElementById('market-cards');
    let chartInstance = null;

    // --- 1. PİYASA ÖZET KARTLARINI DOLDUR ---
    async function loadMarketSummary() {
        try {
            const response = await fetch('/api/market_summary');
            const data = await response.json();
            
            marketCardsContainer.innerHTML = ''; // Loading animasyonunu temizle

            data.forEach(item => {
                const isUp = item.trend === 'up';
                const colorClass = isUp ? 'text-green-400' : 'text-red-400';
                const arrow = isUp ? '▲' : '▼';
                
                const cardHtml = `
                    <div class="bg-gray-800 p-4 rounded-lg shadow border-l-4 ${isUp ? 'border-green-500' : 'border-red-500'} hover:bg-gray-700 transition cursor-pointer" onclick="changeGraph('${item.ticker}')">
                        <div class="flex justify-between items-center">
                            <h3 class="font-bold text-gray-200">${item.ticker}</h3>
                            <span class="text-xs text-gray-500">Günlük</span>
                        </div>
                        <div class="mt-2 flex items-end justify-between">
                            <span class="text-xl font-bold text-white">$${item.price}</span>
                            <span class="${colorClass} text-sm font-medium flex items-center">
                                ${arrow} %${Math.abs(item.change)}
                            </span>
                        </div>
                    </div>
                `;
                marketCardsContainer.innerHTML += cardHtml;
            });

        } catch (err) {
            console.error("Özet yüklenemedi:", err);
            marketCardsContainer.innerHTML = '<p class="text-red-500 col-span-3">Piyasa verileri alınamadı.</p>';
        }
    }

    // --- 2. GRAFİĞİ GÜNCELLE ---
    async function loadChart(ticker) {
        chartTitle.innerText = ticker;
        
        // Varsa eski grafiği yok et
        if (chartInstance) {
            chartInstance.destroy();
        }

        try {
            const response = await fetch(`/api/history/${ticker}`);
            const data = await response.json();

            if (data.error) return;

            // Trend rengini belirle (Son fiyat > İlk fiyat ise Yeşil, yoksa Kırmızı)
            const isUp = data.prices[data.prices.length - 1] > data.prices[0];
            const lineColor = isUp ? '#10b981' : '#ef4444'; // Yeşil veya Kırmızı
            const areaColor = isUp ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [{
                        label: `${ticker} Fiyatı`,
                        data: data.prices,
                        borderColor: lineColor,
                        backgroundColor: areaColor,
                        borderWidth: 2,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { display: false },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { ticks: { color: '#6b7280', maxTicksLimit: 6 }, grid: { display: false } },
                        y: { ticks: { color: '#6b7280' }, grid: { color: '#374151' } }
                    }
                }
            });
        } catch (err) {
            console.error("Grafik hatası:", err);
        }
    }

    // --- 3. EVENT LISTENERS ---
    
    // Dropdown değişince grafiği güncelle
    tickerSelect.addEventListener('change', (e) => {
        loadChart(e.target.value);
    });

    // Kartlara tıklayınca grafiği güncellemek için global fonksiyon
    window.changeGraph = (ticker) => {
        tickerSelect.value = ticker; // Dropdown'u da güncelle
        loadChart(ticker);
    };

    // Başlangıçta çalıştır
    loadMarketSummary();
    loadChart(tickerSelect.value || 'AAPL'); // Varsayılan veya ilk sıradaki
});