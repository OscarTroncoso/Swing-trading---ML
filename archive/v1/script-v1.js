document.addEventListener("DOMContentLoaded", () => {
    fetchRealData();
});

async function fetchRealData() {
    try {
        const response = await fetch('data.json');
        if (!response.ok) throw new Error('Could not load data.json');
        
        const data = await response.json();
        
        const dates = data.dates;
        const prices = data.prices;
        const currentPrice = data.currentPrice;
        const upperBand = data.upperBand;
        const lowerBand = data.lowerBand;
        const sma = data.sma;
        const rsi = data.rsi;
        const signal = data.signal;
        const stopLoss = data.stopLoss;
        const takeProfit = data.takeProfit;
        
        let compositeSignalText = "50% Range Bound";
        let compositeColor = "#3b82f6";

        if (signal.includes("BUY")) {
            compositeSignalText = "78% Strong Buy";
            compositeColor = "#10b981";
        } else if (signal.includes("SELL")) {
            compositeSignalText = "72% Strong Sell";
            compositeColor = "#ef4444";
        }

        // Update DOM Metrics
        document.getElementById("current-price").innerText = currentPrice;
        document.getElementById("composite-signal").innerText = compositeSignalText;
        document.getElementById("composite-signal").style.color = compositeColor;
        document.getElementById("stop-loss").innerText = stopLoss;
        document.getElementById("take-profit").innerText = takeProfit;
        
        const stDev = (upperBand - sma) / 2;
        document.getElementById("stat-vol").innerText = (stDev * 100).toFixed(3) + "%";
        document.getElementById("stat-rsi").innerText = rsi;
        document.getElementById("stat-macd").innerText = rsi > 55 ? "Bullish Crossover" : (rsi < 45 ? "Bearish Crossover" : "Consolidation");

        // Populate Signals Table
        let tableHtml = `
            <tr>
                <td>${dates[dates.length - 1]}</td>
                <td class="${signal.includes('BUY') ? 'badge-buy' : (signal.includes('SELL') ? 'badge-sell' : '')}">${signal}</td>
                <td>${currentPrice}</td>
                <td style="color: var(--accent-red);">${stopLoss}</td>
                <td style="color: var(--accent-green);">${takeProfit}</td>
                <td>${rsi}</td>
            </tr>
        `;
        document.getElementById("signals-table").innerHTML = tableHtml;

        // Backtest Data Rendering
        const bt = data.backtest;
        if (bt) {
            document.getElementById("bt-final").innerText = "€" + bt.finalCapital.toLocaleString();
            const returnElem = document.getElementById("bt-return");
            returnElem.innerText = (bt.totalReturn >= 0 ? "+" : "") + bt.totalReturn + "%";
            returnElem.style.color = bt.totalReturn >= 0 ? "var(--accent-green)" : "var(--accent-red)";
            
            document.getElementById("bt-winrate").innerText = bt.winRate + "%";

            let btTableHtml = "";
            if (bt.trades && bt.trades.length > 0) {
                bt.trades.forEach(t => {
                    const profitColor = t.profit >= 0 ? "var(--accent-green)" : "var(--accent-red)";
                    btTableHtml += `
                        <tr>
                            <td>${t.entry}</td>
                            <td>${t.exit}</td>
                            <td class="${t.type === 'BUY' ? 'badge-buy' : 'badge-sell'}">${t.type}</td>
                            <td>${t.entryPrice}</td>
                            <td>${t.exitPrice}</td>
                            <td style="color: ${profitColor}; font-weight: bold;">€${t.profit.toLocaleString()}</td>
                        </tr>
                    `;
                });
            } else {
                btTableHtml = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No completed trades in current period.</td></tr>`;
            }
            document.getElementById("backtest-table").innerHTML = btTableHtml;
        }

        // Render Chart
        renderChart(dates, prices, upperBand, lowerBand, sma);

    } catch (error) {
        console.error("Error loading market data:", error);
        document.getElementById("signals-table").innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--accent-red);">Waiting for GitHub Actions update (data.json missing)...</td></tr>`;
    }
}

function renderChart(dates, prices, upperBand, lowerBand, sma) {
    const ctx = document.getElementById('eurusdChart').getContext('2d');
    
    const upperArr = prices.map(() => upperBand);
    const lowerArr = prices.map(() => lowerBand);
    const smaArr = prices.map(() => sma);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'EUR/USD Rate',
                    data: prices,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'Upper Bollinger',
                    data: upperArr,
                    borderColor: 'rgba(239, 68, 68, 0.4)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Lower Bollinger',
                    data: lowerArr,
                    borderColor: 'rgba(16, 185, 129, 0.4)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'SMA (14)',
                    data: smaArr,
                    borderColor: '#cbd5e1',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc', font: { size: 11 } }
                }
            },
            scales: {
                x: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}
