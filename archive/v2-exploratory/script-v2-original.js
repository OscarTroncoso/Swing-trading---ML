document.addEventListener("DOMContentLoaded", fetchRealData);

async function fetchRealData() {
  try {
    const response = await fetch("data-v2.json");
    if (!response.ok) throw new Error("Could not load data-v2.json");
    const data = await response.json();
    const bt = data.backtest || {};

    document.getElementById("current-price").innerText = data.currentPrice ?? "—";
    document.getElementById("stop-loss").innerText = "Dynamic ATR";
    document.getElementById("take-profit").innerText = "Dynamic ATR";
    document.getElementById("stat-rsi").innerText = data.rsi ?? "—";
    document.getElementById("stat-vol").innerText = data.atr != null ? `${data.atr} ATR` : "—";
    document.getElementById("stat-macd").innerText =
      data.macdHist > 0 ? "Bullish" : data.macdHist < 0 ? "Bearish" : "Neutral";

    // Never present an invented probability. This is a model state, not a calibrated P(success).
    const sig = data.signal || "NEUTRAL (WAIT)";
    let label = "Neutral / no setup";
    let color = "#3b82f6";
    if (sig.includes("BUY")) { label = "Confirmed momentum BUY"; color = "#10b981"; }
    if (sig.includes("SELL")) { label = "Confirmed momentum SELL"; color = "#ef4444"; }
    const sigEl = document.getElementById("composite-signal");
    sigEl.innerText = label;
    sigEl.style.color = color;

    const dates = data.chart?.dates || [];
    const prices = data.chart?.prices || [];
    const latestDate = dates.at(-1) || data.date || "—";
    document.getElementById("signals-table").innerHTML = `
      <tr>
        <td>${latestDate}</td>
        <td class="${sig.includes('BUY') ? 'badge-buy' : sig.includes('SELL') ? 'badge-sell' : ''}">${sig}</td>
        <td>${data.currentPrice ?? '—'}</td>
        <td>ATR × ${data.parameters?.stop_atr ?? '—'}</td>
        <td>ATR × ${data.parameters?.take_atr ?? '—'}</td>
        <td>${data.rsi ?? '—'}</td>
      </tr>`;

    if (document.getElementById("bt-final")) document.getElementById("bt-final").innerText = `$${Number(bt.finalCapital || 0).toLocaleString()}`;
    if (document.getElementById("bt-return")) document.getElementById("bt-return").innerText = `${bt.totalReturn >= 0 ? '+' : ''}${bt.totalReturn ?? 0}%`;
    if (document.getElementById("bt-winrate")) document.getElementById("bt-winrate").innerText = `${bt.winRate ?? 0}%`;

    if (document.getElementById("backtest-table")) {
      document.getElementById("backtest-table").innerHTML = (data.trades || []).map(t => `
        <tr>
          <td>${String(t.entry).slice(0,10)}</td><td>${String(t.exit).slice(0,10)}</td>
          <td class="${t.side === 'LONG' ? 'badge-buy' : 'badge-sell'}">${t.side}</td>
          <td>${Number(t.entryPrice).toFixed(5)}</td><td>${Number(t.exitPrice).toFixed(5)}</td>
          <td>${Number(t.profit).toFixed(2)}</td>
        </tr>`).join("") || `<tr><td colspan="6">No completed trades.</td></tr>`;
    }

    renderChart(
      dates, prices,
      data.chart?.upperBand || [],
      data.chart?.lowerBand || [],
      data.chart?.sma50 || []
    );
  } catch (error) {
    console.error("Error loading market data:", error);
  }
}

function renderChart(dates, prices, upperBand, lowerBand, sma50) {
  const ctx = document.getElementById("eurusdChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        { label: "EUR/USD", data: prices, borderColor: "#3b82f6", borderWidth: 2, pointRadius: 0, fill: false },
        { label: "Upper Bollinger", data: upperBand, borderColor: "rgba(239,68,68,.45)", borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false },
        { label: "Lower Bollinger", data: lowerBand, borderColor: "rgba(16,185,129,.45)", borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false },
        { label: "SMA 50", data: sma50, borderColor: "#cbd5e1", borderWidth: 1, pointRadius: 0, fill: false }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}
