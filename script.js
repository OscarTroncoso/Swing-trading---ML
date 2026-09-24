document.addEventListener("DOMContentLoaded", fetchRealData);

async function fetchRealData() {
  try {
    const response = await fetch(`data.json?t=${Date.now()}`);
    if (!response.ok) throw new Error("Could not load data.json");
    const data = await response.json();
    const bt = data.backtest || {};
    const plan = data.positionPlan || null;
    const ml = data.ml || {};

    setText("current-price", data.currentPrice ?? "—");
    setText("stop-loss", data.stopLoss != null ? Number(data.stopLoss).toFixed(5) : "Wait for setup");
    setText("take-profit", data.takeProfit != null ? Number(data.takeProfit).toFixed(5) : "Wait for setup");
    setText("stat-rsi", data.rsi ?? "—");
    setText("stat-vol", data.atr != null ? `${Number(data.atr).toFixed(5)} ATR` : "—");
    setText("stat-macd", data.macdHist > 0 ? "Bullish" : data.macdHist < 0 ? "Bearish" : "Neutral");
    setText("stat-adx", data.adx ?? "—");
    setText("stat-sharpe", bt.sharpe ?? "—");
    setText("stat-pf", bt.profitFactor ?? "—");
    setText("stat-dd", bt.maxDrawdown != null ? `${bt.maxDrawdown}%` : "—");
    setText("stat-trades", bt.totalTrades ?? "—");

    const sig = data.signal || "NEUTRAL (WAIT)";
    let label = "Neutral / no setup";
    let color = "#3b82f6";
    if (sig.includes("BUY")) { label = "Accepted BUY setup"; color = "#10b981"; }
    if (sig.includes("SELL")) { label = "Accepted SELL setup"; color = "#ef4444"; }
    const sigEl = document.getElementById("composite-signal");
    sigEl.innerText = label;
    sigEl.style.color = color;

    if (plan) {
      const rec = plan.recommended || {};
      const noLev = plan.noLeverageAlternative || {};
      setText("pos-equity", `€${Number(plan.equityEUR || 0).toLocaleString()}`);
      setText("pos-risk", plan.appliedRiskPct != null ? `${plan.appliedRiskPct}% (~€${Number(rec.riskAtStopEUR || 0).toFixed(2)})` : "fixed notional");
      setText("pos-size", `${Number(rec.units || 0).toLocaleString()} EUR units (${Number(rec.lots || 0).toFixed(3)} lots)`);
      setText("pos-leverage", `${Number(rec.leverage || 0).toFixed(2)}× ${rec.usesLeverage ? "(uses leverage)" : "(no leverage)"}`);
      setText("pos-no-lev", `${Number(noLev.units || 0).toLocaleString()} units / ${Number(noLev.leverage || 0).toFixed(2)}×`);
      setText("pos-cap-hit", plan.leverageCapHit ? `YES (cap ${plan.maxLeverage}×)` : `No (cap ${plan.maxLeverage}×)`);
      setText("pos-target", `€${Number(rec.targetGainEUR || 0).toFixed(2)}`);
      setText("pos-rr", `${Number(rec.rewardRisk || 0).toFixed(2)}×`);
      setText("pos-selection", plan.selection || "—");
    } else {
      ["pos-equity","pos-risk","pos-size","pos-leverage","pos-no-lev","pos-cap-hit","pos-target","pos-rr","pos-selection"].forEach(id => setText(id,"—"));
    }
    setText("ml-prob", ml.mlProbability != null ? `${(Number(ml.mlProbability)*100).toFixed(1)}%` : "—");
    const candidate = ml.candidateSignal || "NONE";
    setText("ml-decision", `${candidate} → ${ml.mlAccepted ? "ACCEPT" : candidate === "NONE" ? "WAIT" : "REJECT"}`);
    const mlbt = data.mlBacktest || ml.backtest || {};
    setText("ml-backtest", mlbt.totalReturn != null ? `${mlbt.totalReturn >= 0 ? "+" : ""}${mlbt.totalReturn}% / ${mlbt.totalTrades ?? 0} trades` : "—");
    const mlPlan = ml.positionPlan || null;
    if (mlPlan && mlPlan.recommended) {
      setText("ml-pos-size", `${Number(mlPlan.recommended.units || 0).toLocaleString()} units / €${Number(mlPlan.recommended.notionalEUR || 0).toLocaleString()}`);
      setText("ml-pos-leverage", `${Number(mlPlan.recommended.leverage || 0).toFixed(2)}× ${mlPlan.recommended.usesLeverage ? "(uses leverage)" : "(no leverage)"}`);
    } else {
      setText("ml-pos-size", ml.mlAccepted ? "Pending next-open sizing" : "No trade");
      setText("ml-pos-leverage", "—");
    }

    const dates = data.chart?.dates || [];
    const prices = data.chart?.prices || [];
    const latestDate = dates.at(-1) || String(data.date || "—").slice(0,10);
    const levelNote = data.levelsAreIndicative ? " (indicative; final at next open)" : "";
    document.getElementById("signals-table").innerHTML = `
      <tr>
        <td>${latestDate}</td>
        <td class="${sig.includes('BUY') ? 'badge-buy' : sig.includes('SELL') ? 'badge-sell' : ''}">${sig}</td>
        <td>${data.currentPrice ?? '—'}</td>
        <td>${data.stopLoss != null ? Number(data.stopLoss).toFixed(5) + levelNote : '—'}</td>
        <td>${data.takeProfit != null ? Number(data.takeProfit).toFixed(5) + levelNote : '—'}</td>
        <td>${data.rsi ?? '—'}</td>
      </tr>`;

    setText("bt-initial", `€${Number(bt.initialCapitalEUR || 0).toLocaleString()}`);
    setText("bt-final", `€${Number(bt.finalCapitalEUR || 0).toLocaleString()}`);
    const ret = document.getElementById("bt-return");
    ret.innerText = `${bt.totalReturn >= 0 ? '+' : ''}${bt.totalReturn ?? 0}%`;
    ret.style.color = bt.totalReturn >= 0 ? "var(--accent-green)" : "var(--accent-red)";
    setText("bt-winrate", `${bt.winRate ?? 0}%`);
    setText("bt-window", `${data.backtestWindow?.start ?? '—'} → ${data.backtestWindow?.end ?? '—'}`);

    const trades = data.trades || [];
    document.getElementById("backtest-table").innerHTML = trades.map(t => `
      <tr>
        <td>${String(t.entry).slice(0,10)}</td>
        <td>${String(t.exit).slice(0,10)}</td>
        <td class="${t.side === 'LONG' ? 'badge-buy' : 'badge-sell'}">${t.side}</td>
        <td>${Number(t.entryPrice).toFixed(5)}</td>
        <td>${Number(t.exitPrice).toFixed(5)}</td>
        <td>${Number(t.units || 0).toLocaleString()}</td>
        <td>${Number(t.leverage || 0).toFixed(2)}×</td>
        <td style="color:${Number(t.profitEUR)>=0?'var(--accent-green)':'var(--accent-red)'}">€${Number(t.profitEUR || 0).toFixed(2)}</td>
      </tr>`).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--text-muted)">No completed trades in the selected window.</td></tr>`;

    renderChart(dates, prices, data.chart?.upperBand || [], data.chart?.lowerBand || [], data.chart?.sma50 || []);
  } catch (error) {
    console.error("Error loading market data:", error);
    document.getElementById("signals-table").innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--accent-red)">data.json not available yet. Run the GitHub Action or scripts/update_data.py.</td></tr>`;
  }
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}

function renderChart(dates, prices, upperBand, lowerBand, sma50) {
  const ctx = document.getElementById("eurusdChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: { labels: dates, datasets: [
      { label: "EUR/USD", data: prices, borderColor: "#3b82f6", borderWidth: 2, pointRadius: 0, fill: false },
      { label: "Upper Bollinger", data: upperBand, borderColor: "rgba(239,68,68,.45)", borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false },
      { label: "Lower Bollinger", data: lowerBand, borderColor: "rgba(16,185,129,.45)", borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false },
      { label: "SMA 50", data: sma50, borderColor: "#cbd5e1", borderWidth: 1, pointRadius: 0, fill: false }
    ]},
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false } }
  });
}
