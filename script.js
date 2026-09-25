document.addEventListener("DOMContentLoaded", loadData);
const $=(id)=>document.getElementById(id);
const txt=(id,v)=>{if($(id))$(id).innerText=v??"—"};
const money=(v)=>`€${Number(v||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;

async function loadData(){
  try{
    const r=await fetch(`data.json?t=${Date.now()}`);
    if(!r.ok) throw new Error("data.json unavailable");
    const d=await r.json();
    const b=d.backtest||{}, p=d.positionPlan, ml=d.ml||{}, open=d.openTrade;

    txt("model-version",d.modelVersion||"3.3");
    txt("current-price",Number(d.currentPrice).toFixed(5));
    let sig=d.signal||"NEUTRAL (WAIT)";
    txt("signal",sig);
    $("signal").className=`value ${sig.includes("BUY")?"green":sig.includes("SELL")?"red":"blue"}`;
    txt("trend-regime",d.trendRegime||"—");
    txt("trend-score",`${d.trendScore??0}/5`);
    txt("vol-regime",d.volatilityRegime||"—");
    txt("exit-policy",d.exitPolicy||"TP/SL only");
    txt("rsi",d.rsi);
    txt("macd",d.macdHist);
    txt("adx",d.adx);
    txt("ml-prob",ml.latestProbability==null?"No activa":`${(Number(ml.latestProbability)*100).toFixed(1)}%`);
    txt("ml-model",ml.lastModelMeta?.selectedModel||ml.lastModelMeta?.status||"Sin modelo sano");

    if(p&&p.valid){
      txt("pos-equity",money(p.equityEUR));
      txt("pos-stake",money(p.stakeEUR));
      txt("pos-exposure",money(p.grossExposureEUR));
      txt("pos-lev",`x${p.leverage} (rango ${p.leverageRange})`);
      txt("pos-risk",`${money(p.riskAtStopEUR)} / ${Number(p.riskAtStopPct||0).toFixed(2)}%`);
      txt("pos-risk-budget",money(p.riskBudgetEUR));
      txt("stop-loss",Number(p.stopLoss).toFixed(5));
      txt("take-profit",Number(p.takeProfit).toFixed(5));
      txt("pos-rr",Number(p.rewardRisk||0).toFixed(2));
      txt("lev-method",p.selectionMethod||"—");
    }else{
      ["pos-equity","pos-stake","pos-exposure","pos-lev","pos-risk","pos-risk-budget","stop-loss","take-profit","pos-rr","lev-method"].forEach(x=>txt(x,"—"));
    }

    txt("bt-window",`${d.backtestWindow?.start||"—"} → ${d.backtestWindow?.end||"—"}`);
    txt("bt-final",money(b.finalCapitalEUR));
    txt("bt-realized",money(b.realizedCapitalEUR));
    txt("bt-return",`${b.totalReturn>=0?"+":""}${b.totalReturn??0}%`);
    txt("bt-win",`${b.winRate??0}%`);
    txt("bt-pf",b.profitFactor);
    txt("bt-sharpe",b.sharpe);
    txt("bt-dd",`${b.maxDrawdown??0}%`);
    txt("bt-trades",b.totalTrades??0);
    txt("bt-open",b.openPositions??0);

    if(open){
      $("open-wrap").style.display="block";
      txt("open-side",open.side);
      txt("open-entry",open.entryPrice);
      txt("open-current",open.currentPrice);
      txt("open-stake",money(open.stakeEUR));
      txt("open-exposure",money(open.grossExposureEUR));
      txt("open-lev",`x${open.leverage}`);
      txt("open-sl",open.stopLoss);
      txt("open-tp",open.takeProfit);
      txt("open-pnl",money(open.unrealizedPnLEUR));
    }else if($("open-wrap")) $("open-wrap").style.display="none";

    $("trades").innerHTML=(d.trades||[]).slice(-20).reverse().map(t=>`<tr>
      <td>${String(t.entry).slice(0,10)}</td><td>${String(t.exit).slice(0,10)}</td>
      <td class="${t.side==='LONG'?'green':'red'}">${t.side}</td>
      <td>${money(t.stakeEUR)}</td><td>${money(t.grossExposureEUR)}</td><td>x${t.leverage}</td>
      <td>${money(t.initialRiskEUR)}</td>
      <td class="${Number(t.profitEUR)>=0?'green':'red'}">${money(t.profitEUR)}</td>
      <td>${Number(t.rMultiple||0).toFixed(2)}R</td><td>${t.reason}</td></tr>`).join("")||`<tr><td colspan="10">Sin operaciones completadas.</td></tr>`;

    draw(d.chart||{});
  }catch(e){
    console.error(e);
    txt("signal","Ejecuta Update EURUSD dashboard");
  }
}

function draw(c){
  const ctx=$("chart").getContext("2d");
  new Chart(ctx,{type:"line",data:{labels:c.dates||[],datasets:[
    {label:"EUR/USD",data:c.prices||[],borderColor:"#60a5fa",borderWidth:2,pointRadius:0},
    {label:"EMA 50",data:c.ema50||[],borderColor:"#10b981",borderWidth:1,pointRadius:0},
    {label:"EMA 200",data:c.ema200||[],borderColor:"#f59e0b",borderWidth:1,pointRadius:0}
  ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
    plugins:{legend:{labels:{color:"#cbd5e1"}}},
    scales:{x:{ticks:{color:"#94a3b8"},grid:{color:"#1e293b"}},y:{ticks:{color:"#94a3b8"},grid:{color:"#334155"}}}
  }});
}
