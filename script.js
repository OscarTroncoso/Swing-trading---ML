document.addEventListener("DOMContentLoaded", loadData);
const $=(id)=>document.getElementById(id); const txt=(id,v)=>{if($(id))$(id).innerText=v??"—"};
function money(v){return `€${Number(v||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`}
async function loadData(){
 try{
  const r=await fetch(`data.json?t=${Date.now()}`); if(!r.ok) throw new Error("data.json unavailable"); const d=await r.json(); const b=d.backtest||{}, p=d.positionPlan, ml=d.mlAdvisory||{};
  txt("current-price",Number(d.currentPrice).toFixed(5)); txt("trend-regime",d.trendRegime||"—"); txt("trend-score",`${d.trendScore??0}/${d.trendScoreMax??5}`); txt("vol-regime",d.volatilityRegime);
  let sig=d.signal||"NEUTRAL (WAIT)"; txt("signal",sig); $("signal").className=`value ${sig.includes("BUY")?"green":sig.includes("SELL")?"red":"blue"}`;
  txt("candidate",d.candidateSignal); txt("signal-reason",d.signalReason); txt("rsi",d.rsi); txt("macd",d.macdHist); txt("adx",d.adx); txt("dmi",`${d.plusDI??"—"} / ${d.minusDI??"—"}`); txt("ema50",d.ema50); txt("ema200",d.ema200);

  txt("ml-status",ml.candidate===false?"Sin candidato":(ml.healthy?`${ml.selectedModel||"ML"} · AUC ${ml.validationAUC??"—"}`:`Abstención · ${ml.status||"no saludable"}`));
  txt("ml-prob",ml.healthy&&ml.probability!=null?`${(Number(ml.probability)*100).toFixed(1)}%`:"—");

  if(p&&p.valid){
    txt("pos-equity",money(p.equityEUR));
    txt("pos-eur",money(p.positionEUR));
    txt("pos-lev",`x${Number(p.tradeLeverage||1).toFixed(0)} ${p.spotEquivalent?"(x1 spot-equivalent)":""}`);
    txt("pos-notional",money(p.notionalEUR));
    txt("pos-exposure",`${Number(p.accountExposureX||0).toFixed(2)}× del equity`);
    txt("pos-risk",`${money(p.riskAtStopEUR)} (${Number(p.riskAtStopPct||0).toFixed(2)}%)`);
    txt("risk-target",`${money(p.targetRiskEUR)} / ${Number(p.riskBudgetUtilizationPct||0).toFixed(1)}% usado`);
    txt("stop-source",`${p.stopSource} / ${p.trend?.volatilityRegime||"—"}`);
    txt("stop-dist",`${Number(p.stopDistanceATR||0).toFixed(2)} ATR`);
    txt("stop-loss",Number(p.stopLoss).toFixed(5));
    txt("take-profit",Number(p.takeProfit).toFixed(5));
    txt("pos-rr",`${Number(p.rewardRisk||0).toFixed(2)}R`);
  } else {
    ["pos-equity","pos-eur","pos-lev","pos-notional","pos-exposure","pos-risk","risk-target","stop-source","stop-dist","stop-loss","take-profit","pos-rr"].forEach(x=>txt(x,"—"));
  }

  txt("bt-window",`${d.backtestWindow?.start||"—"} → ${d.backtestWindow?.end||"—"}`);
  txt("bt-final",money(b.finalEquityEUR??b.finalCapitalEUR));
  txt("bt-return",`${b.totalReturn>=0?"+":""}${b.totalReturn??0}%`);
  txt("bt-win",`${b.winRate??0}%`); txt("bt-pf",b.profitFactor); txt("bt-sharpe",b.sharpe); txt("bt-dd",`${b.maxDrawdown??0}%`); txt("bt-r",`${b.expectancyR??0}R`); txt("bt-trades",`${b.totalTrades??0} cerrados / ${b.openTrades??0} abiertos`);

  $("trades").innerHTML=(d.trades||[]).slice(-20).reverse().map(t=>{
    const open=t.status==="OPEN"; const pnl=open?t.unrealizedPnLEUR:t.profitEUR; const exit=open?"OPEN":String(t.exit).slice(0,10);
    return `<tr><td>${String(t.entry).slice(0,10)}</td><td>${exit}</td><td class="${t.side==='LONG'?'green':'red'}">${t.side}</td><td>€${Number(t.positionEUR||0).toFixed(0)}</td><td>x${Number(t.tradeLeverage||1).toFixed(0)}</td><td>€${Number(t.notionalEUR||0).toFixed(0)}</td><td class="${Number(pnl)>=0?'green':'red'}">${money(pnl)}</td><td>${t.rMultiple==null?"—":Number(t.rMultiple).toFixed(2)+"R"}</td><td>${t.reason||"OPEN"}</td></tr>`;
  }).join("")||`<tr><td colspan="9">Sin operaciones.</td></tr>`;
  draw(d.chart||{});
 }catch(e){console.error(e); txt("signal","Ejecuta Update EURUSD V4 dashboard");}
}
function draw(c){const ctx=$("chart").getContext("2d"); new Chart(ctx,{type:"line",data:{labels:c.dates||[],datasets:[{label:"EUR/USD",data:c.prices||[],borderColor:"#60a5fa",borderWidth:2,pointRadius:0},{label:"EMA 50",data:c.ema50||[],borderColor:"#10b981",borderWidth:1,pointRadius:0},{label:"EMA 200",data:c.ema200||[],borderColor:"#f59e0b",borderWidth:1,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},plugins:{legend:{labels:{color:"#cbd5e1"}}},scales:{x:{ticks:{color:"#94a3b8"},grid:{color:"#1e293b"}},y:{ticks:{color:"#94a3b8"},grid:{color:"#334155"}}}}});}
