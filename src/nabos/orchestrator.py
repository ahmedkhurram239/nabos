import time, logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd

from .data_generator import generate_financial_history, generate_pipeline, generate_deal_history, generate_workforce
from .sales_engine import DealMLModel, RevenueEngine, MonthlyRevenueForecast, MLMetrics
from .finance_engine import ExpenseEngine, CashFlowEngine, MonthlyExpenseForecast, MonthlyCashFlow, FinanceSummary
from .hr_engine import ChurnModel, HiringForecast, ChurnPrediction, MonthlyWorkforceForecast, HRSummary
from .risk_engine import RiskEngine, RiskProfile, RiskAdjustment, RiskInsight

logger=logging.getLogger(__name__)

@dataclass
class ScenarioOutput:
    label:str; color:str; risk_score:float; risk_grade:str
    revenue_fc:List; expense_fc:List; cashflow:List; workforce_fc:List
    finance_summary:FinanceSummary; hr_summary:HRSummary
    risk_adj:Optional[RiskAdjustment]; risk_insights:List; all_insights:List

@dataclass
class NABOSResult:
    history:pd.DataFrame; pipeline:pd.DataFrame; deal_history:pd.DataFrame; workforce:pd.DataFrame
    ml_metrics:Optional[MLMetrics]; churn_preds:List; dept_risk:pd.DataFrame
    revenue_fc:List; expense_fc:List; cashflow:List
    workforce_fc:List; hr_summary:HRSummary; finance_summary:FinanceSummary
    all_insights:List; scenarios:Dict; base_scenario:ScenarioOutput
    generated_at:str=""; duration_s:float=0.0; company_name:str="Nestlé Pakistan"

def run_full_forecast(financial_csv=None, pipeline_csv=None, deal_hist_csv=None,
                      workforce_csv=None, starting_cash=500000, horizon=6, risk_profile=None):
    t0=time.time()
    def load(path, fn, **kw):
        if path and Path(path).exists(): return pd.read_csv(path)
        return fn(**kw)

    history=load(financial_csv, generate_financial_history, months=24)
    if "ds" in history.columns: history["ds"]=pd.to_datetime(history["ds"])
    pipeline=load(pipeline_csv, generate_pipeline, n=55)
    deal_hist=load(deal_hist_csv, generate_deal_history, n=320)
    workforce=load(workforce_csv, generate_workforce, n_employees=52)

    avg_salary=float(history["salaries"].iloc[-3:].mean()/history["headcount"].iloc[-3:].mean()*12) if "headcount" in history.columns else 95000
    avg_salary=max(avg_salary,75000)

    deal_model=DealMLModel(); deal_model.fit(deal_hist)
    pipeline=deal_model.score(pipeline); ml_metrics=deal_model.metrics()

    churn_model=ChurnModel(); churn_model.fit(workforce)
    churn_preds=churn_model.predict(workforce, avg_salary=avg_salary)
    dept_risk=churn_model.dept_risk_summary(churn_preds)

    risk_engine=RiskEngine()
    base_profile=risk_profile or RiskProfile(label="Base Case", color="#6366f1")
    base_adj=risk_engine.compute(base_profile); base_score,base_grade=risk_engine.score(base_profile)

    rev_engine=RevenueEngine(horizon=horizon); rev_engine.fit(history)
    revenue_fc=rev_engine.forecast(history, pipeline, base_adj.rev_delta, base_adj.conv_delta)

    current_hc=int(history["headcount"].iloc[-1]) if "headcount" in history.columns else 30
    hiring_eng=HiringForecast(current_hc, avg_salary, horizon=horizon)
    workforce_fc=hiring_eng.project(revenue_fc, churn_preds, base_adj.hr_adj)
    hr_summary=hiring_eng.summarise(workforce_fc, churn_preds)

    exp_engine=ExpenseEngine(horizon=horizon); exp_engine.fit(history)
    hc_adj=hiring_eng.salary_adj_factors(workforce_fc, float(history["salaries"].iloc[-1]))
    expense_fc=exp_engine.scenario_shift(history, base_adj.exp_delta, hc_adj)

    cf_engine=CashFlowEngine(starting_cash=starting_cash)
    cashflow=cf_engine.integrate(revenue_fc, expense_fc)
    fin_summary=cf_engine.summarise(cashflow)

    fin_ins=cf_engine.generate_insights(cashflow, fin_summary)
    hr_ins=hiring_eng.generate_insights(hr_summary, churn_preds, dept_risk)
    risk_ins=risk_engine.generate_insights(base_profile, base_adj, fin_summary.total_revenue_6m, fin_summary.total_expenses_6m)
    risk_dicts=[{"severity":r.severity,"category":f"Risk/{r.driver}","headline":r.headline,"detail":r.detail,"action":r.action} for r in risk_ins]
    all_ins=sorted(fin_ins+hr_ins+risk_dicts, key=lambda i:{"CRITICAL":0,"WARNING":1,"INFO":2}.get(i.get("severity","INFO"),3))

    base_sc=ScenarioOutput(label="Base",color="#6366f1",risk_score=base_score,risk_grade=base_grade,
        revenue_fc=revenue_fc,expense_fc=expense_fc,cashflow=cashflow,workforce_fc=workforce_fc,
        finance_summary=fin_summary,hr_summary=hr_summary,risk_adj=base_adj,
        risk_insights=risk_ins,all_insights=all_ins)

    from .risk_engine import low_risk_profile, elevated_risk_profile, high_risk_profile, crisis_profile
    named={"Base":base_profile,"Low Risk":low_risk_profile(),"Elevated":elevated_risk_profile(),
           "High Risk":high_risk_profile(),"Crisis":crisis_profile()}
    scenarios={}
    for name,prof in named.items():
        adj=risk_engine.compute(prof); sc,sg=risk_engine.score(prof)
        r_fc=rev_engine.forecast(history, pipeline, adj.rev_delta, adj.conv_delta)
        w_fc=hiring_eng.project(r_fc, churn_preds, adj.hr_adj)
        hca=hiring_eng.salary_adj_factors(w_fc, float(history["salaries"].iloc[-1]))
        e_fc=exp_engine.scenario_shift(history, adj.exp_delta, hca)
        cf=cf_engine.integrate(r_fc, e_fc); fs=cf_engine.summarise(cf)
        hs=hiring_eng.summarise(w_fc, churn_preds)
        ri=risk_engine.generate_insights(prof, adj, fs.total_revenue_6m, fs.total_expenses_6m)
        ri_d=[{"severity":x.severity,"category":f"Risk/{x.driver}","headline":x.headline,"detail":x.detail,"action":x.action} for x in ri]
        si=sorted(cf_engine.generate_insights(cf,fs)+hiring_eng.generate_insights(hs,churn_preds,dept_risk)+ri_d,
                  key=lambda i:{"CRITICAL":0,"WARNING":1,"INFO":2}.get(i.get("severity","INFO"),3))
        scenarios[name]=ScenarioOutput(label=name,color=prof.color,risk_score=sc,risk_grade=sg,
            revenue_fc=r_fc,expense_fc=e_fc,cashflow=cf,workforce_fc=w_fc,
            finance_summary=fs,hr_summary=hs,risk_adj=adj,risk_insights=ri,all_insights=si)

    return NABOSResult(history=history,pipeline=pipeline,deal_history=deal_hist,workforce=workforce,
        ml_metrics=ml_metrics,churn_preds=churn_preds,dept_risk=dept_risk,
        revenue_fc=revenue_fc,expense_fc=expense_fc,cashflow=cashflow,
        workforce_fc=workforce_fc,hr_summary=hr_summary,finance_summary=fin_summary,
        all_insights=all_ins,scenarios=scenarios,base_scenario=base_sc,
        generated_at=datetime.now().isoformat(),duration_s=round(time.time()-t0,2))
