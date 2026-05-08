
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

STAGE_BENCHMARKS = {
    "Lead":        {"avg":85,"warn":60,"critical":90},
    "Qualified":   {"avg":60,"warn":45,"critical":75},
    "Proposal":    {"avg":38,"warn":30,"critical":50},
    "Negotiation": {"avg":16,"warn":12,"critical":22},
}


@dataclass
class Anomaly:
    date:str; metric:str; actual:float; expected:float
    z_score:float; direction:str; severity:str; headline:str; action:str

class AnomalyDetector:
    METRICS = ["revenue","total_expenses","net_cash_flow"]
    LABELS  = {"revenue":"Revenue","total_expenses":"Expenses","net_cash_flow":"Net Cash Flow"}

    def detect(self, history, window=6):
        anomalies = []
        hist = history.copy()
        hist["ds"] = pd.to_datetime(hist["ds"])
        for metric in self.METRICS:
            if metric not in hist.columns: continue
            vals  = hist[metric].values.astype(float)
            label = self.LABELS[metric]
            for i in range(window, len(vals)):
                wv    = vals[max(0,i-window):i]
                mu    = np.mean(wv); sigma = np.std(wv)
                if sigma < 1e-6: continue
                actual = vals[i]; z = (actual-mu)/sigma
                if abs(z) < 2.0: continue
                direction = "spike" if z>0 else "drop"
                severity  = "CRITICAL" if abs(z)>=3.0 else "WARNING"
                date_str  = hist["ds"].iloc[i].strftime("%b %Y")
                pct       = (actual-mu)/max(abs(mu),1)*100
                if metric=="revenue":
                    if direction=="drop": headline=f"Revenue {date_str} dropped {abs(pct):.0f}% below 6-month average"; action="Review pipeline velocity and check for lost deals."
                    else: headline=f"Revenue {date_str} spiked {pct:.0f}% above 6-month average"; action="Investigate — one-off deal or sustainable growth? Adjust forecast."
                elif metric=="total_expenses":
                    if direction=="spike": headline=f"Expenses {date_str} spiked {pct:.0f}% above average"; action="Audit variable cost lines. Check for unplanned headcount or vendor invoices."
                    else: headline=f"Expenses {date_str} dropped {abs(pct):.0f}% below average"; action="Verify no delayed invoices — check accounts payable for deferred costs."
                else:
                    if direction=="drop": headline=f"Net cash flow {date_str} fell {abs(pct):.0f}% below average"; action="Cash deteriorating faster than expected. Review burn rate immediately."
                    else: headline=f"Net cash flow {date_str} improved {pct:.0f}% above average"; action="Positive outlier — identify what drove it and replicate."
                anomalies.append(Anomaly(date=date_str,metric=label,actual=round(actual,2),expected=round(mu,2),z_score=round(z,2),direction=direction,severity=severity,headline=headline,action=action))
        anomalies.sort(key=lambda a:{"CRITICAL":0,"WARNING":1}[a.severity])
        return anomalies

    def summary(self, anomalies):
        return {"total":len(anomalies),"critical":sum(1 for a in anomalies if a.severity=="CRITICAL"),
                "warnings":sum(1 for a in anomalies if a.severity=="WARNING"),
                "metrics_affected":list({a.metric for a in anomalies})}


@dataclass
class EmployeeTrend:
    employee_id:str; department:str; current_risk:float
    trend:str; trend_delta:float; top_concern:str
    urgency:str; trajectory:List[float]

class EmployeeTrendTracker:
    def compute_trends(self, workforce, churn_preds, months=6):
        np.random.seed(42)
        pred_map = {p.employee_id:p for p in churn_preds}
        results  = []
        for _,row in workforce.iterrows():
            eid      = str(row.get("employee_id","?"))
            dept     = str(row.get("department","?"))
            pred     = pred_map.get(eid)
            cur_risk = float(pred.churn_prob if pred else row.get("churn_prob",0.35))
            workload = float(row.get("workload_score",5.0))
            tenure   = int(row.get("tenure_months",24))
            perf     = str(row.get("performance","meets"))
            pay      = float(row.get("pay_parity",1.0))
            drift = 0.0
            if workload>7.5: drift+=0.012
            if workload<4.0: drift-=0.008
            if perf=="below": drift+=0.015
            if perf=="exceeds": drift-=0.010
            if pay<0.90: drift+=0.010
            if tenure>48: drift-=0.005
            trajectory = []
            for m in range(months-1,-1,-1):
                point = np.clip(cur_risk-drift*m+np.random.normal(0,0.015),0.02,0.97)
                trajectory.append(round(float(point),3))
            delta = trajectory[-1]-trajectory[0]
            trend = "deteriorating" if delta>0.08 else "worsening" if delta>0.02 else "improving" if delta<-0.05 else "stable"
            urgency = "immediate" if cur_risk>0.70 and delta>0.03 else "monitor" if cur_risk>0.55 else "watch"
            concerns = {"High workload":workload/10 if workload>7 else 0,"Low performance":0.8 if perf=="below" else 0,
                        "Below market pay":max(0,1-pay)*1.5,"New employee":0.7 if tenure<6 else 0}
            top_concern = max(concerns,key=concerns.get) if any(v>0 for v in concerns.values()) else "Multiple factors"
            results.append(EmployeeTrend(employee_id=eid,department=dept,current_risk=round(cur_risk,3),
                trend=trend,trend_delta=round(delta,3),top_concern=top_concern,urgency=urgency,trajectory=trajectory))
        results.sort(key=lambda e:({"immediate":0,"monitor":1,"watch":2}[e.urgency],-e.current_risk))
        return results


@dataclass
class DealVelocityAlert:
    deal_id:str; company:str; stage:str; days_in_stage:int
    benchmark_avg:int; benchmark_warn:int; overdue_by:int
    severity:str; deal_value:float; ml_probability:float
    action:str; stall_risk:str

class DealVelocityTracker:
    def analyse(self, pipeline):
        alerts = []
        for _,row in pipeline.iterrows():
            stage   = str(row.get("stage","Lead"))
            days    = int(row.get("days_in_stage",0))
            bench   = STAGE_BENCHMARKS.get(stage,{"avg":60,"warn":45,"critical":75})
            if days<=bench["warn"]: severity="OK"
            elif days<=bench["critical"]: severity="WARNING"
            else: severity="CRITICAL"
            overdue = max(0,days-bench["avg"])
            val     = float(row.get("deal_value",0))
            prob    = float(row.get("blended_probability",row.get("probability",0.3)))
            company = str(row.get("company","?"))
            deal_id = str(row.get("deal_id","?"))
            if severity=="CRITICAL": action=f"Call {company} today — {overdue} days overdue. Re-qualify or close the opportunity."; stall="High"
            elif severity=="WARNING": action=f"Check in with {company} — approaching stall threshold. Request clear next step."; stall="Medium"
            else: action="On track"; stall="Low"
            alerts.append(DealVelocityAlert(deal_id=deal_id,company=company,stage=stage,days_in_stage=days,
                benchmark_avg=bench["avg"],benchmark_warn=bench["warn"],overdue_by=overdue,
                severity=severity,deal_value=val,ml_probability=round(prob,3),action=action,stall_risk=stall))
        alerts.sort(key=lambda a:({"CRITICAL":0,"WARNING":1,"OK":2}[a.severity],-a.overdue_by))
        return alerts

    def summary(self, alerts):
        stalled = sum(a.deal_value*a.ml_probability for a in alerts if a.severity in ("CRITICAL","WARNING"))
        return {"critical":sum(1 for a in alerts if a.severity=="CRITICAL"),
                "warning":sum(1 for a in alerts if a.severity=="WARNING"),
                "on_track":sum(1 for a in alerts if a.severity=="OK"),
                "stalled_value":round(stalled,2),"total_deals":len(alerts)}


@dataclass
class BudgetVariance:
    month:str; metric:str; budget:float; actual:float
    variance:float; variance_pct:float; status:str

class BudgetVsActual:
    def compare(self, history, forecast_months=3):
        results = []
        hist = history.copy(); hist["ds"]=pd.to_datetime(hist["ds"])
        n = len(hist)
        if n < forecast_months*2: return results
        metrics = {"revenue":"Revenue","total_expenses":"Expenses","net_cash_flow":"Net Cash Flow"}
        for metric,label in metrics.items():
            if metric not in hist.columns: continue
            vals     = hist[metric].values.astype(float)
            baseline = vals[:-forecast_months]; actuals=vals[-forecast_months:]; dates=hist["ds"].values[-forecast_months:]
            bx = np.arange(len(baseline))
            slope,intercept = np.polyfit(bx,baseline,1) if len(bx)>=2 else (0,baseline[-1] if len(baseline) else 0)
            for i,(actual,date) in enumerate(zip(actuals,dates)):
                budget=intercept+slope*(len(baseline)+i); variance=actual-budget; var_pct=variance/max(abs(budget),1)*100
                if metric=="revenue": status="Ahead of Plan" if var_pct>=5 else "On Track" if var_pct>=-5 else "Under Budget"
                elif metric=="total_expenses": status="Over Budget" if var_pct>=5 else "On Track" if var_pct>=-5 else "Under Budget"
                else: status="Ahead of Plan" if var_pct>=5 else "On Track" if var_pct>=-5 else "Under Budget"
                results.append(BudgetVariance(month=pd.Timestamp(date).strftime("%b %Y"),metric=label,
                    budget=round(float(budget),2),actual=round(float(actual),2),variance=round(float(variance),2),
                    variance_pct=round(float(var_pct),1),status=status))
        return results

    def accuracy_score(self, variances):
        if not variances: return 0.0
        return round(sum(1 for v in variances if v.status in ("On Track","Ahead of Plan"))/len(variances)*100,1)
