import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List

HIRING_COST_RATIO=0.18; REVENUE_PER_HEAD=180000

@dataclass
class ChurnPrediction:
    employee_id:str; department:str; churn_prob:float
    risk_tier:str; top_driver:str; est_cost_usd:float

@dataclass
class MonthlyWorkforceForecast:
    month_iso:str; month_label:str; headcount:int
    departures_est:int; hires_needed:int; salary_cost:float
    hiring_cost:float; total_workforce_cost:float; headcount_delta:int

@dataclass
class HRSummary:
    current_headcount:int; high_risk_employees:int
    projected_departures_6m:int; projected_hires_6m:int
    total_hiring_cost_6m:float; avg_monthly_salary:float; churn_rate_pct:float

class ChurnModel:
    def __init__(self, random_state=42):
        self.rs=random_state; self._fitted=False; self._model=None

    def _featurize(self, df):
        perf_map={"below":2.0,"meets":0.0,"exceeds":-1.0}
        return np.column_stack([
            df.get("tenure_months",pd.Series([24]*len(df))).fillna(24).values,
            df.get("performance",pd.Series(["meets"]*len(df))).map(perf_map).fillna(0).values,
            df.get("workload_score",pd.Series([5.0]*len(df))).fillna(5.0).values,
            df.get("manager_score",pd.Series([7.0]*len(df))).fillna(7.0).values,
            df.get("pay_parity",pd.Series([1.0]*len(df))).fillna(1.0).values]).astype(float)

    def fit(self, workforce):
        try:
            import warnings; warnings.filterwarnings("ignore")
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            df=workforce.copy(); X=self._featurize(df)
            y=(df["churn_prob"]>0.55).astype(int)
            if y.sum()>=3:
                self._model=Pipeline([("scaler",StandardScaler()),
                    ("clf",LogisticRegression(C=1.0,class_weight="balanced",
                                             max_iter=300,random_state=self.rs))])
                self._model.fit(X,y); self._fitted=True
        except ImportError: pass
        return self

    def predict(self, workforce, avg_salary=95000):
        df=workforce.copy()
        if self._fitted and self._model:
            df["pred_churn_prob"]=self._model.predict_proba(self._featurize(df))[:,1]
        elif "churn_prob" in df.columns:
            df["pred_churn_prob"]=df["churn_prob"]
        else:
            perf_map={"below":0.65,"meets":0.35,"exceeds":0.12}
            df["pred_churn_prob"]=(df.get("performance",pd.Series(["meets"]*len(df))).map(perf_map).fillna(0.35)+
                df.get("workload_score",pd.Series([5.0]*len(df))).fillna(5.0)*0.025).clip(0.05,0.95)
        results=[]
        for _,row in df.iterrows():
            prob=float(row["pred_churn_prob"])
            tier="HIGH" if prob>0.55 else "MEDIUM" if prob>0.30 else "LOW"
            drivers={"Low performance":{"below":0.78,"meets":0.40,"exceeds":0.15}.get(row.get("performance","meets"),0.40),
                     "High workload":float(row.get("workload_score",5.0))/10,
                     "Below market pay":max(0,1.0-float(row.get("pay_parity",1.0)))*2,
                     "New hire":1.0 if float(row.get("tenure_months",24))<6 else 0.0}
            results.append(ChurnPrediction(
                employee_id=str(row.get("employee_id",_)),
                department=str(row.get("department","Unknown")),
                churn_prob=round(prob,3),risk_tier=tier,
                top_driver=max(drivers,key=drivers.get),
                est_cost_usd=round(avg_salary*HIRING_COST_RATIO+avg_salary*0.50,0)))
        return results

    def dept_risk_summary(self, predictions):
        rows=[{"department":p.department,"tier":p.risk_tier,"prob":p.churn_prob} for p in predictions]
        df=pd.DataFrame(rows)
        return df.groupby("department").agg(
            headcount=("prob","count"),avg_churn_prob=("prob","mean"),
            high_risk=("tier",lambda x:(x=="HIGH").sum())).round(3).reset_index().sort_values("avg_churn_prob",ascending=False)

class HiringForecast:
    def __init__(self, current_headcount, avg_annual_salary=95000,
                 hiring_cost_ratio=HIRING_COST_RATIO, revenue_per_head=REVENUE_PER_HEAD, horizon=6):
        self.current_hc=current_headcount; self.avg_salary=avg_annual_salary
        self.hiring_ratio=hiring_cost_ratio; self.rev_per_head=revenue_per_head; self.horizon=horizon

    def project(self, revenue_forecast, churn_predictions, risk_adj=0.0):
        high_risk=sum(1 for p in churn_predictions if p.risk_tier=="HIGH")
        medium_risk=sum(1 for p in churn_predictions if p.risk_tier=="MEDIUM")
        base_dep=round((high_risk*0.20+medium_risk*0.05)*(1+risk_adj),1)
        headcount=self.current_hc; results=[]
        for step,rev_fc in enumerate(revenue_forecast):
            target_hc=max(self.current_hc,int(rev_fc.blended_revenue*12/self.rev_per_head))
            departures=max(0,round(base_dep)); replace=departures
            growth=max(0,target_hc-headcount)
            total_hires=min(replace+growth,8)
            headcount=max(headcount-departures+total_hires,1)
            monthly_sal=headcount*(self.avg_salary/12)
            hire_cost=total_hires*self.avg_salary*self.hiring_ratio
            results.append(MonthlyWorkforceForecast(
                month_iso=rev_fc.month_iso,month_label=rev_fc.month_label,
                headcount=headcount,departures_est=departures,hires_needed=total_hires,
                salary_cost=round(monthly_sal,2),hiring_cost=round(hire_cost,2),
                total_workforce_cost=round(monthly_sal+hire_cost,2),
                headcount_delta=total_hires-departures))
        return results

    def salary_adj_factors(self, wf_forecast, current_salary_expense):
        return [round((m.salary_cost-current_salary_expense)/max(current_salary_expense,1),4)
                for m in wf_forecast]

    def summarise(self, wf_forecast, churn_predictions):
        return HRSummary(current_headcount=self.current_hc,
            high_risk_employees=sum(1 for p in churn_predictions if p.risk_tier=="HIGH"),
            projected_departures_6m=int(sum(m.departures_est for m in wf_forecast)),
            projected_hires_6m=int(sum(m.hires_needed for m in wf_forecast)),
            total_hiring_cost_6m=round(sum(m.hiring_cost for m in wf_forecast),2),
            avg_monthly_salary=round(float(np.mean([m.salary_cost for m in wf_forecast])),2),
            churn_rate_pct=round(sum(m.departures_est for m in wf_forecast)/(self.current_hc*6)*100,1))

    def generate_insights(self, hr_summary, churn_preds, dept_risk):
        insights=[]
        if hr_summary.high_risk_employees>0:
            top_dept=dept_risk.iloc[0]["department"] if not dept_risk.empty else "Engineering"
            cost_risk=hr_summary.high_risk_employees*(self.avg_salary*(HIRING_COST_RATIO+0.50))
            insights.append({"severity":"WARNING","category":"HR",
                "headline":f"{hr_summary.high_risk_employees} high-churn-risk employees — ${cost_risk:,.0f} at risk",
                "detail":f"Highest concentration in {top_dept}.",
                "action":"Run stay interviews. Review compensation parity."})
        if hr_summary.projected_hires_6m>0:
            insights.append({"severity":"INFO","category":"HR",
                "headline":f"{hr_summary.projected_hires_6m} hires needed over 6 months",
                "detail":f"Total hiring cost: ${hr_summary.total_hiring_cost_6m:,.0f}.",
                "action":"Open requisitions now — average time-to-hire is 6-8 weeks."})
        return insights
