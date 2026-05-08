
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

HIRING_COST_RATIO = 0.18
REVENUE_PER_HEAD  = 180_000
PERF_TO_RISK = {"below": 0.78, "meets": 0.40, "exceeds": 0.15}
LOW_ATTENDANCE_THRESHOLD = 0.85
HIGH_LATE_THRESHOLD      = 0.15
MON_FRI_RISK_THRESHOLD   = 0.25


def generate_attendance(workforce_df, months=6, ref_date="2025-01-01", seed=42):
    np.random.seed(seed)
    rows = []
    ref = pd.Timestamp(ref_date)
    work_days = pd.bdate_range(ref, periods=months * 22)
    for _, emp in workforce_df.iterrows():
        eid  = str(emp.get("employee_id", "EMP-0001"))
        cp   = float(emp.get("churn_prob", 0.35))
        tier = "HIGH" if cp > 0.55 else "MEDIUM" if cp > 0.30 else "LOW"
        base_att  = {"HIGH": 0.78, "MEDIUM": 0.88, "LOW": 0.96}[tier]
        late_rate = {"HIGH": 0.22, "MEDIUM": 0.12, "LOW": 0.04}[tier]
        mon_fri   = {"HIGH": 0.40, "MEDIUM": 0.25, "LOW": 0.08}[tier]
        slope     = {"HIGH": -0.004, "MEDIUM": 0.001, "LOW": 0.002}[tier]
        streak    = 0
        for di, day in enumerate(work_days):
            is_mf = day.weekday() in (0, 4)
            month_f = 1 + slope * (di // 22)
            att_p = float(np.clip(base_att * month_f, 0.40, 0.99))
            if is_mf: att_p *= (1 - mon_fri * 0.3)
            if streak >= 2: att_p *= 0.60
            r = np.random.random()
            if r > att_p:
                status = "absent"; check_in = None; streak += 1
            elif r > att_p * (1 - late_rate):
                lm = int(np.random.normal(25, 12))
                status = "late"; check_in = f"{9+lm//60:02d}:{lm%60:02d}"; streak = 0
            elif np.random.random() < 0.08:
                status = "wfh"; check_in = f"09:{np.random.randint(0,15):02d}"; streak = 0
            else:
                status = "present"; check_in = f"08:{np.random.randint(45,60):02d}"; streak = 0
            rows.append({"employee_id": eid, "date": day.strftime("%Y-%m-%d"),
                         "day_of_week": day.strftime("%A"), "month": day.strftime("%Y-%m"),
                         "status": status, "check_in": check_in})
    return pd.DataFrame(rows)


@dataclass
class AttendanceProfile:
    employee_id: str; attendance_rate: float; late_rate: float
    max_streak: int;  mon_fri_ratio: float;   trend_30d: float
    absence_days: int; total_working_days: int
    risk_flag: str;   risk_score: float


class AttendanceAnalyzer:
    def compute_profiles(self, attendance_df):
        profiles = {}
        att = attendance_df.copy()
        att["date"] = pd.to_datetime(att["date"])
        att = att.sort_values(["employee_id","date"])
        for eid, grp in att.groupby("employee_id"):
            total  = len(grp)
            absent = (grp["status"] == "absent").sum()
            late   = (grp["status"] == "late").sum()
            present= total - absent
            att_rate  = round(present / max(total,1), 4)
            late_rate = round(late    / max(total,1), 4)
            streak = mx = 0
            for s in grp["status"]:
                streak = (streak+1) if s == "absent" else 0
                mx = max(mx, streak)
            ab_days = grp[grp["status"]=="absent"]
            mf_ab   = ab_days[ab_days["day_of_week"].isin(["Monday","Friday"])]
            mon_fri = round(len(mf_ab)/max(len(ab_days),1), 4)
            if total >= 60:
                r30 = grp.tail(30); p30 = grp.iloc[-60:-30]
                trend = round(float((r30["status"]!="absent").mean()-(p30["status"]!="absent").mean()), 4)
            else:
                trend = 0.0
            risk  = 0.0
            risk += max(0, (LOW_ATTENDANCE_THRESHOLD - att_rate) / LOW_ATTENDANCE_THRESHOLD) * 0.40
            risk += min(late_rate / HIGH_LATE_THRESHOLD, 1.0) * 0.15
            risk += min(mx / 5, 1.0) * 0.20
            risk += min(mon_fri / MON_FRI_RISK_THRESHOLD, 1.0) * 0.10
            risk += max(0, -trend) * 2.0 * 0.15
            risk  = float(np.clip(risk, 0.0, 1.0))
            flag  = "CRITICAL" if risk>0.65 else "ALERT" if risk>0.40 else "WATCH" if risk>0.20 else "OK"
            profiles[str(eid)] = AttendanceProfile(
                employee_id=str(eid), attendance_rate=att_rate, late_rate=late_rate,
                max_streak=int(mx), mon_fri_ratio=mon_fri, trend_30d=trend,
                absence_days=int(absent), total_working_days=int(total),
                risk_flag=flag, risk_score=round(risk,4))
        return profiles

    def enrich_workforce(self, workforce_df, attendance_df):
        profiles = self.compute_profiles(attendance_df)
        df = workforce_df.copy()
        df["att_rate"]       = df["employee_id"].map(lambda e: profiles[e].attendance_rate if e in profiles else 0.92)
        df["att_late_rate"]  = df["employee_id"].map(lambda e: profiles[e].late_rate if e in profiles else 0.05)
        df["att_max_streak"] = df["employee_id"].map(lambda e: float(profiles[e].max_streak) if e in profiles else 0.0)
        df["att_mon_fri"]    = df["employee_id"].map(lambda e: profiles[e].mon_fri_ratio if e in profiles else 0.0)
        df["att_trend"]      = df["employee_id"].map(lambda e: profiles[e].trend_30d if e in profiles else 0.0)
        df["att_risk_score"] = df["employee_id"].map(lambda e: profiles[e].risk_score if e in profiles else 0.0)
        return df

    def monthly_summary(self, attendance_df):
        att = attendance_df.copy()
        att["date"] = pd.to_datetime(att["date"])
        return (att.groupby(["employee_id","month"])
            .agg(total_days=("status","count"),
                 absent_days=("status", lambda x:(x=="absent").sum()),
                 late_days=("status",   lambda x:(x=="late").sum()))
            .assign(attendance_rate=lambda d: 1-d["absent_days"]/d["total_days"])
            .round(3).reset_index())

    def department_attendance(self, attendance_df, workforce_df):
        profiles = self.compute_profiles(attendance_df)
        dept_map = dict(zip(workforce_df["employee_id"].astype(str),
                            workforce_df.get("department", pd.Series(["Unknown"]*len(workforce_df)))))
        rows = [{"employee_id":eid,"department":dept_map.get(eid,"Unknown"),
                 "attendance_rate":p.attendance_rate,"late_rate":p.late_rate,
                 "max_streak":p.max_streak,"risk_score":p.risk_score,"risk_flag":p.risk_flag}
                for eid,p in profiles.items()]
        df = pd.DataFrame(rows)
        if df.empty: return df
        return (df.groupby("department")
            .agg(headcount=("employee_id","count"),avg_attendance=("attendance_rate","mean"),
                 avg_late_rate=("late_rate","mean"),avg_risk_score=("risk_score","mean"),
                 critical_count=("risk_flag",lambda x:(x=="CRITICAL").sum()),
                 alert_count=("risk_flag",lambda x:(x=="ALERT").sum()))
            .round(3).reset_index().sort_values("avg_risk_score",ascending=False))

    def absenteeism_cost(self, profiles, avg_daily_cost=5000):
        return round(sum(p.absence_days for p in profiles.values()) * avg_daily_cost, 2)


@dataclass
class ChurnPrediction:
    employee_id: str; department: str; churn_prob: float
    risk_tier: str;   top_driver: str; est_cost_usd: float
    attendance_rate: float = 1.0; attendance_flag: str = "OK"; att_risk_score: float = 0.0

@dataclass
class MonthlyWorkforceForecast:
    month_iso: str; month_label: str; headcount: int; departures_est: int
    hires_needed: int; salary_cost: float; hiring_cost: float
    total_workforce_cost: float; headcount_delta: int

@dataclass
class HRSummary:
    current_headcount: int; high_risk_employees: int; attendance_critical: int
    projected_departures_6m: int; projected_hires_6m: int
    total_hiring_cost_6m: float; avg_monthly_salary: float
    churn_rate_pct: float; absenteeism_cost_annual: float


class ChurnModel:
    BASE_FEATURES = ["tenure_months","performance","workload_score","manager_score","pay_parity"]
    ATT_FEATURES  = ["att_rate","att_late_rate","att_max_streak","att_mon_fri","att_trend"]

    def __init__(self, random_state=42):
        self.rs=random_state; self._fitted=False; self._model=None; self._has_att=False

    def _has_att_cols(self, df): return all(c in df.columns for c in self.ATT_FEATURES)

    def _featurize(self, df):
        pm = {"below":2.0,"meets":0.0,"exceeds":-1.0}
        base = np.column_stack([
            df.get("tenure_months",  pd.Series([24]*len(df))).fillna(24).values,
            df.get("performance",    pd.Series(["meets"]*len(df))).map(pm).fillna(0).values,
            df.get("workload_score", pd.Series([5.0]*len(df))).fillna(5.0).values,
            df.get("manager_score",  pd.Series([7.0]*len(df))).fillna(7.0).values,
            df.get("pay_parity",     pd.Series([1.0]*len(df))).fillna(1.0).values,
        ])
        if self._has_att and self._has_att_cols(df):
            att = np.column_stack([
                1.0 - df["att_rate"].fillna(0.92).values,
                df["att_late_rate"].fillna(0.05).values,
                np.minimum(df["att_max_streak"].fillna(0).values/5.0, 1.0),
                df["att_mon_fri"].fillna(0).values,
                (-df["att_trend"].fillna(0)).clip(0,0.5).values*2,
            ])
            return np.hstack([base,att]).astype(float)
        return base.astype(float)

    def fit(self, workforce):
        self._has_att = self._has_att_cols(workforce)
        try:
            import warnings; warnings.filterwarnings("ignore")
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            df=workforce.copy(); X=self._featurize(df); y=(df["churn_prob"]>0.55).astype(int)
            if y.sum()<3: return self
            self._model=Pipeline([("scaler",StandardScaler()),
                ("clf",LogisticRegression(C=1.0,class_weight="balanced",max_iter=500,random_state=self.rs))])
            self._model.fit(X,y); self._fitted=True
        except ImportError: pass
        return self

    def predict(self, workforce, avg_salary=95_000, attendance_profiles=None):
        df=workforce.copy()
        if self._fitted and self._model:
            X=self._featurize(df); df["pred_churn_prob"]=self._model.predict_proba(X)[:,1]
        elif "churn_prob" in df.columns:
            df["pred_churn_prob"]=df["churn_prob"]
        else:
            pm={"below":0.65,"meets":0.35,"exceeds":0.12}
            df["pred_churn_prob"]=(df.get("performance",pd.Series(["meets"]*len(df))).map(pm).fillna(0.35)+df.get("workload_score",pd.Series([5.0]*len(df))).fillna(5.0)*0.025).clip(0.05,0.95)
            if "att_risk_score" in df.columns:
                df["pred_churn_prob"]=(df["pred_churn_prob"]+df["att_risk_score"]*0.20).clip(0.05,0.97)
        results=[]
        for _,row in df.iterrows():
            prob=float(row["pred_churn_prob"]); tier="HIGH" if prob>0.55 else "MEDIUM" if prob>0.30 else "LOW"
            att_rate=float(row.get("att_rate",0.92)); att_risk=float(row.get("att_risk_score",0.0))
            a_flag="CRITICAL" if att_risk>0.65 else "ALERT" if att_risk>0.40 else "WATCH" if att_risk>0.20 else "OK"
            drivers={"Low performance":PERF_TO_RISK.get(str(row.get("performance","meets")),0.40),
                     "High workload":float(row.get("workload_score",5.0))/10,
                     "Below market pay":max(0,1.0-float(row.get("pay_parity",1.0)))*2,
                     "New hire (<6mo)":1.0 if float(row.get("tenure_months",24))<6 else 0.0,
                     "Low mgr score":max(0,(5.0-float(row.get("manager_score",7.0)))/5),
                     "Poor attendance":att_risk}
            results.append(ChurnPrediction(
                employee_id=str(row.get("employee_id","?")),department=str(row.get("department","Unknown")),
                churn_prob=round(prob,3),risk_tier=tier,top_driver=max(drivers,key=drivers.get),
                est_cost_usd=round(avg_salary*HIRING_COST_RATIO+avg_salary*0.50,0),
                attendance_rate=round(att_rate,3),attendance_flag=a_flag,att_risk_score=round(att_risk,3)))
        return results

    def dept_risk_summary(self, predictions):
        rows=[{"department":p.department,"tier":p.risk_tier,"prob":p.churn_prob,
               "att_rate":p.attendance_rate,"att_flag":p.attendance_flag} for p in predictions]
        df=pd.DataFrame(rows)
        if df.empty: return df
        return (df.groupby("department")
            .agg(headcount=("prob","count"),avg_churn_prob=("prob","mean"),
                 high_risk=("tier",lambda x:(x=="HIGH").sum()),
                 avg_attendance=("att_rate","mean"),
                 att_critical=("att_flag",lambda x:(x.isin(["CRITICAL","ALERT"])).sum()))
            .round(3).reset_index().sort_values("avg_churn_prob",ascending=False))


class HiringForecast:
    def __init__(self,current_headcount,avg_annual_salary=95_000,hiring_cost_ratio=HIRING_COST_RATIO,revenue_per_head=REVENUE_PER_HEAD,horizon=6):
        self.current_hc=current_headcount;self.avg_salary=avg_annual_salary
        self.hiring_ratio=hiring_cost_ratio;self.rev_per_head=revenue_per_head;self.horizon=horizon

    def project(self,revenue_forecast,churn_predictions,risk_adj=0.0):
        high=sum(1 for p in churn_predictions if p.risk_tier=="HIGH")
        med =sum(1 for p in churn_predictions if p.risk_tier=="MEDIUM")
        att_crit=sum(1 for p in churn_predictions if p.attendance_flag in("CRITICAL","ALERT"))
        base_dep=round((high*0.20+med*0.05+att_crit*0.10)*(1+risk_adj),1)
        hc=self.current_hc;results=[]
        for rev_fc in revenue_forecast:
            target=max(self.current_hc,int(rev_fc.blended_revenue*12/self.rev_per_head))
            dep=max(0,round(base_dep));growth=max(0,target-hc);total=min(dep+growth,8)
            hc=max(hc-dep+total,1)
            results.append(MonthlyWorkforceForecast(
                month_iso=rev_fc.month_iso,month_label=rev_fc.month_label,headcount=hc,
                departures_est=dep,hires_needed=total,salary_cost=round(hc*(self.avg_salary/12),2),
                hiring_cost=round(total*self.avg_salary*self.hiring_ratio,2),
                total_workforce_cost=round(hc*(self.avg_salary/12)+total*self.avg_salary*self.hiring_ratio,2),
                headcount_delta=total-dep))
        return results

    def salary_adj_factors(self,wf_forecast,current_salary_expense):
        return [round((m.salary_cost-current_salary_expense)/max(current_salary_expense,1),4) for m in wf_forecast]

    def summarise(self,wf_forecast,churn_predictions):
        att_crit=sum(1 for p in churn_predictions if p.attendance_flag in("CRITICAL","ALERT"))
        avg_daily=self.avg_salary/260
        total_absent=sum(int((1-p.attendance_rate)*130) for p in churn_predictions)
        return HRSummary(
            current_headcount=self.current_hc,
            high_risk_employees=sum(1 for p in churn_predictions if p.risk_tier=="HIGH"),
            attendance_critical=att_crit,
            projected_departures_6m=int(sum(m.departures_est for m in wf_forecast)),
            projected_hires_6m=int(sum(m.hires_needed for m in wf_forecast)),
            total_hiring_cost_6m=round(sum(m.hiring_cost for m in wf_forecast),2),
            avg_monthly_salary=round(float(np.mean([m.salary_cost for m in wf_forecast])),2) if wf_forecast else 0,
            churn_rate_pct=round(sum(m.departures_est for m in wf_forecast)/(self.current_hc*6)*100,1),
            absenteeism_cost_annual=round(total_absent*avg_daily*2,2))

    def generate_insights(self,hr_summary,churn_preds,dept_risk):
        ins=[]
        if hr_summary.high_risk_employees>0:
            top=dept_risk.iloc[0]["department"] if not dept_risk.empty else "unknown"
            cost=hr_summary.high_risk_employees*(self.avg_salary*(HIRING_COST_RATIO+0.50))
            ins.append({"severity":"WARNING","category":"HR",
                "headline":f"{hr_summary.high_risk_employees} high-churn-risk employees — ${cost:,.0f} replacement exposure",
                "detail":f"Highest concentration: {top}. Churn: {hr_summary.churn_rate_pct:.1f}%.",
                "action":"Run stay interviews. Review compensation parity."})
        if hr_summary.attendance_critical>0:
            ins.append({"severity":"WARNING","category":"HR / Attendance",
                "headline":f"{hr_summary.attendance_critical} employees flagged CRITICAL/ALERT attendance",
                "detail":f"Absenteeism cost: ${hr_summary.absenteeism_cost_annual:,.0f}/year. Poor attendance precedes departure by 60-90 days.",
                "action":"Schedule 1-on-1s immediately. Check for unreported burnout."})
        if hr_summary.projected_hires_6m>0:
            ins.append({"severity":"INFO","category":"HR",
                "headline":f"{hr_summary.projected_hires_6m} hires needed in next 6 months",
                "detail":f"Hiring cost: ${hr_summary.total_hiring_cost_6m:,.0f}.",
                "action":"Open requisitions now — time-to-hire averages 6-8 weeks."})
        if hr_summary.churn_rate_pct>20:
            ins.append({"severity":"CRITICAL","category":"HR",
                "headline":f"Churn rate {hr_summary.churn_rate_pct:.0f}% exceeds 15% healthy threshold",
                "detail":"High turnover compounds team performance and morale.",
                "action":"Launch retention programme. Spot bonuses for high-risk, high-value staff."})
        return ins
