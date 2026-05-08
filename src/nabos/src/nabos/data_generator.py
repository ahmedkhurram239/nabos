import numpy as np, pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

np.random.seed(42)
COMPANY = {"name":"AcmeCorp","starting_cash":500000,"starting_heads":28,"avg_salary":95000}
SEASONAL = np.array([0.87,0.90,0.95,0.99,1.02,1.05,1.08,1.07,1.03,1.01,0.98,1.18])

def generate_financial_history(months=24):
    dates = pd.date_range("2023-01-01", periods=months, freq="MS")
    rows, headcount = [], COMPANY["starting_heads"]
    base_rev = 145000
    for i, d in enumerate(dates):
        trend = base_rev * (1.062 ** i)
        revenue = trend * SEASONAL[d.month-1] * np.random.normal(1.0, 0.04)
        headcount += max(0, int(np.random.poisson(0.7)))
        salaries = headcount * (COMPANY["avg_salary"]/12) * np.random.normal(1.0, 0.01)
        cogs = revenue * np.random.uniform(0.17, 0.22)
        marketing = (14000*(1.04**i)) * np.random.normal(1.0, 0.12)
        rd = (18000*(1.03**i)) * np.random.normal(1.0, 0.07)
        infrastructure = (7000*(1.07**i)) * np.random.normal(1.0, 0.05)
        ga = (9500*(1.015**i)) * np.random.normal(1.0, 0.03)
        total_exp = salaries+cogs+marketing+rd+infrastructure+ga
        rows.append({"ds":d,"revenue":round(revenue,2),"headcount":headcount,
            "salaries":round(salaries,2),"cogs":round(cogs,2),"marketing":round(marketing,2),
            "rd":round(rd,2),"infrastructure":round(infrastructure,2),"ga":round(ga,2),
            "total_expenses":round(total_exp,2),"net_cash_flow":round(revenue-total_exp,2)})
    df = pd.DataFrame(rows)
    df["cumulative_cash"] = df["net_cash_flow"].cumsum() + COMPANY["starting_cash"]
    return df

STAGES = ["Lead","Qualified","Proposal","Negotiation","Closed Won"]
STAGE_PROB = {"Lead":0.10,"Qualified":0.28,"Proposal":0.50,"Negotiation":0.74,"Closed Won":1.0}
STAGE_DAYS = {"Lead":85,"Qualified":60,"Proposal":38,"Negotiation":16,"Closed Won":0}
SIZES = {"SMB":1.0,"Mid-Market":3.5,"Enterprise":11.0}
REPS = [f"Rep-{i:02d}" for i in range(1,9)]

def generate_pipeline(n=55, ref_date="2025-01-01"):
    ref = pd.Timestamp(ref_date)
    rows = []
    for deal_id in range(1000, 1000+n):
        stage = np.random.choice(STAGES[:-1], p=[0.22,0.28,0.27,0.23])
        size = np.random.choice(list(SIZES), p=[0.45,0.38,0.17])
        value = round(np.random.lognormal(10.7,0.75)*SIZES[size], -2)
        prob = float(np.clip(np.random.normal(STAGE_PROB[stage],0.09),0.04,0.97))
        days = max(5, STAGE_DAYS[stage]+np.random.randint(-18,25))
        close = (ref+timedelta(days=days)).strftime("%Y-%m-%d")
        rows.append({"deal_id":f"D{deal_id}","company":f"Co-{deal_id}",
            "vertical":np.random.choice(["FinTech","SaaS","HealthTech","E-Commerce"]),
            "size":size,"deal_value":value,"stage":stage,"probability":round(prob,3),
            "expected_close":close,"days_in_stage":np.random.randint(3,50),
            "rep":np.random.choice(REPS),"has_champion":np.random.choice([0,1],p=[0.35,0.65]),
            "has_competitor":np.random.choice([0,1],p=[0.55,0.45]),
            "n_touchpoints":np.random.randint(4,25),"weighted_value":round(value*prob,2)})
    return pd.DataFrame(rows)

def generate_deal_history(n=320):
    rows = []
    for i in range(n):
        stage = np.random.choice(STAGES[:-1], p=[0.28,0.30,0.24,0.18])
        size = np.random.choice(list(SIZES), p=[0.45,0.38,0.17])
        value = round(np.random.lognormal(10.7,0.75)*SIZES[size], -2)
        has_champion = np.random.choice([0,1],p=[0.35,0.65])
        has_competitor = np.random.choice([0,1],p=[0.55,0.45])
        n_touch = np.random.randint(4,30)
        days = np.random.randint(3,70)
        base_p = STAGE_PROB[stage]
        if has_champion: base_p *= 1.28
        if has_competitor: base_p *= 0.80
        if n_touch > 15: base_p *= 1.12
        base_p = float(np.clip(base_p,0.05,0.95))
        outcome = int(np.random.random() < base_p)
        close_date = datetime(2024,1,1)+timedelta(days=np.random.randint(0,360))
        rows.append({"deal_value":value,"stage":stage,"size":size,
            "has_champion":has_champion,"has_competitor":has_competitor,
            "n_touchpoints":n_touch,"days_in_stage":days,
            "rep_experience":round(np.random.uniform(0.5,8.0),1),
            "won":outcome,"close_date":close_date.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

def generate_workforce(n_employees=52):
    DEPARTMENTS = ["Engineering","Sales","Marketing","Customer Success","Finance","Product","G&A"]
    DEPT_W = [0.30,0.22,0.10,0.15,0.07,0.10,0.06]
    rows = []
    for emp_id in range(1, n_employees+1):
        dept = np.random.choice(DEPARTMENTS, p=DEPT_W)
        tenure = int(np.random.gamma(shape=3, scale=8))
        performance = np.random.choice(["exceeds","meets","below"], p=[0.30,0.55,0.15])
        workload = round(np.random.beta(3,2)*10, 1)
        mgr_score = round(np.random.normal(7.0,1.5), 1)
        pay_parity = round(np.random.normal(1.0,0.12), 2)
        base_salary = int(np.random.normal(95000,15000))
        churn_logit = (-1.5
            + (1.8 if performance=="below" else -0.5 if performance=="exceeds" else 0.0)
            + 0.12*workload + (1.2 if tenure < 6 else 0.0)
            + (-0.4*(mgr_score-5)/5)
            + (1.5 if pay_parity < 0.85 else -0.5 if pay_parity > 1.10 else 0.0))
        churn_prob = float(np.clip(1/(1+np.exp(-churn_logit))+np.random.normal(0,0.03),0.02,0.97))
        rows.append({"employee_id":f"EMP-{emp_id:04d}","department":dept,
            "tenure_months":tenure,"performance":performance,"workload_score":workload,
            "manager_score":round(np.clip(mgr_score,1,10),1),"pay_parity":pay_parity,
            "base_salary":base_salary,"churn_prob":round(churn_prob,3),
            "is_high_risk":int(churn_prob>0.55)})
    return pd.DataFrame(rows)
