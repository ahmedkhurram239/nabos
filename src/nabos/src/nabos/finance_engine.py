import numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dateutil.relativedelta import relativedelta

EXPENSE_TYPE = {"salaries":"fixed","cogs":"variable","marketing":"variable",
                "rd":"semi-fixed","infrastructure":"semi-fixed","ga":"fixed"}

@dataclass
class MonthlyExpenseForecast:
    month_iso:str; month_label:str; categories:Dict[str,float]
    total:float; fixed:float; variable:float; lower_90:float; upper_90:float

@dataclass
class MonthlyCashFlow:
    month_iso:str; month_label:str; revenue:float; expenses:float
    net:float; balance:float; net_lower:float; net_upper:float
    balance_lower:float; balance_upper:float; alert:str="OK"

@dataclass
class FinanceSummary:
    total_revenue_6m:float; total_expenses_6m:float; total_net_6m:float
    ending_balance:float; min_balance:float; deficit_months:int
    months_positive:int; avg_burn_ratio:float; peak_revenue_month:str

class ExpenseEngine:
    def __init__(self, horizon=6):
        self.horizon=horizon; self._models={}; self._cats=[]

    def fit(self, history):
        self._cats=[c for c in EXPENSE_TYPE if c in history.columns]
        for cat in self._cats:
            vals=history[cat].values.astype(float); n=len(vals)
            x=np.arange(n,dtype=float)
            slope,intercept=np.polyfit(x,vals,1)
            last_n=min(24,n)
            trend_last=intercept+slope*np.arange(n-last_n,n)
            ratios=vals[-last_n:]/np.maximum(trend_last,1.0)
            self._models[cat]={"slope":slope,"intercept":intercept,"n":n,
                               "seasonal":ratios,"std":float(np.std(vals-(intercept+slope*x)))}
        return self

    def forecast(self, history, headcount_adj=None):
        ref=pd.Timestamp(history["ds"].iloc[-1])+relativedelta(months=1)
        months=[ref+relativedelta(months=i) for i in range(self.horizon)]
        results=[]
        for step,m in enumerate(months):
            cats_out={}
            for cat in self._cats:
                md=self._models[cat]
                trend=md["intercept"]+md["slope"]*(md["n"]+step)
                s_idx=step%len(md["seasonal"])
                season=float(np.clip(md["seasonal"][s_idx] if len(md["seasonal"])>s_idx else 1.0,0.6,1.5))
                val=trend*season
                if cat=="salaries" and headcount_adj:
                    adj_val=headcount_adj[step] if step<len(headcount_adj) else 0.0
                    val=val*(1+adj_val)
                cats_out[cat]=round(max(val,0.0),2)
            total=sum(cats_out.values())
            fixed=sum(v for k,v in cats_out.items() if EXPENSE_TYPE.get(k)=="fixed")
            var=sum(v for k,v in cats_out.items() if EXPENSE_TYPE.get(k)=="variable")
            pool_std=np.mean([md["std"] for md in self._models.values()])
            ci=1.645*pool_std*np.sqrt(step+1)*0.45
            results.append(MonthlyExpenseForecast(
                month_iso=m.strftime("%Y-%m"),month_label=m.strftime("%B %Y"),
                categories=cats_out,total=round(total,2),fixed=round(fixed,2),
                variable=round(var,2),lower_90=round(max(total-ci,total*0.80),2),
                upper_90=round(total+ci,2)))
        return results

    def scenario_shift(self, history, exp_delta, headcount_adj=None):
        base=self.forecast(history,headcount_adj)
        for m in base:
            for cat in m.categories:
                if EXPENSE_TYPE.get(cat) in ("variable","semi-fixed"):
                    m.categories[cat]=round(m.categories[cat]*(1+exp_delta),2)
            m.total=round(sum(m.categories.values()),2)
            m.lower_90=round(m.total*0.88,2); m.upper_90=round(m.total*1.12,2)
        return base

class CashFlowEngine:
    def __init__(self, starting_cash=500000):
        self.starting_cash=starting_cash

    def integrate(self, revenue_fc, expense_fc):
        balance=self.starting_cash; bal_lo=self.starting_cash; bal_hi=self.starting_cash
        monthly=[]
        for rev,exp in zip(revenue_fc,expense_fc):
            net=float(rev.blended_revenue)-exp.total
            net_lo=float(rev.lower_90)-exp.upper_90
            net_hi=float(rev.upper_90)-exp.lower_90
            balance+=net; bal_lo+=net_lo; bal_hi+=net_hi
            if balance<=0: alert="CRITICAL"
            elif balance<=75000: alert="DEFICIT"
            elif net>=150000: alert="SURPLUS"
            else: alert="OK"
            monthly.append(MonthlyCashFlow(
                month_iso=rev.month_iso,month_label=rev.month_label,
                revenue=round(float(rev.blended_revenue),2),expenses=round(exp.total,2),
                net=round(net,2),balance=round(balance,2),
                net_lower=round(net_lo,2),net_upper=round(net_hi,2),
                balance_lower=round(bal_lo,2),balance_upper=round(bal_hi,2),alert=alert))
        return monthly

    def summarise(self, cf):
        revenues=[m.revenue for m in cf]; expenses=[m.expenses for m in cf]
        nets=[m.net for m in cf]; balances=[m.balance for m in cf]
        burn=[e/max(r,1) for r,e in zip(revenues,expenses)]
        peak_idx=int(np.argmax(revenues))
        return FinanceSummary(
            total_revenue_6m=round(sum(revenues),2),total_expenses_6m=round(sum(expenses),2),
            total_net_6m=round(sum(nets),2),ending_balance=round(balances[-1],2),
            min_balance=round(min(balances),2),deficit_months=sum(1 for b in balances if b<=0),
            months_positive=sum(1 for n in nets if n>0),
            avg_burn_ratio=round(float(np.mean(burn)),3),
            peak_revenue_month=cf[peak_idx].month_label)

    def generate_insights(self, cf, summary, pipeline_gap=0.0):
        insights=[]
        if summary.deficit_months>0:
            d=[m for m in cf if m.balance<=0]
            insights.append({"severity":"CRITICAL","category":"Cash",
                "headline":f"Cash deficit in {d[0].month_label}",
                "detail":f"Balance hits ${d[0].balance:,.0f}.",
                "action":"Reduce variable costs 15% or close 3+ deals immediately."})
        if summary.avg_burn_ratio>0.80:
            insights.append({"severity":"WARNING","category":"Expenses",
                "headline":f"Burn ratio {summary.avg_burn_ratio:.0%} above 80%",
                "detail":"SaaS benchmark is under 70%.",
                "action":"Audit marketing and infrastructure spend."})
        if pipeline_gap>0:
            insights.append({"severity":"INFO","category":"Revenue",
                "headline":f"Pipeline gap: ${pipeline_gap:,.0f} below target",
                "detail":"Pipeline coverage below 100% of 6M target.",
                "action":f"Add {int(pipeline_gap/58000)+1} mid-market deals."})
        if not insights:
            insights.append({"severity":"INFO","category":"Cash",
                "headline":"Cash position healthy — no deficit risk",
                "detail":f"Minimum balance ${summary.min_balance:,.0f}.",
                "action":"Consider deploying surplus into growth."})
        return insights
