
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from datetime import timedelta


@dataclass
class DailyPoint:
    date: str; day_label: str; revenue: float; expenses: float
    net: float; balance: float; is_weekend: bool; alert: str = "OK"

@dataclass
class WeeklyPoint:
    week_start: str; week_label: str; revenue: float; expenses: float
    net: float; balance: float; lower: float; upper: float; alert: str = "OK"

@dataclass
class YearlyPoint:
    year: int; year_label: str; revenue: float; expenses: float
    net: float; revenue_growth: float; headcount_est: int; lower: float; upper: float


class TimeSeriesEngine:

    def __init__(self, history, starting_cash=0):
        self.hist = history.copy()
        self.starting_cash = starting_cash
        self._fit()

    def _fit(self):
        rev = self.hist["revenue"].values.astype(float)
        exp = self.hist["total_expenses"].values.astype(float)
        n   = len(rev); x = np.arange(n, dtype=float)
        self._rev_slope, self._rev_int = np.polyfit(x, rev, 1)
        self._exp_slope, self._exp_int = np.polyfit(x, exp, 1)
        self._n = n
        rev_trend = self._rev_int + self._rev_slope * x
        ratios = rev / np.maximum(rev_trend, 1.0)
        self._seasonal = np.clip(np.array([np.mean(ratios[i::12]) if len(ratios[i::12])>0 else 1.0 for i in range(12)]), 0.7, 1.4)
        self._rev_std = float(np.std(rev - rev_trend))
        self._last_date = pd.Timestamp(self.hist["ds"].iloc[-1])
        self._last_cash = float(self.hist.get("cumulative_cash", pd.Series([self.starting_cash])).iloc[-1])

    def _monthly_rev(self, step, month):
        return max((self._rev_int + self._rev_slope*(self._n+step)) * float(self._seasonal[(month-1)%12]), 0.0)

    def _monthly_exp(self, step):
        return max(self._exp_int + self._exp_slope*(self._n+step), 0.0)

    def daily_forecast(self, days=30):
        np.random.seed(42)
        results = []; balance = self._last_cash
        ref = self._last_date + timedelta(days=1)
        for i in range(days):
            d = ref + timedelta(days=i)
            is_wkd = d.weekday() >= 5
            if is_wkd:
                rev = 0.0; exp = self._monthly_exp(0) / 30.0
            else:
                rev = self._monthly_rev(0, d.month) / 22.0 * np.random.normal(1.0, 0.02)
                exp = self._monthly_exp(0) / 22.0 * np.random.normal(1.0, 0.01)
            net = rev - exp; balance += net
            alert = "CRITICAL" if balance<0 else "LOW" if balance<self._last_cash*0.05 else "OK"
            results.append(DailyPoint(date=d.strftime("%Y-%m-%d"), day_label=d.strftime("%a %d %b"),
                revenue=round(rev,2), expenses=round(exp,2), net=round(net,2),
                balance=round(balance,2), is_weekend=is_wkd, alert=alert))
        return results

    def weekly_forecast(self, weeks=12):
        results = []; balance = self._last_cash
        ref = self._last_date + timedelta(days=1)
        while ref.weekday() != 0: ref += timedelta(days=1)
        for w in range(weeks):
            wk = ref + timedelta(weeks=w)
            rev = self._monthly_rev(int(w/4.3), wk.month) / 4.3
            exp = self._monthly_exp(int(w/4.3)) / 4.3
            net = rev - exp; balance += net
            ci  = 1.645 * self._rev_std / 4.3 * np.sqrt(w+1) * 0.4
            alert = "CRITICAL" if balance<0 else "OK"
            results.append(WeeklyPoint(
                week_start=wk.strftime("%Y-%m-%d"),
                week_label=f"W{wk.isocalendar()[1]} {wk.strftime('%b %d')}",
                revenue=round(rev,2), expenses=round(exp,2), net=round(net,2),
                balance=round(balance,2), lower=round(max(net-ci,net*0.75),2),
                upper=round(net+ci,2), alert=alert))
        return results

    def yearly_forecast(self, years=3, current_headcount=50, revenue_per_head=180_000):
        results = []; ref_year = self._last_date.year + 1
        annual_rev_now = sum(self._monthly_rev(m, m+1) for m in range(12))
        for y in range(years):
            step = (y+1)*12
            annual_rev = sum(self._monthly_rev(step+m, m+1) for m in range(12))
            annual_exp = sum(self._monthly_exp(step+m) for m in range(12))
            net = annual_rev - annual_exp
            growth = (annual_rev/max(annual_rev_now if y==0 else results[-1].revenue,1)-1)
            hc = max(int(annual_rev/revenue_per_head), current_headcount)
            ci_pct = 0.10 + y*0.08
            results.append(YearlyPoint(
                year=ref_year+y, year_label=str(ref_year+y),
                revenue=round(annual_rev,2), expenses=round(annual_exp,2), net=round(net,2),
                revenue_growth=round(growth,4), headcount_est=hc,
                lower=round(net*(1-ci_pct),2), upper=round(net*(1+ci_pct),2)))
            annual_rev_now = annual_rev
        return results

    def horizon_summary(self, daily, weekly, monthly, yearly):
        return {
            "daily_net_30d":      round(sum(d.net for d in daily),2),
            "weekly_net_12w":     round(sum(w.net for w in weekly),2),
            "monthly_net_6m":     round(sum(m.net for m in monthly),2),
            "yearly_net_3yr":     round(sum(y.net for y in yearly),2),
            "daily_min_balance":  round(min(d.balance for d in daily),2),
            "weekly_min_balance": round(min(w.balance for w in weekly),2),
            "yearly_rev_y1":      yearly[0].revenue if yearly else 0,
            "yearly_rev_y3":      yearly[-1].revenue if yearly else 0,
            "yearly_growth_y1":   yearly[0].revenue_growth if yearly else 0,
            "headcount_y3":       yearly[-1].headcount_est if yearly else 0,
        }
