import math, numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASELINE={"inflation":0.025,"interest":0.045,"demand":0.05,"volatility":18.0,"geo_risk":2.0,"competition":2.0}
SUPPLY_LEVELS={"none":0.00,"low":0.05,"medium":0.13,"high":0.25}

@dataclass
class RiskProfile:
    inflation_rate:float=0.025; interest_rate:float=0.045; demand_growth:float=0.05
    supply_disruption:str="none"; market_volatility:float=18.0
    geo_risk:float=2.0; competition:float=2.0; label:str="Custom"; color:str="#6b7280"
    def supply_level(self): return SUPPLY_LEVELS.get(self.supply_disruption.lower(),0.0)

@dataclass
class RiskAdjustment:
    rev_delta:float; exp_delta:float; conv_delta:float; hr_adj:float
    attr:Dict[str,float]=field(default_factory=dict)

@dataclass
class RiskInsight:
    severity:str; driver:str; headline:str; detail:str; action:str
    impact_usd:Optional[float]=None

def low_risk_profile():
    return RiskProfile(inflation_rate=0.020,interest_rate=0.040,demand_growth=0.08,
        supply_disruption="none",market_volatility=13.0,geo_risk=1.5,competition=2.0,
        label="Low Risk",color="#10b981")

def elevated_risk_profile():
    return RiskProfile(inflation_rate=0.045,interest_rate=0.055,demand_growth=0.02,
        supply_disruption="medium",market_volatility=28.0,geo_risk=4.5,competition=5.0,
        label="Elevated Risk",color="#f59e0b")

def high_risk_profile():
    return RiskProfile(inflation_rate=0.075,interest_rate=0.070,demand_growth=-0.08,
        supply_disruption="high",market_volatility=42.0,geo_risk=7.5,competition=7.5,
        label="High Risk",color="#ef4444")

def crisis_profile():
    return RiskProfile(inflation_rate=0.120,interest_rate=0.090,demand_growth=-0.20,
        supply_disruption="high",market_volatility=65.0,geo_risk=9.0,competition=8.5,
        label="Crisis",color="#7f1d1d")

class RiskEngine:
    COEF={"inflation_to_exp":0.60,"interest_to_exp":0.12,"supply_to_exp":1.00,
          "vol_to_exp":0.008,"demand_to_rev":0.80,"geo_to_rev":0.025,"vol_to_rev":0.006,
          "interest_to_conv":0.30,"supply_to_conv":0.075,"comp_to_conv":0.018,"risk_to_attrition":0.08}
    MAX={"exp":0.45,"rev":0.50,"conv":0.40,"hr":0.50}

    def compute(self, profile):
        c=self.COEF
        infl_x=max(profile.inflation_rate-BASELINE["inflation"],0)
        rate_x=max(profile.interest_rate-BASELINE["interest"],0)
        vol_x=max(profile.market_volatility-BASELINE["volatility"],0)
        geo_x=max(profile.geo_risk-BASELINE["geo_risk"],0)
        comp_x=max(profile.competition-BASELINE["competition"],0)
        dem_x=profile.demand_growth-BASELINE["demand"]; sup=profile.supply_level()
        exp_d=min(infl_x*c["inflation_to_exp"]+rate_x*c["interest_to_exp"]+
                  sup*c["supply_to_exp"]+vol_x*c["vol_to_exp"],self.MAX["exp"])
        rev_d=max(dem_x*c["demand_to_rev"]-geo_x*c["geo_to_rev"]-vol_x*c["vol_to_rev"],-self.MAX["rev"])
        conv_d=max(-rate_x*c["interest_to_conv"]-sup*c["supply_to_conv"]-comp_x*c["comp_to_conv"],-self.MAX["conv"])
        stress=(abs(rev_d)+exp_d+abs(conv_d))/3
        hr_adj=min(stress*c["risk_to_attrition"]/0.10,self.MAX["hr"])
        attr={"Inflation":round(infl_x*c["inflation_to_exp"]*100,2),
              "Demand":round(dem_x*c["demand_to_rev"]*100,2),
              "Supply":round(sup*c["supply_to_exp"]*100,2)}
        return RiskAdjustment(rev_delta=round(rev_d,4),exp_delta=round(exp_d,4),
                              conv_delta=round(conv_d,4),hr_adj=round(hr_adj,4),attr=attr)

    def score(self, profile):
        adj=self.compute(profile)
        wtd=abs(adj.rev_delta)*0.45+adj.exp_delta*0.35+abs(adj.conv_delta)*0.20
        score=min(max(wtd/0.50*100,0),100)
        grade=("A — Low" if score<15 else "B — Manageable" if score<35 else
               "C — Elevated" if score<55 else "D — High" if score<75 else "F — Critical")
        return round(score,1), grade

    def generate_insights(self, profile, adj, base_rev_6m, base_exp_6m, pipe_weighted=0):
        insights=[]
        if profile.inflation_rate>BASELINE["inflation"]+0.01:
            cost=base_exp_6m*adj.attr.get("Inflation",0)/100
            insights.append(RiskInsight(
                "CRITICAL" if profile.inflation_rate>0.06 else "WARNING","Inflation",
                f"Inflation {profile.inflation_rate:.1%} raises costs by ${cost:,.0f}",
                f"Estimated expense uplift: ${cost:,.0f} over 6 months.",
                "Pre-negotiate vendor contracts at current rates.",impact_usd=cost))
        if profile.demand_growth<BASELINE["demand"]-0.03:
            rev_hit=base_rev_6m*abs(adj.rev_delta)
            insights.append(RiskInsight(
                "CRITICAL" if adj.rev_delta<-0.12 else "WARNING","Demand",
                f"Demand at {profile.demand_growth:.1%} — revenue short ${rev_hit:,.0f}",
                f"Revenue miss: {abs(adj.rev_delta):.1%} below base forecast.",
                f"Add {math.ceil(rev_hit/58000)} qualified opportunities.",impact_usd=-rev_hit))
        return sorted(insights,key=lambda i:{"CRITICAL":0,"WARNING":1,"INFO":2}.get(i.severity,3))

    def sensitivity_table(self, base_profile, base_rev, base_exp):
        rows=[]
        for label,overrides in [
            ("Inflation +1pp",{"inflation_rate":base_profile.inflation_rate+0.01}),
            ("Interest +0.5pp",{"interest_rate":base_profile.interest_rate+0.005}),
            ("Demand -5%",{"demand_growth":base_profile.demand_growth-0.05}),
            ("Supply → high",{"supply_disruption":"high"}),
            ("Volatility +10",{"market_volatility":base_profile.market_volatility+10})]:
            p=RiskProfile(**{**base_profile.__dict__,**overrides})
            adj=self.compute(p)
            rows.append({"Perturbation":label,
                         "Rev Δ":f"${base_rev*adj.rev_delta:+,.0f}",
                         "Exp Δ":f"${base_exp*adj.exp_delta:+,.0f}",
                         "Conv Δ":f"{adj.conv_delta:+.1%}"})
        return rows
