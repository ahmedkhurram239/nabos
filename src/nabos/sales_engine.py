import numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional
from dateutil.relativedelta import relativedelta

STAGE_BASELINES={"Lead":0.10,"Qualified":0.28,"Proposal":0.50,"Negotiation":0.74,"Closed Won":1.0}
FEATURE_COLS=["stage_ord","log_value","has_champion","has_competitor","n_touchpoints","days_in_stage","size_smb","size_mm","size_ent"]

@dataclass
class MonthlyRevenueForecast:
    month_iso:str; month_label:str; pipeline_revenue:float; trend_revenue:float
    blended_revenue:float; lower_90:float; upper_90:float; deal_count:int; pipeline_weight:float

@dataclass
class MLMetrics:
    cv_auc:float; brier_score:float; n_train:int; win_rate:float
    feature_importance:Dict[str,float]; stage_win_rates:Dict[str,float]

class DealMLModel:
    def __init__(self, random_state=42):
        self.rs=random_state; self._fitted=False; self._pipe=None; self._metrics=None

    def _featurize(self, df):
        stage_map={s:i for i,s in enumerate(["Lead","Qualified","Proposal","Negotiation","Closed Won"])}
        X=pd.DataFrame()
        s_col=df.get("stage",df.get("entry_stage",pd.Series(["Qualified"]*len(df))))
        X["stage_ord"]=s_col.map(stage_map).fillna(1)
        X["log_value"]=np.log1p(pd.to_numeric(df.get("deal_value",50000),errors="coerce").fillna(50000))
        X["has_champion"]=pd.to_numeric(df.get("has_champion",0),errors="coerce").fillna(0)
        X["has_competitor"]=pd.to_numeric(df.get("has_competitor",0),errors="coerce").fillna(0)
        X["n_touchpoints"]=pd.to_numeric(df.get("n_touchpoints",8),errors="coerce").fillna(8)
        X["days_in_stage"]=pd.to_numeric(df.get("days_in_stage",14),errors="coerce").fillna(14)
        sz=df.get("size",df.get("company_size",pd.Series(["Mid-Market"]*len(df))))
        X["size_smb"]=(sz=="SMB").astype(int)
        X["size_mm"]=(sz=="Mid-Market").astype(int)
        X["size_ent"]=(sz=="Enterprise").astype(int)
        return X[FEATURE_COLS]

    def fit(self, history):
        try:
            import warnings; warnings.filterwarnings("ignore")
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.metrics import brier_score_loss
            df=history.copy()
            if "won" in df.columns and "outcome" not in df.columns:
                df["outcome"]=df["won"]
            X=self._featurize(df); y=df["outcome"].astype(int)
            base=LogisticRegression(C=0.5,class_weight="balanced",max_iter=500,random_state=self.rs)
            self._pipe=Pipeline([("scaler",StandardScaler()),
                                 ("model",CalibratedClassifierCV(base,cv=5,method="isotonic"))])
            self._pipe.fit(X,y); self._fitted=True
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=self.rs)
            auc=cross_val_score(self._pipe,X,y,cv=cv,scoring="roc_auc")
            probs=self._pipe.predict_proba(X)[:,1]
            stage_wr={s:round(float(df[df.get("stage",df.get("entry_stage",""))==s]["outcome"].mean()),3)
                      if len(df[df.get("stage","")==s])>=5 else STAGE_BASELINES[s] for s in STAGE_BASELINES}
            try:
                coefs=np.abs(self._pipe.named_steps["model"].calibrated_classifiers_[0].estimator.coef_[0])
                fi=dict(zip(FEATURE_COLS,coefs/coefs.sum()))
            except:
                fi={c:1/len(FEATURE_COLS) for c in FEATURE_COLS}
            self._metrics=MLMetrics(cv_auc=round(float(auc.mean()),4),
                brier_score=round(float(brier_score_loss(y,probs)),4),
                n_train=len(y),win_rate=round(float(y.mean()),3),
                feature_importance={k:round(float(v),4) for k,v in fi.items()},
                stage_win_rates=stage_wr)
        except ImportError:
            pass
        return self

    def score(self, pipeline):
        df=pipeline.copy()
        df["baseline_prob"]=df["stage"].map(STAGE_BASELINES).fillna(0.25)
        if self._fitted and self._pipe:
            X=self._featurize(df)
            df["ml_probability"]=np.clip(self._pipe.predict_proba(X)[:,1],0.02,0.97).round(3)
        else:
            df["ml_probability"]=df["baseline_prob"]
        n=df.get("n_touchpoints",pd.Series([8]*len(df))).clip(lower=5)
        p=df["ml_probability"]; z=1.645
        denom=1+z**2/n; centre=(p+z**2/(2*n))/denom
        margin=z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/denom
        df["prob_lower"]=(centre-margin).clip(0.01,0.95).round(3)
        df["prob_upper"]=(centre+margin).clip(0.05,0.99).round(3)
        df["blended_probability"]=df["ml_probability"]
        df["weighted_value"]=(df["deal_value"]*df["blended_probability"]).round(2)
        return df

    def metrics(self): return self._metrics

class RevenueEngine:
    PIPELINE_WEIGHT={0:0.72,1:0.62,2:0.48,3:0.33,4:0.20,5:0.12}
    def __init__(self, horizon=6):
        self.horizon=horizon; self._slope=0.0; self._intercept=0.0
        self._seasonal=np.ones(12); self._resid_std=0.0; self._hist_len=0

    def fit(self, history):
        rev=history["revenue"].values; n=len(rev); x=np.arange(n,dtype=float)
        self._slope,self._intercept=np.polyfit(x,rev,1); self._hist_len=n
        last_n=min(24,n); trend_last=self._intercept+self._slope*np.arange(n-last_n,n)
        ratios=rev[-last_n:]/np.maximum(trend_last,1.0)
        self._seasonal=np.array([np.mean(ratios[i::12]) for i in range(12)])
        self._resid_std=float(np.std(rev-(self._intercept+self._slope*x)))
        return self

    def _trend(self, step, m):
        return max((self._intercept+self._slope*(self._hist_len+step))*
                   float(np.clip(self._seasonal[(m.month-1)%12],0.7,1.4)),0.0)

    def forecast(self, history, pipeline, rev_delta=0.0, conv_delta=0.0):
        ref=pd.Timestamp(history["ds"].iloc[-1])+relativedelta(months=1)
        months=[ref+relativedelta(months=i) for i in range(self.horizon)]
        adj=pipeline.copy()
        adj["blended_probability"]=(adj["blended_probability"]+conv_delta).clip(0.01,0.99)
        adj["weighted_value"]=adj["deal_value"]*adj["blended_probability"]
        results=[]
        for step,m in enumerate(months):
            end=m+relativedelta(months=1)-timedelta(days=1)
            subs=adj[(pd.to_datetime(adj["expected_close"])>=m)&
                     (pd.to_datetime(adj["expected_close"])<=end)&
                     (adj["stage"]!="Closed Won")]
            pipe_rev=float((subs["deal_value"]*subs["blended_probability"]).sum()) if len(subs)>0 else 0.0
            pipe_best=float((subs["deal_value"]*subs["prob_upper"]).sum()) if len(subs)>0 else 0.0
            pipe_worst=float((subs["deal_value"]*subs["prob_lower"]).sum()) if len(subs)>0 else 0.0
            pw=self.PIPELINE_WEIGHT.get(step,0.12)
            tv=self._trend(step,m)*(1+rev_delta)
            blend=pipe_rev*pw+tv*(1-pw)
            ci=1.645*self._resid_std*np.sqrt(step+1)*0.5
            results.append(MonthlyRevenueForecast(
                month_iso=m.strftime("%Y-%m"),month_label=m.strftime("%B %Y"),
                pipeline_revenue=round(pipe_rev,2),trend_revenue=round(tv,2),
                blended_revenue=round(blend,2),lower_90=round(max(blend-ci,blend*0.75),2),
                upper_90=round(blend+ci,2),deal_count=len(subs),pipeline_weight=round(pw,2)))
        return results
