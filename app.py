import sys, warnings, os
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import streamlit as st

# Load API key from Streamlit secrets
try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    os.environ["ANTHROPIC_API_KEY"] = ""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from nabos.orchestrator import run_full_forecast
from nabos.hr_engine import AttendanceAnalyzer, generate_attendance
from nabos.ai_engine import AIEngine
from nabos.skill_engine import SkillMatchEngine
from nabos.timeseries_engine import TimeSeriesEngine
from nabos.analytics_engine import AnomalyDetector, EmployeeTrendTracker, DealVelocityTracker, BudgetVsActual

st.set_page_config(page_title="NABOS — Nestlé Pakistan", page_icon="🧠", layout="wide")

# Load API key from Streamlit secrets

@st.cache_resource(show_spinner=False)
def load_data():
    return run_full_forecast(
        financial_csv="data/nestle_financial.csv",
        pipeline_csv="data/nestle_pipeline.csv",
        deal_hist_csv="data/nestle_deals.csv",
        workforce_csv="data/nestle_workforce.csv",
        starting_cash=21268,
    )

@st.cache_resource(show_spinner=False)
def load_attendance(_wf):
    att=generate_attendance(_wf,months=6,seed=42)
    az=AttendanceAnalyzer()
    return att,az.enrich_workforce(_wf,att),az.department_attendance(att,_wf),az.monthly_summary(att),az.compute_profiles(att),az

@st.cache_resource(show_spinner=False)
def get_ai(_r):
    ai=AIEngine(_r,_r.company_name); ai.load_history(); return ai

@st.cache_resource(show_spinner=False)
def get_skill_engine(_wf):
    e=SkillMatchEngine(); return e,e.match_all(_wf)

@st.cache_resource(show_spinner=False)
def get_timeseries(_hist,_cash):
    return TimeSeriesEngine(_hist,starting_cash=_cash)

@st.cache_resource(show_spinner=False)
def get_analytics(_r, _pipe):
    anomalies = AnomalyDetector().detect(_r.history)
    dv_alerts = DealVelocityTracker().analyse(_pipe)
    variances = BudgetVsActual().compare(_r.history, forecast_months=3)
    return anomalies, dv_alerts, variances

def fmtM(v):
    a=abs(v);s="$" if v>=0 else "-$"
    if a>=1e9: return s+f"{a/1e9:.2f}B"
    if a>=1e6: return s+f"{a/1e6:.2f}M"
    if a>=1e3: return s+f"{a/1e3:.0f}K"
    return s+f"{a:.0f}"

def ins_html(i,clickable=False):
    sev=i.get("severity","INFO")
    bg={"CRITICAL":"#fef2f2","WARNING":"#fffbeb","INFO":"#eff6ff"}.get(sev,"#eff6ff")
    bd={"CRITICAL":"#ef4444","WARNING":"#f59e0b","INFO":"#3b82f6"}.get(sev,"#3b82f6")
    ic={"CRITICAL":"🔴","WARNING":"⚠️","INFO":"ℹ️"}.get(sev,"ℹ️")
    return (f'<div style="border-radius:8px;padding:11px 14px;margin-bottom:7px;border-left:4px solid {bd};background:{bg};font-size:13px">'
            f'<b>{ic} {i["headline"]}</b><br>'
            f'<span style="color:#555;font-size:12px">{i.get("detail","")}</span><br>'
            f'<span style="font-size:11px;font-style:italic;color:#1d4ed8">→ {i.get("action","")}</span></div>')

G="rgba(0,0,0,.05)"
SCC={"Base":"#6366f1","Low Risk":"#10b981","Elevated":"#f59e0b","High Risk":"#ef4444","Crisis":"#7f1d1d"}

def bl(fig,h=300):
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",height=h,
        margin=dict(l=10,r=10,t=30,b=10),font=dict(size=11,color="#555"),
        legend=dict(orientation="h",y=1.04,x=0,font=dict(size=10)),
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor=G))
    return fig

with st.spinner("Running NABOS AI pipeline — Nestlé Pakistan..."):
    r=load_data()
with st.spinner("Analysing attendance, skills and anomalies..."):
    att_df,enr_wf,dep_att,mon_att,att_pro,att_az=load_attendance(r.workforce)
    skill_engine,skill_results=get_skill_engine(enr_wf)
    anomalies,dv_alerts,variances=get_analytics(r,r.pipeline)

ai=get_ai(r)
ts_engine=get_timeseries(r.history,21268)
trend_tracker=EmployeeTrendTracker()
emp_trends=trend_tracker.compute_trends(enr_wf,r.churn_preds)
dv_summary=DealVelocityTracker().summary(dv_alerts)
bva=BudgetVsActual()
accuracy=bva.accuracy_score(variances)

fin=r.finance_summary;hr=r.hr_summary
cf=r.cashflow;rfc=r.revenue_fc;efc=r.expense_fc
sc=r.scenarios;hist=r.history
mo=[m.month_label.split()[0][:3] for m in cf]
fmo=[m.month_label for m in cf]
hd=[str(d)[:7] for d in hist["ds"]]
fx=[m.month_iso for m in cf]

with st.sidebar:
    st.markdown("## 🧠 NABOS")
    st.markdown("*Nestlé Pakistan*")
    st.divider()
    starting_cash=st.number_input("Starting cash ($)",0,50000000,21268,step=1000)
    st.divider()
    sim_rev=st.slider("Revenue adj %",-40,40,0)
    sim_exp=st.slider("Expense adj %",-20,40,0)
    sim_conv=st.slider("Conversion adj %",-30,30,0)
    st.divider()
    # Show alert badge counts
    crit_count=len([i for i in r.all_insights if i.get("severity")=="CRITICAL"])
    anom_count=len([a for a in anomalies if a.severity=="CRITICAL"])
    stall_count=dv_summary["critical"]
    if crit_count+anom_count+stall_count>0:
        st.error(f"🔴 {crit_count+anom_count+stall_count} critical issues")
    page=st.radio("",["🏠 Overview","💰 Revenue","📉 Expenses","💵 Cash Flow",
        "👥 HR & Churn","📋 Attendance","🔁 Skill Match","📅 Timeline",
        "📊 Analytics","🌐 Risk","🔀 Scenarios","🧠 AI Advisor","🎯 Decisions"],
        label_visibility="collapsed")

asc=sc.get("Base",r.base_scenario)
srv=[m.revenue*(1+sim_rev/100)*(1+sim_conv/200) for m in asc.cashflow]
sev_=[m.expenses*(1+sim_exp/100) for m in asc.cashflow]
snv=[rv-ex for rv,ex in zip(srv,sev_)]
b=starting_cash;sbv=[]
for n in snv: b+=n;sbv.append(round(b))

# ─── OVERVIEW ─────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("### Nestlé Pakistan — AI Business Operating System")
    ac=(enr_wf["att_risk_score"]>0.65).sum()
    tc=len(skill_engine.transfer_candidates(skill_results))
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("6M Revenue",fmtM(fin.total_revenue_6m),f"burn {fin.avg_burn_ratio:.0%}")
    c2.metric("6M Net",fmtM(fin.total_net_6m),f"{fin.months_positive}/6 positive")
    c3.metric("End Balance",fmtM(fin.ending_balance),f"min {fmtM(fin.min_balance)}")
    c4.metric("HR High Risk",hr.high_risk_employees,f"{hr.churn_rate_pct:.0f}% churn")
    c5.metric("Anomalies",len(anomalies),f"{len([a for a in anomalies if a.severity=='CRITICAL'])} critical")
    c6.metric("Stalled Deals",dv_summary["critical"],f"{fmtM(dv_summary['stalled_value'])} at risk")
    col_l,col_r=st.columns([3,1])
    with col_l:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hd,y=hist["revenue"].tolist(),name="Revenue (hist)",mode="lines",line=dict(color="rgba(99,102,241,.35)",width=1.5)))
        fig.add_trace(go.Scatter(x=fx,y=[m.blended_revenue for m in rfc],name="Revenue (fc)",mode="lines+markers",line=dict(color="#6366f1",width=2.5),marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=hd,y=hist["total_expenses"].tolist(),name="Expenses (hist)",mode="lines",line=dict(color="rgba(239,68,68,.3)",width=1.5)))
        fig.add_trace(go.Scatter(x=fx,y=[m.expenses for m in cf],name="Expenses (fc)",mode="lines",line=dict(color="#ef4444",width=2)))
        # Mark anomalies on chart
        anom_months=[a.date for a in anomalies if a.severity=="CRITICAL"]
        for am in anom_months[:3]:
            fig.add_vline(x=am,line_dash="dot",line_color="#ef4444",line_width=1)
        fig.add_vline(x=fx[0],line_dash="dash",line_color="rgba(0,0,0,.15)",line_width=1)
        bl(fig,300);fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col_r:
        st.markdown("##### Top alerts")
        for ins in r.all_insights[:3]: st.markdown(ins_html(ins),unsafe_allow_html=True)
        if anomalies:
            st.markdown(ins_html({"severity":anomalies[0].severity,"headline":anomalies[0].headline,
                "detail":f"Z-score: {anomalies[0].z_score:.1f}","action":anomalies[0].action}),unsafe_allow_html=True)
    col_a,col_b=st.columns(2)
    with col_a:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=hd,y=hist["cumulative_cash"].tolist(),name="Historical",mode="lines",line=dict(color="rgba(109,40,217,.35)",width=1.5),fill="tozeroy",fillcolor="rgba(109,40,217,.04)"))
        fig2.add_trace(go.Scatter(x=fx,y=[m.balance for m in cf],name="Forecast",mode="lines+markers",line=dict(color="#7c3aed",width=2.5),marker=dict(size=4)))
        fig2.add_hline(y=0,line_color="rgba(239,68,68,.4)",line_dash="dot")
        bl(fig2,250);fig2.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
    with col_b:
        tbl=pd.DataFrame({"Month":fmo,"Revenue":[fmtM(m.revenue) for m in cf],
            "Expenses":[fmtM(m.expenses) for m in cf],"Net":[fmtM(m.net) for m in cf],
            "Balance":[fmtM(m.balance) for m in cf],"Status":[m.alert for m in cf]})
        st.dataframe(tbl,use_container_width=True,hide_index=True,height=250)

# ─── REVENUE ──────────────────────────────────────────────────────────
elif "Revenue" in page:
    st.markdown("### Revenue & Sales")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Pipeline",fmtM(r.pipeline["weighted_value"].sum()),f"{len(r.pipeline)} deals")
    c2.metric("6M Forecast",fmtM(fin.total_revenue_6m))
    c3.metric("Stalled Deals",dv_summary["critical"],f"{fmtM(dv_summary['stalled_value'])} at risk")
    if r.ml_metrics: c4.metric("ML AUC",f"{r.ml_metrics.cv_auc:.3f}")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fx,y=[m.upper_90 for m in rfc],fill=None,mode="lines",line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=fx,y=[m.lower_90 for m in rfc],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(99,102,241,.10)",name="90% CI"))
    fig.add_trace(go.Scatter(x=fx,y=[m.blended_revenue for m in rfc],name="Forecast",mode="lines+markers",line=dict(color="#6366f1",width=2.5),marker=dict(size=6)))
    bl(fig,280);fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    # Deal velocity table
    st.markdown("##### Deal Velocity — Stalled Deals")
    stalled=[a for a in dv_alerts if a.severity in ("CRITICAL","WARNING")]
    if stalled:
        dv_df=pd.DataFrame([{"Deal":a.deal_id,"Company":a.company,"Stage":a.stage,
            "Days in Stage":a.days_in_stage,"Benchmark":a.benchmark_avg,"Overdue By":a.overdue_by,
            "Value":fmtM(a.deal_value),"ML Prob":f"{a.ml_probability:.0%}","Status":a.severity,"Action":a.action[:60]+"..."} for a in stalled])
        st.dataframe(dv_df,use_container_width=True,hide_index=True)
    else:
        st.success("No stalled deals — all deals moving at expected velocity")
    d=r.pipeline.nlargest(10,"deal_value")[["company","stage","deal_value","blended_probability","weighted_value","expected_close"]].copy()
    d["deal_value"]=d["deal_value"].apply(lambda v:f"${v:,.0f}")
    d["blended_probability"]=d["blended_probability"].apply(lambda v:f"{v:.0%}")
    d["weighted_value"]=d["weighted_value"].apply(lambda v:f"${v:,.0f}")
    d.columns=["Company","Stage","Value","ML Prob","Weighted","Close"]
    st.markdown("##### Top 10 Deals by Value")
    st.dataframe(d,use_container_width=True,hide_index=True)

# ─── EXPENSES ─────────────────────────────────────────────────────────
elif "Expenses" in page:
    st.markdown("### Expense Forecast")
    # Budget vs actual for expenses
    exp_var=[v for v in variances if v.metric=="Expenses"]
    if exp_var:
        c1,c2,c3=st.columns(3)
        c1.metric("Forecast Accuracy",f"{accuracy}%")
        over=[v for v in exp_var if v.status=="Over Budget"]
        c2.metric("Over Budget Months",len(over))
        avg_var=np.mean([v.variance_pct for v in exp_var])
        c3.metric("Avg Variance",f"{avg_var:+.1f}%")
    cats=list(efc[0].categories.keys())
    CC=["#6366f1","#ef4444","#10b981","#7c3aed","#f59e0b","#06b6d4"]
    fig=go.Figure()
    for cat,col in zip(cats,CC):
        fig.add_trace(go.Bar(x=mo,y=[m.categories.get(cat,0) for m in efc],name=cat,marker_color=col))
    bl(fig,300);fig.update_layout(barmode="stack",yaxis_tickformat="$,.0f")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ─── CASH FLOW ────────────────────────────────────────────────────────
elif "Cash" in page:
    st.markdown("### Cash Flow")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Start",fmtM(starting_cash));c2.metric("Min",fmtM(fin.min_balance))
    c3.metric("End",fmtM(fin.ending_balance));c4.metric("Deficit",f"{fin.deficit_months} months")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hd,y=hist["cumulative_cash"].tolist(),name="Historical",mode="lines",line=dict(color="rgba(109,40,217,.35)",width=1.5)))
    fig.add_trace(go.Scatter(x=fx,y=[m.balance for m in cf],name="Forecast",mode="lines+markers",line=dict(color="#7c3aed",width=2.5),marker=dict(size=5)))
    bsc=sc.get("Low Risk");wsc=sc.get("High Risk")
    if bsc: fig.add_trace(go.Scatter(x=fx,y=[m.balance for m in bsc.cashflow],name="Low Risk",mode="lines",line=dict(color="#10b981",width=1.2,dash="dash")))
    if wsc: fig.add_trace(go.Scatter(x=fx,y=[m.balance for m in wsc.cashflow],name="High Risk",mode="lines",line=dict(color="#ef4444",width=1.2,dash="dash")))
    fig.add_hline(y=0,line_color="rgba(239,68,68,.5)",line_dash="dot")
    bl(fig,300);fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ─── HR & CHURN ───────────────────────────────────────────────────────
elif "HR" in page:
    st.markdown("### HR & Workforce Intelligence")
    ac=(enr_wf["att_risk_score"]>0.65).sum();aa=(enr_wf["att_risk_score"]>0.40).sum()
    immediate=[t for t in emp_trends if t.urgency=="immediate"]
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Headcount",hr.current_headcount);c2.metric("High Churn",hr.high_risk_employees)
    c3.metric("Immediate Action",len(immediate),"deteriorating trend")
    c4.metric("Att Critical",int(ac));c5.metric("Hires 6M",hr.projected_hires_6m)
    # Employee trend chart
    st.markdown("##### Churn Risk Trend — Top 10 At-Risk Employees")
    fig_trend=go.Figure()
    months_labels=[f"M-{5-i}" for i in range(6)]
    months_labels[-1]="Now"
    for t in emp_trends[:10]:
        color="#ef4444" if t.urgency=="immediate" else "#f59e0b" if t.urgency=="monitor" else "#10b981"
        fig_trend.add_trace(go.Scatter(x=months_labels,y=[v*100 for v in t.trajectory],
            name=f"{t.employee_id} ({t.department})",mode="lines+markers",
            line=dict(color=color,width=2),marker=dict(size=5)))
    fig_trend.add_hline(y=55,line_dash="dot",line_color="#ef4444",annotation_text="HIGH risk threshold")
    bl(fig_trend,300);fig_trend.update_layout(yaxis=dict(title="Churn probability %",range=[0,105]))
    st.plotly_chart(fig_trend,use_container_width=True,config={"displayModeBar":False})
    col_l,col_r=st.columns(2)
    with col_l:
        wf=enr_wf.copy()
        wf["risk_tier"]=wf["churn_prob"].apply(lambda v:"HIGH" if v>0.55 else "MEDIUM" if v>0.30 else "LOW")
        fig=go.Figure()
        for tier,tc in {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#10b981"}.items():
            sub=wf[wf["risk_tier"]==tier]
            fig.add_trace(go.Scatter(x=sub["tenure_months"],y=sub["workload_score"],mode="markers",
                name=f"{tier} ({len(sub)})",marker=dict(size=9,color=tc,opacity=0.75),text=sub["department"]))
        bl(fig,260);fig.update_layout(xaxis=dict(title="Tenure (months)"),yaxis=dict(title="Workload"))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col_r:
        trend_df=pd.DataFrame([{"Employee":t.employee_id,"Dept":t.department,
            "Current Risk":f"{t.current_risk:.0%}","Trend":t.trend,"6M Delta":f"{t.trend_delta:+.1%}",
            "Top Concern":t.top_concern,"Urgency":t.urgency} for t in emp_trends[:15]])
        st.dataframe(trend_df,use_container_width=True,hide_index=True,height=280)
    st.dataframe(r.dept_risk,use_container_width=True,hide_index=True)

# ─── ATTENDANCE ───────────────────────────────────────────────────────
elif "Attendance" in page:
    st.markdown("### Employee Attendance Intelligence")
    total=len(att_df);absent=(att_df["status"]=="absent").sum();late=(att_df["status"]=="late").sum()
    avg_att=enr_wf["att_rate"].mean();ac=(enr_wf["att_risk_score"]>0.65).sum()
    abs_cost=att_az.absenteeism_cost(att_pro,avg_daily_cost=hr.avg_monthly_salary/22)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Avg Attendance",f"{avg_att:.1%}");c2.metric("Absent Days",f"{absent:,}",f"{absent/total:.1%}")
    c3.metric("Late Arrivals",f"{late:,}",f"{late/total:.1%}");c4.metric("Critical Risk",int(ac))
    c5.metric("Absenteeism Cost",fmtM(abs_cost),"annual")
    col_l,col_r=st.columns(2)
    with col_l:
        fig=go.Figure(go.Histogram(x=enr_wf["att_rate"]*100,nbinsx=20,marker_color="#6366f1",marker_opacity=0.75))
        fig.add_vline(x=85,line_dash="dash",line_color="#ef4444",annotation_text="85% min")
        bl(fig,260);fig.update_layout(xaxis_title="Attendance rate (%)",yaxis_title="Employees")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col_r:
        sc2=att_df["status"].value_counts()
        fig2=go.Figure(go.Pie(labels=sc2.index,values=sc2.values,hole=0.55,marker=dict(colors=["#10b981","#ef4444","#f59e0b","#6366f1"])))
        bl(fig2,260);fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
    if not dep_att.empty:
        fig3=go.Figure(go.Bar(x=dep_att["department"],y=dep_att["avg_attendance"]*100,
            marker_color=dep_att["avg_attendance"].apply(lambda v:"#ef4444" if v<0.85 else "#f59e0b" if v<0.92 else "#10b981"),
            text=dep_att["avg_attendance"].apply(lambda v:f"{v:.0%}"),textposition="outside"))
        fig3.add_hline(y=85,line_dash="dash",line_color="#ef4444")
        bl(fig3,220);fig3.update_layout(yaxis=dict(title="Attendance %",range=[70,105]),xaxis=dict(showgrid=False))
        st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})
    att_tbl=pd.DataFrame([{"Employee":eid,"Attendance":f"{p.attendance_rate:.0%}",
        "Late rate":f"{p.late_rate:.0%}","Max streak":p.max_streak,
        "Mon/Fri %":f"{p.mon_fri_ratio:.0%}","30d trend":f"{p.trend_30d:+.1%}",
        "Risk":p.risk_flag,"Score":f"{p.risk_score:.2f}"} for eid,p in att_pro.items()]).sort_values("Score",ascending=False)
    st.dataframe(att_tbl,use_container_width=True,hide_index=True,height=300)

# ─── SKILL MATCH ──────────────────────────────────────────────────────
elif "Skill" in page:
    st.markdown("### 🔁 Employee Skill Match & Department Transfer")
    candidates=skill_engine.transfer_candidates(skill_results)
    dept_summary=skill_engine.dept_fit_summary(skill_results)
    c1,c2,c3=st.columns(3)
    c1.metric("Employees Analysed",len(skill_results))
    c2.metric("Transfer Candidates",len(candidates),"would benefit from a move")
    avg_fit=sum(r2.current_fit for r2 in skill_results)/max(len(skill_results),1)
    c3.metric("Avg Department Fit",f"{avg_fit:.0f}/100")
    if candidates:
        st.markdown("##### Recommended Transfers")
        for c in candidates[:6]:
            gain=c.recommended_fit-c.current_fit
            color="#10b981" if gain>=20 else "#f59e0b"
            st.markdown(f'<div style="border-radius:8px;padding:12px 16px;margin-bottom:8px;border-left:4px solid {color};background:white;font-size:13px">'
                f'<b>{c.employee_id}</b> — <span style="color:#555">{c.current_dept} ({c.current_fit:.0f}/100)</span>'
                f' → <b style="color:{color}">{c.recommended_dept} ({c.recommended_fit:.0f}/100)</b>'
                f' <span style="float:right;background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:11px">+{gain:.0f} fit pts</span><br>'
                f'<span style="font-size:12px;color:#555">{c.transfer_plan[:220]}...</span></div>',unsafe_allow_html=True)
    col_l,col_r=st.columns(2)
    with col_l:
        if not dept_summary.empty:
            fig_fit=go.Figure(go.Bar(x=dept_summary["avg_fit"],y=dept_summary["department"],orientation="h",
                marker_color=dept_summary["avg_fit"].apply(lambda v:"#10b981" if v>=65 else "#f59e0b" if v>=50 else "#ef4444"),
                text=dept_summary["avg_fit"].apply(lambda v:f"{v:.0f}"),textposition="outside"))
            bl(fig_fit,320);fig_fit.update_layout(xaxis=dict(title="Avg fit score",range=[0,110]),yaxis=dict(showgrid=False))
            st.plotly_chart(fig_fit,use_container_width=True,config={"displayModeBar":False})
    with col_r:
        selected=st.selectbox("Explore employee",[r2.employee_id for r2 in skill_results],key="skill_emp")
        emp_result=next((r2 for r2 in skill_results if r2.employee_id==selected),None)
        if emp_result:
            scores_df=pd.DataFrame([{"Department":s.department,"Fit Score":s.fit_score} for s in emp_result.all_scores]).sort_values("Fit Score",ascending=False)
            fig_emp=go.Figure(go.Bar(x=scores_df["Fit Score"],y=scores_df["Department"],orientation="h",
                marker_color=scores_df["Fit Score"].apply(lambda v:"#10b981" if v>=65 else "#f59e0b" if v>=50 else "#ef4444")))
            bl(fig_emp,320);fig_emp.update_layout(xaxis=dict(range=[0,110]),yaxis=dict(showgrid=False))
            st.plotly_chart(fig_emp,use_container_width=True,config={"displayModeBar":False})
            color="#10b981" if emp_result.should_transfer else "#6366f1"
            st.markdown(f'<div style="padding:10px;border-radius:8px;background:#f8f9fa;border-left:4px solid {color};font-size:13px">{emp_result.transfer_plan}</div>',unsafe_allow_html=True)
    fit_tbl=pd.DataFrame([{"Employee":r2.employee_id,"Current Dept":r2.current_dept,
        "Current Fit":f"{r2.current_fit:.0f}/100","Best Dept":r2.recommended_dept,
        "Best Fit":f"{r2.recommended_fit:.0f}/100","Gain":f"+{r2.recommended_fit-r2.current_fit:.0f}",
        "Transfer?":"✅ Yes" if r2.should_transfer else "—"
    } for r2 in sorted(skill_results,key=lambda x:x.recommended_fit-x.current_fit,reverse=True)])
    st.dataframe(fit_tbl,use_container_width=True,hide_index=True,height=300)

# ─── TIMELINE ─────────────────────────────────────────────────────────
elif "Timeline" in page:
    st.markdown("### 📅 Multi-Horizon Forecast")
    horizon=st.radio("Horizon",["📆 Daily (30 days)","📅 Weekly (12 weeks)","📊 Monthly (6 months)","📈 Yearly (3 years)"],horizontal=True)
    if "Daily" in horizon:
        daily=ts_engine.daily_forecast(30)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("30-day revenue",fmtM(sum(d.revenue for d in daily)))
        c2.metric("30-day expenses",fmtM(sum(d.expenses for d in daily)))
        c3.metric("30-day net",fmtM(sum(d.net for d in daily)))
        c4.metric("Min daily balance",fmtM(min(d.balance for d in daily)))
        wk_days=[d for d in daily if not d.is_weekend]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=[d.day_label for d in wk_days],y=[d.net for d in wk_days],
            name="Daily net",marker_color=["#10b981" if d.net>=0 else "#ef4444" for d in wk_days]))
        fig.add_trace(go.Scatter(x=[d.day_label for d in wk_days],y=[d.balance for d in wk_days],
            name="Running balance",mode="lines",line=dict(color="#6366f1",width=2),yaxis="y2"))
        bl(fig,320);fig.update_layout(yaxis_tickformat="$,.0f",yaxis2=dict(overlaying="y",side="right",tickformat="$,.0f"))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        daily_tbl=pd.DataFrame([{"Date":d.date,"Day":d.day_label,"Revenue":fmtM(d.revenue),
            "Expenses":fmtM(d.expenses),"Net":fmtM(d.net),"Balance":fmtM(d.balance),
            "Weekend":"✓" if d.is_weekend else "","Alert":d.alert} for d in daily])
        st.dataframe(daily_tbl,use_container_width=True,hide_index=True,height=280)
    elif "Weekly" in horizon:
        weekly=ts_engine.weekly_forecast(12)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("12-week revenue",fmtM(sum(w.revenue for w in weekly)))
        c2.metric("12-week expenses",fmtM(sum(w.expenses for w in weekly)))
        c3.metric("12-week net",fmtM(sum(w.net for w in weekly)))
        c4.metric("Min balance",fmtM(min(w.balance for w in weekly)))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=[w.week_label for w in weekly],y=[w.upper for w in weekly],fill=None,mode="lines",line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=[w.week_label for w in weekly],y=[w.lower for w in weekly],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(99,102,241,.12)",name="90% CI"))
        fig.add_trace(go.Scatter(x=[w.week_label for w in weekly],y=[w.net for w in weekly],name="Weekly net",mode="lines+markers",line=dict(color="#6366f1",width=2.5),marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=[w.week_label for w in weekly],y=[w.balance for w in weekly],name="Balance",mode="lines",line=dict(color="#7c3aed",width=1.5,dash="dash")))
        bl(fig,300);fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    elif "Monthly" in horizon:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fx,y=[m.upper_90 for m in rfc],fill=None,mode="lines",line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=fx,y=[m.lower_90 for m in rfc],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(99,102,241,.10)",name="90% CI"))
        fig.add_trace(go.Scatter(x=hd,y=hist["revenue"].tolist(),name="Revenue (hist)",mode="lines",line=dict(color="rgba(99,102,241,.3)",width=1.5)))
        fig.add_trace(go.Scatter(x=fx,y=[m.blended_revenue for m in rfc],name="Revenue (fc)",mode="lines+markers",line=dict(color="#6366f1",width=2.5),marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=fx,y=[m.balance for m in cf],name="Cash balance",mode="lines",line=dict(color="#7c3aed",width=2,dash="dash")))
        bl(fig,300);fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        tbl=pd.DataFrame({"Month":fmo,"Revenue":[fmtM(m.revenue) for m in cf],
            "Expenses":[fmtM(m.expenses) for m in cf],"Net":[fmtM(m.net) for m in cf],
            "Balance":[fmtM(m.balance) for m in cf],"Status":[m.alert for m in cf]})
        st.dataframe(tbl,use_container_width=True,hide_index=True)
    else:
        yearly=ts_engine.yearly_forecast(3,current_headcount=hr.current_headcount)
        c1,c2,c3=st.columns(3)
        c1.metric(f"Year {yearly[0].year} Revenue",fmtM(yearly[0].revenue),f"{yearly[0].revenue_growth:+.1%}")
        c2.metric(f"Year {yearly[1].year} Revenue",fmtM(yearly[1].revenue),f"{yearly[1].revenue_growth:+.1%}")
        c3.metric(f"Year {yearly[2].year} Revenue",fmtM(yearly[2].revenue),f"{yearly[2].revenue_growth:+.1%}")
        fig=go.Figure()
        labels=[y.year_label for y in yearly]
        fig.add_trace(go.Bar(x=labels,y=[y.revenue for y in yearly],name="Revenue",marker_color="#6366f1",marker_opacity=0.8))
        fig.add_trace(go.Bar(x=labels,y=[y.expenses for y in yearly],name="Expenses",marker_color="#ef4444",marker_opacity=0.7))
        fig.add_trace(go.Scatter(x=labels,y=[y.net for y in yearly],name="Net profit",mode="lines+markers",line=dict(color="#10b981",width=3),marker=dict(size=10)))
        bl(fig,280);fig.update_layout(barmode="group",yaxis_tickformat="$,.0f")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        yr_tbl=pd.DataFrame({"Year":[y.year_label for y in yearly],
            "Revenue":[fmtM(y.revenue) for y in yearly],"Expenses":[fmtM(y.expenses) for y in yearly],
            "Net":[fmtM(y.net) for y in yearly],"YoY Growth":[f"{y.revenue_growth:+.1%}" for y in yearly],
            "Headcount Est.":[y.headcount_est for y in yearly]})
        st.dataframe(yr_tbl,use_container_width=True,hide_index=True)

# ─── ANALYTICS ────────────────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown("### 📊 Advanced Analytics")
    tab1,tab2,tab3=st.tabs(["🔔 Anomaly Detection","📈 Budget vs Actual","⚡ Deal Velocity"])
    with tab1:
        st.markdown("##### Statistical anomalies detected in historical data")
        anom_sum=AnomalyDetector().summary(anomalies)
        c1,c2,c3=st.columns(3)
        c1.metric("Total Anomalies",anom_sum["total"])
        c2.metric("Critical",anom_sum["critical"])
        c3.metric("Warnings",anom_sum["warnings"])
        if anomalies:
            for a in anomalies:
                sev=a.severity
                bg="#fef2f2" if sev=="CRITICAL" else "#fffbeb"
                bd="#ef4444" if sev=="CRITICAL" else "#f59e0b"
                ic="🔴" if sev=="CRITICAL" else "⚠️"
                st.markdown(f'<div style="border-radius:8px;padding:11px 14px;margin-bottom:7px;border-left:4px solid {bd};background:{bg};font-size:13px">'
                    f'<b>{ic} {a.headline}</b><br>'
                    f'<span style="color:#555;font-size:12px">{a.metric} | Z-score: {a.z_score:.2f} | Actual: {fmtM(a.actual)} vs Expected: {fmtM(a.expected)}</span><br>'
                    f'<span style="font-size:11px;font-style:italic;color:#1d4ed8">→ {a.action}</span></div>',unsafe_allow_html=True)
        else:
            st.success("No statistical anomalies detected in historical data")
    with tab2:
        st.markdown("##### Forecast vs actual — how accurate were our predictions?")
        c1,c2=st.columns(2)
        c1.metric("Forecast Accuracy Score",f"{accuracy}%")
        c2.metric("Periods Analysed",len(variances))
        if variances:
            fig_bva=go.Figure()
            rev_var=[v for v in variances if v.metric=="Revenue"]
            exp_var=[v for v in variances if v.metric=="Expenses"]
            if rev_var:
                fig_bva.add_trace(go.Bar(name="Revenue Budget",x=[v.month for v in rev_var],y=[v.budget for v in rev_var],marker_color="rgba(99,102,241,.5)"))
                fig_bva.add_trace(go.Scatter(name="Revenue Actual",x=[v.month for v in rev_var],y=[v.actual for v in rev_var],mode="lines+markers",line=dict(color="#6366f1",width=2.5)))
            bl(fig_bva,280);fig_bva.update_layout(yaxis_tickformat="$,.0f",barmode="group")
            st.plotly_chart(fig_bva,use_container_width=True,config={"displayModeBar":False})
            var_tbl=pd.DataFrame([{"Month":v.month,"Metric":v.metric,"Budget":fmtM(v.budget),
                "Actual":fmtM(v.actual),"Variance":fmtM(v.variance),
                "Variance %":f"{v.variance_pct:+.1f}%","Status":v.status} for v in variances])
            st.dataframe(var_tbl,use_container_width=True,hide_index=True)
        else:
            st.info("Need at least 6 months of history to compute budget vs actual")
    with tab3:
        st.markdown("##### Deal velocity — which deals are stalled?")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Critical (overdue)",dv_summary["critical"])
        c2.metric("Warning zone",dv_summary["warning"])
        c3.metric("On track",dv_summary["on_track"])
        c4.metric("Stalled value",fmtM(dv_summary["stalled_value"]))
        # Stage benchmark chart
        bench_df=pd.DataFrame([{"Stage":s,"Avg Days":b["avg"],"Warning at":b["warn"],"Critical at":b["critical"]} for s,b in {"Lead":{"avg":85,"warn":60,"critical":90},"Qualified":{"avg":60,"warn":45,"critical":75},"Proposal":{"avg":38,"warn":30,"critical":50},"Negotiation":{"avg":16,"warn":12,"critical":22}}.items()])
        fig_bench=go.Figure()
        fig_bench.add_trace(go.Bar(x=bench_df["Stage"],y=bench_df["Avg Days"],name="Avg days",marker_color="#6366f1",marker_opacity=0.7))
        fig_bench.add_trace(go.Bar(x=bench_df["Stage"],y=bench_df["Critical at"],name="Critical threshold",marker_color="#ef4444",marker_opacity=0.5))
        bl(fig_bench,220);fig_bench.update_layout(barmode="group",yaxis_title="Days in stage")
        st.plotly_chart(fig_bench,use_container_width=True,config={"displayModeBar":False})
        if dv_alerts:
            alert_tbl=pd.DataFrame([{"Deal":a.deal_id,"Company":a.company,"Stage":a.stage,
                "Days":a.days_in_stage,"Benchmark":a.benchmark_avg,"Overdue":a.overdue_by,
                "Value":fmtM(a.deal_value),"Prob":f"{a.ml_probability:.0%}","Status":a.severity} for a in dv_alerts if a.severity!="OK"])
            if not alert_tbl.empty:
                st.dataframe(alert_tbl,use_container_width=True,hide_index=True)

# ─── RISK ─────────────────────────────────────────────────────────────
elif "Risk" in page:
    st.markdown("### Risk Intelligence")
    cols=st.columns(5)
    for col,(name,s) in zip(cols,sc.items()): col.metric(name,f"{s.risk_score:.1f}",s.risk_grade)
    fig=go.Figure()
    for name,s in sc.items():
        fig.add_trace(go.Scatter(x=fx,y=[m.balance for m in s.cashflow],name=name,mode="lines",line=dict(color=SCC.get(name,"#6b7280"),width=2)))
    fig.add_hline(y=0,line_color="rgba(239,68,68,.4)",line_dash="dot")
    bl(fig,280);fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ─── SCENARIOS ────────────────────────────────────────────────────────
elif "Scenario" in page:
    st.markdown("### Scenario Comparison")
    scc=st.columns(5)
    for col,(name,s) in zip(scc,sc.items()):
        color=SCC.get(name,"#6b7280"); delta=s.finance_summary.total_net_6m-r.base_scenario.finance_summary.total_net_6m
        vc="#10b981" if s.finance_summary.min_balance>0 else "#ef4444"
        col.markdown(f'<div style="background:white;border:1px solid {color}33;border-top:3px solid {color};border-radius:10px;padding:12px">'
            f'<div style="font-size:10px;color:#888">{name}</div>'
            f'<div style="font-size:18px;font-weight:700;color:{color}">{fmtM(s.finance_summary.total_net_6m)}</div>'
            f'<div style="font-size:9.5px;color:#888">{"+" if delta>=0 else ""}{fmtM(delta)} vs base</div>'
            f'<div style="font-size:9px;font-weight:700;color:{vc}">{"Viable" if s.finance_summary.min_balance>0 else "Deficit"}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>',unsafe_allow_html=True)
    metric=st.selectbox("Metric",["Net cash","Revenue","Expenses","Balance"])
    getter={"Net cash":lambda s:[m.net for m in s.cashflow],"Revenue":lambda s:[m.revenue for m in s.cashflow],"Expenses":lambda s:[m.expenses for m in s.cashflow],"Balance":lambda s:[m.balance for m in s.cashflow]}[metric]
    fig=go.Figure()
    for name,s in sc.items():
        fig.add_trace(go.Scatter(x=mo,y=getter(s),name=name,mode="lines+markers",line=dict(color=SCC.get(name,"#6b7280"),width=2.5 if name=="Base" else 1.8)))
    bl(fig,280);fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    s1,s2,s3=st.columns(3)
    s1.metric("Simulated 6M net",fmtM(sum(snv)))
    s2.metric("Min balance",fmtM(min(sbv)))
    s3.metric("Viable","Yes" if min(sbv)>0 else "No")

# ─── AI ADVISOR ───────────────────────────────────────────────────────
elif "AI" in page:
    st.markdown("### 🧠 NABOS AI Advisor")
    st.caption("Powered by Claude — CFO + COO + Strategy + People Officer combined")
    col_l,col_r=st.columns([3,1])
    with col_r:
        st.markdown("**📋 Structured Plans**")
        if st.button("🌅 Morning briefing",use_container_width=True):
            st.session_state["pq"]="Generate my executive morning briefing for today with exact numbers from the forecast."; st.rerun()
        if st.button("💰 Full budget plan",use_container_width=True):
            st.session_state["pq"]="Build me a complete annual budget plan with all line items, amounts, and rationale."; st.rerun()
        if st.button("🏢 Acquisition strategy",use_container_width=True):
            st.session_state["pq"]="I want to do a financial takeover this year. What is my exact acquisition budget and strategy?"; st.rerun()
        if st.button("👥 Hiring plan",use_container_width=True):
            st.session_state["pq"]="Which departments should I hire in and how many? Fresh graduates or experienced? Complete plan with role titles, salary bands, and timeline."; st.rerun()
        if st.button("🗓️ Monthly operating plan",use_container_width=True):
            st.session_state["pq"]="Run the company for me this month. Week-by-week operating plan covering finance, sales, HR, and operations with exact targets."; st.rerun()
        if st.button("🔁 Skill transfer advice",use_container_width=True):
            cands=skill_engine.transfer_candidates(skill_results)
            if cands:
                top3="; ".join(f"{c.employee_id} ({c.current_dept}→{c.recommended_dept})" for c in cands[:3])
                st.session_state["pq"]=f"We have {len(cands)} employees recommended for transfers: {top3}. Should we action these? What is the best approach and financial impact?"
            else:
                st.session_state["pq"]="Analyse our workforce skill fit and tell me if any employees should move departments."
            st.rerun()
        if st.button("📈 3-year strategy",use_container_width=True):
            yearly=ts_engine.yearly_forecast(3,current_headcount=hr.current_headcount)
            st.session_state["pq"]=f"Based on our 3-year forecast (Y1: {fmtM(yearly[0].revenue)}, Y2: {fmtM(yearly[1].revenue)}, Y3: {fmtM(yearly[2].revenue)}), build me a complete 3-year strategic plan with growth targets, hiring roadmap, and investment priorities."
            st.rerun()
        if st.button("🔔 Explain anomalies",use_container_width=True):
            if anomalies:
                anom_list="; ".join(f"{a.headline}" for a in anomalies[:3])
                st.session_state["pq"]=f"Our AI detected these anomalies in our financial data: {anom_list}. Explain what likely caused each one and what we should do."
            else:
                st.session_state["pq"]="Are there any unusual patterns in our financial history I should be aware of?"
            st.rerun()
        st.divider()
        st.markdown("**⚡ Quick questions**")
        for q in ["Will we run out of cash?","Who is most likely to quit?","What if revenue drops 20%?","Which dept has worst attendance?","Top 3 priorities today?"]:
            if st.button(q,use_container_width=True,key=f"q{hash(q)}"):
                st.session_state["pq"]=q; st.rerun()
        st.divider()
        msg_count=len(st.session_state.get("chat",[]))
        st.caption(f"💬 {msg_count} messages in history")
        if st.button("↺ Clear chat",use_container_width=True):
            ai.reset(); st.session_state["chat"]=[]; st.rerun()
    with col_l:
        if "chat" not in st.session_state:
            st.session_state["chat"]=[{"role":m.role,"content":m.content} for m in ai.history] if ai.history else []
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        pending=st.session_state.pop("pq",None)
        user_input=st.chat_input("Ask anything — budgets, acquisitions, hiring, strategy, anomalies, operations...")
        question=pending or user_input
        if question:
            st.session_state["chat"].append({"role":"user","content":question})
            with st.chat_message("user"): st.markdown(question)
            with st.chat_message("assistant"):
                ph=st.empty(); full=""
                for token in ai.ask_stream(question):
                    full+=token; ph.markdown(full+"▌")
                ph.markdown(full)
            st.session_state["chat"].append({"role":"assistant","content":full})
            ai.save_history()

# ─── DECISIONS ────────────────────────────────────────────────────────
elif "Decision" in page:
    st.markdown("### 🎯 Decision Intelligence")
    st.caption("Every alert links to the exact page — click Open to navigate directly")

    def category_to_page(category):
        cat=category.lower()
        if "attendance" in cat: return "📋 Attendance"
        if "hr / cost" in cat:  return "👥 HR & Churn"
        if "hr" in cat:         return "👥 HR & Churn"
        if "revenue" in cat:    return "💰 Revenue"
        if "expense" in cat:    return "📉 Expenses"
        if "cash" in cat:       return "💵 Cash Flow"
        if "risk" in cat:       return "🌐 Risk"
        if "skill" in cat:      return "🔁 Skill Match"
        if "pipeline" in cat:   return "💰 Revenue"
        if "anomaly" in cat:    return "📊 Analytics"
        if "deal" in cat:       return "💰 Revenue"
        if "velocity" in cat:   return "💰 Revenue"
        return "🏠 Overview"

    crit=[i for i in r.all_insights if i.get("severity")=="CRITICAL"]
    warn=[i for i in r.all_insights if i.get("severity")=="WARNING"]
    if crit: st.error(f"🔴 {len(crit)} critical alert(s) require immediate attention")
    elif warn: st.warning(f"⚠️ {len(warn)} warning(s) — action recommended")
    else: st.success("✅ No critical issues")

    # Build master alert list from all sources
    all_ins=list(r.all_insights)
    # Add anomalies
    for a in anomalies[:3]:
        all_ins.append({"severity":a.severity,"category":"Anomaly Detection",
            "headline":a.headline,"detail":f"{a.metric} | Z-score: {a.z_score:.2f} | Actual: {fmtM(a.actual)} vs Expected: {fmtM(a.expected)}",
            "action":a.action})
    # Add stalled deals
    stalled=[x for x in dv_alerts if x.severity=="CRITICAL"]
    if stalled:
        all_ins.append({"severity":"WARNING","category":"Deal Velocity",
            "headline":f"{len(stalled)} deals critically overdue — {fmtM(dv_summary['stalled_value'])} pipeline at risk",
            "detail":f"Top stalled: {stalled[0].company} in {stalled[0].stage} for {stalled[0].days_in_stage} days (avg {stalled[0].benchmark_avg})",
            "action":"Go to Revenue page → Deal Velocity section. Call overdue accounts today."})
    # Add skill transfers
    cands=skill_engine.transfer_candidates(skill_results)
    if cands:
        all_ins.append({"severity":"INFO","category":"Skill Match",
            "headline":f"{len(cands)} employees recommended for department transfers",
            "detail":f"Top: {cands[0].employee_id} ({cands[0].current_dept} → {cands[0].recommended_dept}, +{cands[0].recommended_fit-cands[0].current_fit:.0f} fit pts)",
            "action":"Go to Skill Match page to review transfer plans"})
    # Add employee trend alerts
    immediate=[t for t in emp_trends if t.urgency=="immediate"]
    if immediate:
        all_ins.append({"severity":"WARNING","category":"HR Trend",
            "headline":f"{len(immediate)} employees showing deteriorating churn risk trend",
            "detail":f"Most urgent: {immediate[0].employee_id} ({immediate[0].department}), trend: {immediate[0].trend_delta:+.1%} in 6 months",
            "action":"Go to HR & Churn page → Employee trend chart. Schedule 1-on-1s immediately."})

    # Sort by severity
    sev_order={"CRITICAL":0,"WARNING":1,"INFO":2}
    all_ins.sort(key=lambda i:sev_order.get(i.get("severity","INFO"),2))

    st.markdown("---")
    for ins in all_ins:
        sev=ins.get("severity","INFO"); cat=ins.get("category","")
        target=category_to_page(cat)
        page_label=target.split(" ",1)[1] if " " in target else target
        bg={"CRITICAL":"#fef2f2","WARNING":"#fffbeb","INFO":"#eff6ff"}.get(sev,"#eff6ff")
        bd={"CRITICAL":"#ef4444","WARNING":"#f59e0b","INFO":"#3b82f6"}.get(sev,"#3b82f6")
        ic={"CRITICAL":"🔴","WARNING":"⚠️","INFO":"ℹ️"}.get(sev,"ℹ️")
        col_card,col_btn=st.columns([5,1])
        with col_card:
            st.markdown(f'<div style="border-radius:8px;padding:12px 16px;margin-bottom:4px;border-left:4px solid {bd};background:{bg};font-size:13px">'
                f'<div style="margin-bottom:4px"><span style="background:{bd};color:white;font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;margin-right:6px">{sev}</span>'
                f'<span style="font-size:10px;color:#888">→ {target}</span></div>'
                f'<b>{ic} {ins["headline"]}</b><br>'
                f'<span style="color:#555;font-size:12px">{ins.get("detail","")}</span><br>'
                f'<span style="font-size:11px;font-style:italic;color:#1d4ed8">→ {ins.get("action","")}</span></div>',unsafe_allow_html=True)
        with col_btn:
            st.markdown('<div style="height:8px"></div>',unsafe_allow_html=True)
            if st.button(f"Open",key=f"nav_{hash(ins['headline'])}",use_container_width=True,type="primary"):
                st.session_state["_nabos_page"]=target; st.rerun()
        st.markdown('<div style="height:2px"></div>',unsafe_allow_html=True)

    if "_nabos_page" in st.session_state:
        dest=st.session_state.pop("_nabos_page")
        st.info(f"💡 Click **{dest}** in the left sidebar to go there now.")
