from __future__ import annotations
import json, urllib.request
from dataclasses import dataclass
from typing import Generator, List

ANTHROPIC_KEY = ""
MODEL         = "claude-sonnet-4-5"

DEPT_PROFILES = {
    "Manufacturing":    {"fresh_grad_ratio":0.20,"avg_salary_pkr":120000,"skill_type":"Mostly experienced — safety-critical roles need 2+ years"},
    "Sales":            {"fresh_grad_ratio":0.40,"avg_salary_pkr":110000,"skill_type":"Mix — territory reps can be fresh, KAMs need 3-5 years"},
    "Marketing":        {"fresh_grad_ratio":0.30,"avg_salary_pkr":150000,"skill_type":"Experienced preferred — brand managers need category knowledge"},
    "Supply Chain":     {"fresh_grad_ratio":0.25,"avg_salary_pkr":130000,"skill_type":"Experienced — demand planning needs proven track record"},
    "Finance":          {"fresh_grad_ratio":0.45,"avg_salary_pkr":140000,"skill_type":"Mixed — ACCA/CA freshers fine at junior level"},
    "HR":               {"fresh_grad_ratio":0.35,"avg_salary_pkr":110000,"skill_type":"Mixed — business partners need experience, L&D can hire fresh"},
    "IT":               {"fresh_grad_ratio":0.50,"avg_salary_pkr":160000,"skill_type":"Fresh grads excellent — tech skills trainable"},
    "R&D":              {"fresh_grad_ratio":0.15,"avg_salary_pkr":180000,"skill_type":"Experienced only — food scientists need PhDs or 5+ years"},
    "Customer Success": {"fresh_grad_ratio":0.40,"avg_salary_pkr":100000,"skill_type":"Mix — junior fresh, senior accounts need 3+ years"},
    "Engineering":      {"fresh_grad_ratio":0.50,"avg_salary_pkr":160000,"skill_type":"Fresh grads excellent — strong university pipelines"},
    "Product":          {"fresh_grad_ratio":0.20,"avg_salary_pkr":180000,"skill_type":"Experienced preferred — product sense needs market exposure"},
    "G&A":              {"fresh_grad_ratio":0.35,"avg_salary_pkr":100000,"skill_type":"Mixed — admin fresh, legal/compliance senior"},
}

STARTER_QUESTIONS = [
    "What is our biggest financial risk in the next 6 months?",
    "I want to do a financial takeover this year — what is my acquisition budget?",
    "Build me a complete budget plan for this year.",
    "Which departments should I hire in and should they be fresh graduates or experienced?",
    "Run the company for me this month — give me a complete operating plan.",
    "Will we run out of cash? Under which scenario?",
    "Which employees should I be most worried about losing?",
    "What happens if revenue drops 20%?",
    "Which department has the worst attendance problem?",
    "What should I do in the next 30 days to protect cash flow?",
]


def build_system_prompt(result, company_name):
    fin  = result.finance_summary
    hr   = result.hr_summary
    cf   = result.cashflow
    efc  = result.expense_fc
    pipe = result.pipeline
    sc   = result.scenarios
    wf   = result.workforce

    def fmt(v):
        a = abs(v); s = "$" if v >= 0 else "-$"
        if a >= 1e9: return s + f"{a/1e9:.2f}B"
        if a >= 1e6: return s + f"{a/1e6:.2f}M"
        if a >= 1e3: return s + f"{a/1e3:.0f}K"
        return s + f"{a:.0f}"

    months   = [m.month_label for m in cf]
    cf_lines = "\n".join(f"  {m.month_label}: rev={fmt(m.revenue)}, exp={fmt(m.expenses)}, net={fmt(m.net)}, balance={fmt(m.balance)}, {m.alert}" for m in cf)

    exp_lines = "N/A"
    if efc:
        cats = list(efc[0].categories.keys())
        exp_lines = "\n".join(f"  {cat}: " + " | ".join(f"{m.month_label[:3]}={fmt(m.categories.get(cat,0))}" for m in efc) for cat in cats)

    sc_lines = "\n".join(
        f"  {name}: net={fmt(s.finance_summary.total_net_6m)}, min_bal={fmt(s.finance_summary.min_balance)}, viable={'YES' if s.finance_summary.min_balance>0 else 'NO'}, risk={s.risk_score:.0f}/100 ({s.risk_grade})"
        for name, s in sc.items())

    deal_rows = "N/A"
    if "deal_value" in pipe.columns:
        deal_rows = "\n".join(
            f"  {str(r.get('company','?'))}: {str(r.get('stage','?'))}, val={fmt(float(r.get('deal_value',0)))}, prob={float(r.get('blended_probability',0)):.0%}, close={r.get('expected_close','?')}"
            for _, r in pipe.nlargest(12, "deal_value").iterrows())

    top_churn = sorted(result.churn_preds, key=lambda p: p.churn_prob, reverse=True)[:6]
    churn_lines = "\n".join(
        f"  {p.employee_id} ({p.department}): churn={p.churn_prob:.0%}, risk={p.risk_tier}, driver={p.top_driver}, att={p.attendance_rate:.0%} ({p.attendance_flag})"
        for p in top_churn)

    dept_hc = "N/A"
    if "department" in wf.columns:
        dc  = wf["department"].value_counts()
        dch = wf.groupby("department")["churn_prob"].mean()
        rows = []
        for dept, count in dc.items():
            p = DEPT_PROFILES.get(dept, {})
            rows.append(f"  {dept}: count={count}, avg_churn={dch.get(dept,0):.0%}, fresh_grad%={p.get('fresh_grad_ratio',0.35):.0%}, avg_salary=PKR {p.get('avg_salary_pkr',100000):,}/mo, style={p.get('skill_type','Mixed')}")
        dept_hc = "\n".join(rows)

    ins_lines = "\n".join(
        f"  [{i.get('severity','?')}] {i.get('category','?')}: {i.get('headline','')}"
        for i in result.all_insights)

    free_cash  = fin.total_net_6m
    reserve    = max(fin.total_expenses_6m * 0.25, 0)
    deployable = max(free_cash - reserve, 0)
    ebitda     = fin.total_net_6m * 2
    ma_total   = deployable + ebitda * 3.0
    ma_ev      = ma_total / 1.25
    annual_rev = fin.total_revenue_6m * 2
    ml         = result.ml_metrics
    ml_line    = f"AUC={ml.cv_auc:.3f}, win_rate={ml.win_rate:.0%}" if ml else "N/A"

    return f"""You are NABOS AI, the Chief Executive Intelligence System for {company_name}.

You have the full authority of: CFO + COO + Chief People Officer + Chief Strategy Officer + McKinsey Partner.
NEVER refuse a business question. Always give a complete, specific, quantified answer.

RULES:
1. Always cite EXACT numbers from the data below
2. For budget questions: give SPECIFIC line-item amounts
3. For hiring: give DEPARTMENT, NUMBER, LEVEL, ROLE TITLE, SALARY in PKR
4. For acquisition: give exact budget, financing structure, target criteria, timeline
5. For run-the-company: produce a WEEK-BY-WEEK operating plan
6. End EVERY response with Next action: one specific thing to do TODAY

COMPANY: {company_name} | PERIOD: {months[0]} to {months[-1]}

FINANCIALS (6M):
  Revenue: {fmt(fin.total_revenue_6m)} | Expenses: {fmt(fin.total_expenses_6m)} | Net: {fmt(fin.total_net_6m)}
  End balance: {fmt(fin.ending_balance)} | Min balance: {fmt(fin.min_balance)}
  Deficit months: {fin.deficit_months}/6 | Burn ratio: {fin.avg_burn_ratio:.1%} | Peak: {fin.peak_revenue_month}

MONTHLY CASH FLOW:
{cf_lines}

EXPENSE BREAKDOWN:
{exp_lines}

ANNUAL BUDGET FRAMEWORK:
  Full-year revenue: {fmt(annual_rev)}
  Marketing: {fmt(annual_rev*0.055)} (5.5%) | R&D: {fmt(annual_rev*0.012)} (1.2%)
  Capex: {fmt(annual_rev*0.025)} (2.5%) | Hiring: {fmt(hr.total_hiring_cost_6m*2)}
  Contingency: {fmt(annual_rev*0.10)} (10%)

ACQUISITION CAPACITY:
  Deployable cash: {fmt(deployable)} | Debt (3x EBITDA): {fmt(ebitda*3)}
  TOTAL FIREPOWER: {fmt(ma_total)} | Max target EV: {fmt(ma_ev)}
  Reserve to keep: {fmt(reserve)} | FMCG multiple: 8.5x EV/EBITDA

PIPELINE ({len(pipe)} deals | {ml_line}):
{deal_rows}

HR:
  Headcount: {hr.current_headcount} | High churn: {hr.high_risk_employees} | Att critical: {hr.attendance_critical}
  Hires needed 6M: {hr.projected_hires_6m} | Hiring cost: {fmt(hr.total_hiring_cost_6m)}
  Avg salary: {fmt(hr.avg_monthly_salary)}/mo | Churn: {hr.churn_rate_pct:.1f}%

DEPARTMENTS:
{dept_hc}

SALARY BANDS: Fresh=PKR 70K/mo | Mid=PKR 140K/mo | Senior=PKR 280K/mo
Hire lead time: 8 weeks | Onboarding: 3 months

TOP CHURN RISKS:
{churn_lines}

SCENARIOS:
{sc_lines}

ALERTS:
{ins_lines}

BENCHMARKS: Gross margin 35% | EBITDA 14% | Rev/employee PKR 4.5M/yr | Marketing 5.5% | Healthy churn 12%"""


@dataclass
class ChatMessage:
    role: str
    content: str


class AIEngine:

    def __init__(self, nabos_result, company_name="the company"):
        self.result        = nabos_result
        self.company_name  = company_name
        self.history       = []
        self.system_prompt = build_system_prompt(nabos_result, company_name)
        self.STARTER_QUESTIONS = STARTER_QUESTIONS

    def has_api_key(self):
        return True

    def _call(self, stream=False):
        msgs    = [{"role": m.role, "content": m.content} for m in self.history]
        payload = json.dumps({
            "model":      MODEL,
            "max_tokens": 4000,
            "system":     self.system_prompt,
            "stream":     stream,
            "messages":   msgs,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=120)

    def ask(self, user_message):
        self.history.append(ChatMessage("user", user_message))
        try:
            with self._call(stream=False) as resp:
                data     = json.loads(resp.read())
                response = data["content"][0]["text"]
            self.history.append(ChatMessage("assistant", response))
            return response
        except Exception as e:
            err = f"Error: {str(e)[:400]}"
            self.history.append(ChatMessage("assistant", err))
            return err

    def ask_stream(self, user_message):
        self.history.append(ChatMessage("user", user_message))
        full = ""
        try:
            with self._call(stream=True) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "): continue
                    ds = line[6:]
                    if ds == "[DONE]": break
                    try:
                        d = json.loads(ds)
                        if d.get("type") == "content_block_delta":
                            token = d.get("delta", {}).get("text", "")
                            if token:
                                full += token
                                yield token
                    except: continue
        except Exception as e:
            err = f"Error: {str(e)[:300]}"
            yield err
            full += err
        self.history.append(ChatMessage("assistant", full))

    def reset(self):
        self.history = []

    def generate_executive_briefing(self):
        return self.ask("Write an executive morning briefing with exact numbers. Status (HEALTHY/WATCH/ALERT/CRITICAL), Financial snapshot 3 bullets, People snapshot 2 bullets, Top 3 actions today with owner and deadline, biggest risk flag, Next action CEO does in 2 hours. Under 250 words.")

    def generate_budget_plan(self):
        return self.ask("Build a complete annual budget plan: Executive Summary, Revenue Budget monthly, Expense Budget by category vs FMCG benchmarks, Headcount Budget by department, Capex, M&A Reserve, Contingency, Monthly Cash Flow Targets, Budget Governance. Exact numbers. CFO board quality.")

    def generate_hiring_plan(self):
        return self.ask("Complete 6-month hiring plan. Each department: current headcount, hires needed, role titles, fresh vs mid vs senior split with reasoning, PKR salary, priority, interview criteria. Include Fresh vs Experienced framework, monthly timeline, total cost breakdown, 90-day onboarding plan.")

    def generate_operating_plan(self, period="this month"):
        return self.ask(f"Act as CEO. Complete operating plan for {period}. Week-by-week: Finance, Sales, HR, Operations with exact targets. 5 daily KPIs, decision triggers, board escalation protocol. Real numbers only.")

    def generate_acquisition_strategy(self):
        return self.ask("Financial takeover strategy: exact acquisition budget with equity plus debt firepower, target criteria, financing structure, 12-month timeline, 90-day integration plan, risk assessment, board approval criteria. Exact numbers.")

    def analyze_employee(self, employee_id):
        preds = {p.employee_id: p for p in self.result.churn_preds}
        p = preds.get(employee_id)
        if not p: return f"Employee {employee_id} not found."
        return self.ask(f"Deep dive on {employee_id} ({p.department}). Churn {p.churn_prob:.0%} ({p.risk_tier}), driver: {p.top_driver}, attendance {p.attendance_rate:.0%} ({p.attendance_flag}), replacement ${p.est_cost_usd:,.0f}. What drives risk, retention plan this week, full cost if they leave, retain or accept?")


    def save_history(self, path="data/chat_history.json"):
        import json as _j
        try:
            with open(path, "w") as f:
                _j.dump([{"role":m.role,"content":m.content} for m in self.history], f)
        except: pass

    def load_history(self, path="data/chat_history.json"):
        import json as _j, os as _o
        try:
            if _o.path.exists(path):
                with open(path) as f:
                    saved = _j.load(f)
                self.history = [ChatMessage(m["role"], m["content"]) for m in saved[-40:]]
                return len(self.history)
        except: pass
        return 0

    def reset(self):
        self.history = []
        import os as _o
        try:
            path = "data/chat_history.json"
            if _o.path.exists(path): _o.remove(path)
        except: pass

    def scenario_narrative(self, scenario_name):
        sc = self.result.scenarios.get(scenario_name)
        if not sc: return "Scenario not found."
        fs = sc.finance_summary
        viable = "Yes" if fs.min_balance > 0 else "NO CASH DEFICIT"
        return self.ask(f"Executive narrative for {scenario_name}: revenue {fs.total_revenue_6m:,.0f}, net {fs.total_net_6m:,.0f}, min balance {fs.min_balance:,.0f}, viable {viable}, risk {sc.risk_score:.1f}/100. Three paragraphs: plain English, exact changes vs base, operating plan if this happens.")
