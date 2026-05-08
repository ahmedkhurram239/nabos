
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

DEPT_IDEAL_PROFILES = {
    "Engineering":      {"workload_tolerance":8.0,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"high","mgr_importance":"low","key_traits":["technical problem solving","autonomous work","continuous learning"],"desc":"High autonomy, technical depth, high workload tolerance"},
    "Sales":            {"workload_tolerance":7.0,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"medium","mgr_importance":"medium","key_traits":["relationship building","resilience","target-driven"],"desc":"Relationship-driven, target-focused, moderate-high workload"},
    "Marketing":        {"workload_tolerance":6.5,"performance_min":"meets","tenure_pref":"mid","pay_sensitivity":"medium","mgr_importance":"high","key_traits":["creativity","data interpretation","brand thinking"],"desc":"Creative, analytical, manager-dependent culture"},
    "Customer Success": {"workload_tolerance":6.0,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"low","mgr_importance":"high","key_traits":["empathy","patience","conflict resolution"],"desc":"People-first, empathetic, lower pressure environment"},
    "Finance":          {"workload_tolerance":7.0,"performance_min":"meets","tenure_pref":"mid","pay_sensitivity":"high","mgr_importance":"medium","key_traits":["attention to detail","analytical thinking","compliance"],"desc":"Detail-oriented, structured, deadline-driven"},
    "HR":               {"workload_tolerance":5.5,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"low","mgr_importance":"high","key_traits":["people empathy","confidentiality","conflict mediation"],"desc":"People-centric, low workload, high interpersonal skill"},
    "Supply Chain":     {"workload_tolerance":7.5,"performance_min":"meets","tenure_pref":"mid","pay_sensitivity":"medium","mgr_importance":"medium","key_traits":["process thinking","vendor management","logistics"],"desc":"Process-driven, structured, moderate-high workload"},
    "Manufacturing":    {"workload_tolerance":8.5,"performance_min":"meets","tenure_pref":"long","pay_sensitivity":"medium","mgr_importance":"high","key_traits":["physical stamina","safety compliance","process discipline"],"desc":"High workload, safety-critical, long tenure preferred"},
    "Product":          {"workload_tolerance":7.5,"performance_min":"exceeds","tenure_pref":"mid","pay_sensitivity":"high","mgr_importance":"low","key_traits":["user empathy","strategic thinking","cross-functional influence"],"desc":"High autonomy, strategic, above-average performance needed"},
    "IT":               {"workload_tolerance":7.0,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"high","mgr_importance":"low","key_traits":["technical troubleshooting","systems thinking","documentation"],"desc":"Technical, independent, market-rate sensitive"},
    "R&D":              {"workload_tolerance":6.0,"performance_min":"exceeds","tenure_pref":"long","pay_sensitivity":"medium","mgr_importance":"low","key_traits":["curiosity","scientific rigour","long-term thinking"],"desc":"Deep expertise, high performance bar, long tenure valued"},
    "G&A":              {"workload_tolerance":5.0,"performance_min":"meets","tenure_pref":"any","pay_sensitivity":"low","mgr_importance":"medium","key_traits":["organisation","compliance","administrative precision"],"desc":"Low workload, structured, administrative focus"},
}

PERF_RANK = {"below":0,"meets":1,"exceeds":2}


@dataclass
class DeptFitScore:
    department: str
    fit_score: float
    rank: int
    reasons: List[str]
    gaps: List[str]
    transfer_benefit: str
    desc: str


@dataclass
class EmployeeMatchResult:
    employee_id: str
    current_dept: str
    current_fit: float
    current_rank: int
    recommended_dept: str
    recommended_fit: float
    improvement_est: float
    all_scores: List[DeptFitScore]
    transfer_plan: str
    should_transfer: bool


def _compute_fit(emp, dept):
    profile  = DEPT_IDEAL_PROFILES.get(dept, {})
    score    = 50.0
    reasons  = []
    gaps     = []

    workload   = float(emp.get("workload_score", 5.0))
    perf       = str(emp.get("performance", "meets"))
    tenure     = int(emp.get("tenure_months", 24))
    pay_parity = float(emp.get("pay_parity", 1.0))
    mgr_score  = float(emp.get("manager_score", 7.0))
    att_rate   = float(emp.get("att_rate", 0.92))

    # Workload match
    ideal_wl = profile.get("workload_tolerance", 6.5)
    wl_diff  = workload - ideal_wl
    if abs(wl_diff) <= 1.0:
        score += 15; reasons.append(f"Workload comfort ({workload:.1f}) matches dept ideal ({ideal_wl:.1f})")
    elif wl_diff > 1.0:
        score += 8; reasons.append("Handles higher workload than dept requires — likely to thrive")
    else:
        score -= min(abs(wl_diff)*5, 20); gaps.append(f"May struggle with dept workload ({ideal_wl:.1f}) vs current {workload:.1f}")

    # Performance
    perf_min = profile.get("performance_min","meets")
    if PERF_RANK.get(perf,1) >= PERF_RANK.get(perf_min,1):
        score += 15; reasons.append(f"Performance ({perf}) meets dept requirement ({perf_min})")
    else:
        score -= 20; gaps.append(f"Performance ({perf}) below dept minimum ({perf_min})")

    # Tenure
    tenure_pref = profile.get("tenure_pref","any")
    if tenure_pref == "any":
        score += 8
    elif tenure_pref == "mid":
        if 24 <= tenure <= 96: score += 10; reasons.append(f"Tenure ({tenure}mo) ideal for this dept")
        elif tenure < 24: score -= 5; gaps.append("Dept prefers 2-8 years experience")
        else: score += 5
    elif tenure_pref == "long":
        if tenure >= 60: score += 10; reasons.append(f"Long tenure ({tenure}mo) valued here")
        else: score -= 8; gaps.append(f"Dept prefers 5+ year tenure — currently {tenure//12}yr {tenure%12}mo")

    # Pay parity
    pay_sens = profile.get("pay_sensitivity","medium")
    if pay_parity >= 1.0:
        score += 8; reasons.append("Compensation at/above market — low retention risk")
    elif pay_parity >= 0.90:
        score += 3
        if pay_sens == "high": gaps.append("Pay slightly below market — this dept is pay-sensitive")
    else:
        score -= (10 if pay_sens=="high" else 5); gaps.append(f"Pay {(1-pay_parity):.0%} below market — high flight risk here")

    # Manager score
    mgr_imp = profile.get("mgr_importance","medium")
    if mgr_imp == "high" and mgr_score >= 7.5:
        score += 8; reasons.append("Strong manager relationship — good for manager-dependent culture")
    elif mgr_imp == "low":
        score += 5
    elif mgr_score < 6.0 and mgr_imp == "high":
        score -= 8; gaps.append("Low manager score — this dept has high manager dependency")

    # Attendance
    if att_rate >= 0.92:
        score += 5; reasons.append(f"Strong attendance ({att_rate:.0%})")
    elif att_rate < 0.85:
        score -= 7; gaps.append(f"Attendance ({att_rate:.0%}) below threshold — address before transfer")

    return float(np.clip(score,0,100)), reasons, gaps


class SkillMatchEngine:

    def match_employee(self, emp, all_depts):
        current_dept = str(emp.get("department","Unknown"))
        eid          = str(emp.get("employee_id","?"))
        churn_prob   = float(emp.get("churn_prob",0.35))

        scores = []
        for dept in all_depts:
            fit, reasons, gaps = _compute_fit(emp, dept)
            profile = DEPT_IDEAL_PROFILES.get(dept,{})
            scores.append(DeptFitScore(
                department=dept, fit_score=round(fit,1), rank=0,
                reasons=reasons, gaps=gaps,
                transfer_benefit="High" if fit>=70 else "Medium" if fit>=50 else "Low",
                desc=profile.get("desc","")))

        scores.sort(key=lambda s:s.fit_score, reverse=True)
        for i,s in enumerate(scores): s.rank = i+1

        cur   = next((s for s in scores if s.department==current_dept), None)
        cur_f = cur.fit_score if cur else 50.0
        cur_r = cur.rank if cur else len(scores)
        best  = scores[0]

        should = best.department!=current_dept and best.fit_score-cur_f>=10 and churn_prob>=0.35
        improv = min((best.fit_score-cur_f)*0.003, 0.25) if should else 0.0

        if should:
            plan = (f"Transfer {eid} from {current_dept} (fit:{cur_f:.0f}/100) to {best.department} "
                    f"(fit:{best.fit_score:.0f}/100). Est. churn reduction: {improv:.0%}. "
                    f"Steps: (1) 1-on-1 with current manager, (2) Shadow {best.department} for 2 weeks, "
                    f"(3) Formal transfer after 30 days. "
                    f"Gaps: {'; '.join(best.gaps[:2]) if best.gaps else 'None significant'}.")
        else:
            plan = (f"{eid} is well-placed in {current_dept} (fit:{cur_f:.0f}/100). "
                    f"Focus: {'; '.join((cur.gaps if cur else [])[:2]) or 'no major gaps'}.")

        return EmployeeMatchResult(
            employee_id=eid, current_dept=current_dept,
            current_fit=round(cur_f,1), current_rank=cur_r,
            recommended_dept=best.department, recommended_fit=round(best.fit_score,1),
            improvement_est=round(improv,3), all_scores=scores,
            transfer_plan=plan, should_transfer=should)

    def match_all(self, workforce):
        depts = list(DEPT_IDEAL_PROFILES.keys())
        return [self.match_employee(emp, depts) for _,emp in workforce.iterrows()]

    def transfer_candidates(self, results, min_improvement=8.0):
        return sorted(
            [r for r in results if r.should_transfer and r.recommended_fit-r.current_fit>=min_improvement],
            key=lambda r: r.recommended_fit-r.current_fit, reverse=True)

    def dept_fit_summary(self, results):
        rows = [{"department":r.current_dept,"current_fit":r.current_fit,"should_transfer":int(r.should_transfer)} for r in results]
        df   = pd.DataFrame(rows)
        if df.empty: return df
        return (df.groupby("department")
                .agg(employees=("current_fit","count"),avg_fit=("current_fit","mean"),transfer_candidates=("should_transfer","sum"))
                .round(1).reset_index().sort_values("avg_fit"))
