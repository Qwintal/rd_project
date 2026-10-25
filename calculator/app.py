"""Simple premium calculator.  Run from the project folder:  streamlit run calculator/app.py
Fixed simulated world (gamma = 0.5), models trained on honest 10-year records. Simulated data - not a real quote."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st
from calculator import logic as L
from data_generation.generate import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config(os.path.join(ROOT, "config", "default.yaml"))
GAMMA, REGIME = 0.5, "R1"
inr = lambda x: f"₹{x:,.0f}"
NAMES = {"Arm A (traditional)": "Traditional rules (Arm A)", "GLM": "Statistical model (GLM)", "XGBoost": "Machine learning (XGBoost)"}

st.set_page_config(page_title="Premium calculator", layout="wide")


@st.cache_resource
def get_models():
    return L.Models(GAMMA, REGIME)


# ---------------------------------------------------------------- default values and examples
DEFAULT = dict(age=50, sex="M", occ=2, bmi=26.0, sbp=124.0, chol=200.0, hba1c=5.4, fam=False, alc=0, hobby=False,
               sa=5_000_000, smoker=False, pack=0.0, dm=False, htn=False, cvd=False, hide=False)
EXAMPLES = {
    "Typical applicant": DEFAULT,
    "Low-risk 45-year-old woman": dict(DEFAULT, age=45, sex="F", occ=1, bmi=23.0, sbp=115.0, chol=180.0, hba1c=5.2),
    "Hidden smoker: 50-year-old man says he does not smoke": dict(DEFAULT, bmi=27.0, sbp=126.0, chol=205.0, alc=1, smoker=True, pack=22.0, hide=True),
}
for k, v in DEFAULT.items():
    st.session_state.setdefault(f"in_{k}", v)
st.session_state.setdefault("in_term", 10)
st.session_state.setdefault("example", "Typical applicant")


def load_example():
    for k, v in EXAMPLES[st.session_state.example].items():
        st.session_state[f"in_{k}"] = v


def reset():
    st.session_state.example = "Typical applicant"
    load_example()
    st.session_state.in_term = 10


# ---------------------------------------------------------------- inputs
st.title("Premium calculator")
st.caption("Simulated data, not a real quote. Compares the traditional method with a statistical model and a machine learning model.")
top1, top2 = st.columns([4, 1])
top1.selectbox("Example applicant", list(EXAMPLES), key="example", on_change=load_example)
top2.write(""); top2.button("Reset", on_click=reset)

a, b, c = st.columns(3)
age = a.number_input("Age (40 to 60)", 40, 60, key="in_age")
sex = a.selectbox("Sex", ["M", "F"], key="in_sex")
sa = a.number_input("Sum assured (INR)", 500_000, 50_000_000, step=500_000, key="in_sa")
term = a.number_input("Policy term (years)", 5, 30, key="in_term", help="Models are trained on a 10-year window. Other terms are estimated.")
bmi = b.number_input("BMI", 14.0, 55.0, step=0.1, key="in_bmi")
sbp = b.number_input("Blood pressure (systolic, mmHg)", 85.0, 220.0, step=1.0, key="in_sbp")
chol = b.number_input("Cholesterol (mg/dL)", 110.0, 380.0, step=1.0, key="in_chol")
hba1c = b.number_input("Blood sugar (HbA1c, %)", 4.0, 14.0, step=0.1, key="in_hba1c")
smoker = c.checkbox("Smoker", key="in_smoker")
pack = c.number_input("Pack-years (if smoker)", 0.0, 80.0, step=1.0, key="in_pack", disabled=not smoker)
dm = c.checkbox("Diabetes", key="in_dm")
htn = c.checkbox("High blood pressure (diagnosed)", key="in_htn")
cvd = c.checkbox("Heart disease", key="in_cvd")
fam = c.checkbox("Family history", key="in_fam")
with st.expander("More details (optional)"):
    occ = st.selectbox("Occupation class (4 = hazardous)", [1, 2, 3, 4], key="in_occ")
    alc = st.selectbox("Alcohol (0 none, 1 moderate, 2 heavy)", [0, 1, 2], key="in_alc")
    hobby = st.checkbox("Hazardous hobby", key="in_hobby")
hide = bool(st.checkbox("Applicant hides smoking (declares non-smoker)", key="in_hide", disabled=not smoker) and smoker)

# ---------------------------------------------------------------- calculation
term, sa = int(term), int(sa)
obs = dict(age=int(age), sex=sex, occupation_class=int(occ), bmi=float(bmi), sbp=float(sbp), cholesterol=float(chol), hba1c=float(hba1c),
           family_history=int(fam), alcohol=int(alc), hazardous_hobby=int(hobby), sum_assured=sa)
true = dict(smoker=int(smoker), pack_years=float(pack) if smoker else 0.0, diabetes=int(dm), hypertension=int(htn), cvd=int(cvd))
rec = L.build_record(obs, true, {"smoker": 0, "pack_years": 0.0} if hide else None)
M = get_models()


def price_table(view):
    truth_t = L.exact_true_p(rec, CFG, GAMMA, M.A0, M.b, term)
    truth_prem = L.term_premium(truth_t, sa, term)
    rows, gaps = [], {}
    for key, p10 in M.all_p10(rec, view).items():
        p_t = float(L.extrapolate(p10, term))
        prem = L.term_premium(p_t, sa, term)
        gaps[NAMES[key]] = abs(prem / truth_prem - 1)
        rows.append({"Method": NAMES[key], f"{term}-year claim chance": f"{p_t:.2%}", "Yearly premium": inr(prem), "Gap to true price": f"{100 * (prem / truth_prem - 1):+.1f}%"})
    rows.append({"Method": "True price (simulated truth)", f"{term}-year claim chance": f"{truth_t:.2%}", "Yearly premium": inr(truth_prem), "Gap to true price": "-"})
    best = min(gaps, key=gaps.get)
    return pd.DataFrame(rows), best, gaps[best]


# ---------------------------------------------------------------- output
for w in L.age_warnings(int(age), term):
    st.warning(w)
if term != 10:
    st.info(L.EXTRAP_DISCLOSURE + " The true price is exact for every term, so the gap column shows how far the estimates are off.")

if hide:
    left, right = st.columns(2)
    t_decl, _, _ = price_table("declared")
    t_true, _, _ = price_table("true")
    left.subheader("What the insurer sees (smoking hidden)")
    left.dataframe(t_decl, hide_index=True, width="stretch")
    right.subheader("If the applicant told the truth")
    right.dataframe(t_true, hide_index=True, width="stretch")
    saved = (L.term_premium(float(L.extrapolate(M.arm_a(rec, "true")[0], term)), sa, term)
             - L.term_premium(float(L.extrapolate(M.arm_a(rec, "declared")[0], term)), sa, term))
    st.error(f"By hiding smoking, this applicant saves about {inr(saved)} per year with the traditional method.")
else:
    t_decl, best, gap = price_table("declared")
    st.subheader("Yearly premium")
    st.dataframe(t_decl, hide_index=True, width="stretch")
    st.success(f"Closest to the true price: {best} (gap {100 * gap:.1f}%).")

decision = M.arm_a(rec, "declared")[2]
st.write(f"Traditional method decision: **{decision}** (Standard = normal price, Rated = higher price, Decline = refused).")
st.caption("Fixed simulated world (interaction strength 0.5). Models trained on honest 10-year records. Premium = 1.25 x claim chance x sum assured / term.")