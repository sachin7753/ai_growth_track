import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import streamlit as st
from functools import lru_cache
import joblib
import os
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Child Growth Advisor", page_icon="🧒", layout="wide")

# -------- CONFIG & CONSTANTS --------
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
SCALER_PATH = "scaler.joblib"
PARAMS_PATH = "best_params.json"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# -------- AI MODEL DEFINITION (Must match train.py) --------
class GrowthNet(nn.Module):
    def __init__(self, n_layers=2, n_units=64, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_features = 4
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -------- CACHED FUNCTIONS TO LOAD RESOURCES --------
@st.cache_resource
def load_model_and_scaler(model_path: str, scaler_path: str, params_path: str):
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        
        model = GrowthNet(
            n_layers=best_params['n_layers'],
            n_units=best_params['n_units'],
            dropout_rate=best_params['dropout_rate']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please run train.py first.")
        return None, None

@st.cache_data
def load_ref(path: str, primary_col_regex: str) -> tuple[pd.DataFrame, list[str]]:
    try:
        df = pd.read_excel(path)
        primary_col = next((c for c in df.columns if re.search(primary_col_regex, str(c), re.I)), None)
        if not primary_col: raise ValueError(f"No primary column found in {path}")
        pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
        df = df[[primary_col] + pcols].copy(); df.columns = ["primary"] + pcols
        return df, pcols
    except FileNotFoundError:
        st.error(f"Dataset file not found: '{path}'. Please ensure all .xlsx files are present.")
        return None, None

# -------- CORE CALCULATION LOGIC --------
def interp_curve(ref_df: pd.DataFrame, pcols: list[str], val: float) -> dict[float, float]:
    values = ref_df.iloc[:, 0].values.astype(float)
    if val <= values.min(): row = ref_df.iloc[0]
    elif val >= values.max(): row = ref_df.iloc[-1]
    else:
        idx = np.searchsorted(values, val, side="right"); v0, v1 = values[idx-1], values[idx]
        frac = (val - v0) / (v1 - v0); row0, row1 = ref_df.iloc[idx-1], ref_df.iloc[idx]
        return {float(re.findall(r"\d+",c)[0]): row0[c]+frac*(row1[c]-row0[c]) for c in pcols}
    return {float(re.findall(r"\d+",c)[0]): float(row[c]) for c in pcols}

def est_percentile(value: float, curve: dict[float, float]) -> float:
    pts = sorted(curve.items(), key=lambda item: item[1]); values=[v for p,v in pts]; percs=[p for p,v in pts]
    if value <= values[0]: return percs[0]
    if value >= values[-1]: return percs[-1]
    j = np.searchsorted(values, value, side="right"); v0,v1,p0,p1 = values[j-1],values[j],percs[j-1],percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

def ai_predict(model: GrowthNet, scaler, age_m: int, ht: float, wt: float, sex: str, wfh_p: float, hfa_p: float) -> tuple[str, float]:
    input_data = np.array([[age_m, ht, wt, 1 if sex == "M" else 0]])
    input_scaled = scaler.transform(input_data)
    x = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x); probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx_tensor = torch.max(probabilities, dim=1)
        pred_idx = pred_idx_tensor.item(); confidence_score = confidence.item()

    status = CLASS_LABELS.get(pred_idx, "Unknown")
    bmi = wt / ((ht / 100) ** 2)
    
    if wfh_p < 3: status = "Underweight"
    elif wfh_p > 85: status = "Obese" if bmi >= 30 else "Overweight"
    elif bmi >= 30: status = "Obese"
    elif bmi >= 25: status = "Overweight"
    elif hfa_p < 3 and status in ["Healthy", "Normal Ht"]: status = "Stunted"
    elif status == "Underweight" and wfh_p >= 5 and hfa_p < 5: status = "Stunted"
    return status, confidence_score

def get_ai_recommendations(status: str, age_m: int, wfh_p: float, hfa_p: float, bmi: float) -> list[str]:
    recs = []
    if status in ["Obese", "Overweight"]:
        recs.append(f"**Status: {status}** (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})")
        if bmi >= 35: recs.append("- **Immediate pediatric consultation is critical**.")
        else: recs.append("- A pediatric consultation is strongly recommended.")
        if age_m < 24: recs.append("- **Nutrition:** Avoid sugary drinks/snacks. Prioritize whole foods.")
        else: recs.append("- **Activity:** Encourage ≥60 minutes of active play daily; limit screen time.")
        if hfa_p < 5: recs.append("- **Special Note:** Child is both overweight & stunted. Focus on nutrient-dense foods (not just calorie restriction).")
    elif status == "Underweight":
        recs.append(f"**Status: Underweight** (Weight-for-Height: P{wfh_p:.1f})")
        if wfh_p < 1: recs.append("- **Severe Wasting:** Medical evaluation is urgently needed.")
        else: recs.append("- **Nutrition:** Increase intake of healthy, energy-dense foods (avocado, nuts, etc.).")
        if age_m <= 12: recs.append("- Offer nutrient-rich first foods; do not restrict healthy fats.")
        else: recs.append("- Offer frequent, small meals rich in protein and healthy fats.")
    elif status == "Stunted":
        recs.append(f"**Status: Stunted** (Height-for-Age: P{hfa_p:.1f})")
        recs.append("- **Nutrition:** Focus on a diet rich in **iron, zinc, and vitamin A**.")
        recs.append("- **Food Sources:** Good sources include eggs, dairy, leafy greens, and lentils.")
        recs.append("- **Next Steps:** A medical evaluation for potential supplements is recommended.")
    else:
        recs.append(f"**Status: Healthy Growth Track**")
        recs.append("- **Nutrition:** Continue providing a balanced diet and regular meal times.")
        recs.append("- **Activity:** Encourage at least 60 minutes of varied play daily.")
        recs.append("- **Next Steps:** Maintain regular pediatric check-ups.")
    return recs

def generate_report(age_m: int, ht: float, wt: float, sex: str, model: GrowthNet, scaler) -> dict:
    hfa_ref, hfa_pcols = load_ref(HFA_BOYS_FILE if sex == "M" else HFA_GIRLS_FILE, r'age|day|month')
    wfh_ref, wfh_pcols = load_ref(WFH_BOYS_FILE if sex == "M" else WFH_GIRLS_FILE, r'height|length')
    if hfa_ref is None or wfh_ref is None: return None

    age_d = age_m * DAYS_PER_MONTH
    hfa_curve = interp_curve(hfa_ref, hfa_pcols, age_d)
    hfa_p = est_percentile(ht, hfa_curve)
    wfh_curve = interp_curve(wfh_ref, wfh_pcols, ht)
    wfh_p = est_percentile(wt, wfh_curve)
    ai_status, confidence = ai_predict(model, scaler, age_m, ht, wt, sex, wfh_p, hfa_p)
    bmi = wt / ((ht / 100) ** 2)
    
    who_msgs = []
    if wfh_p < 3: who_msgs.append(f":red[Wasting risk (Weight-for-height at P{wfh_p:.1f}).]")
    elif wfh_p > 85: who_msgs.append(f":red[Possible overweight risk (Weight-for-height at P{wfh_p:.1f}).]")
    else: who_msgs.append(":green[Weight-for-height is in a healthy range.]")
    if hfa_p < 3: who_msgs.append(f":red[Stunting risk (Height-for-age at P{hfa_p:.1f}).]")
    else: who_msgs.append(":green[Height-for-age is in a healthy range.]")
    
    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {"wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi, "who_msgs": who_msgs, "recommendations": recommendations, "ai_status": ai_status, "confidence": confidence}

# -------- PDF GENERATION ----------
def create_pdf_report(child_name: str, age_months: int, report: dict) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(3*cm, height-3*cm, f"Child Growth Report: {child_name}")

    # Basic info
    c.setFont("Helvetica", 12)
    c.drawString(3*cm, height-4*cm, f"Age: {int(age_months)//12}y {int(age_months)%12}m")
    c.drawString(3*cm, height-4.7*cm, f"Height Percentile: P{report['hfa_p']:.1f}")
    c.drawString(3*cm, height-5.4*cm, f"Weight-for-Height Percentile: P{report['wfh_p']:.1f}")
    c.drawString(3*cm, height-6.1*cm, f"BMI: {report['bmi']:.1f}")
    
    # WHO Assessment
    c.setFont("Helvetica-Bold", 14)
    c.drawString(3*cm, height-7*cm, "WHO Assessment:")
    c.setFont("Helvetica", 12)
    y = height-7.7*cm
    for msg in report['who_msgs']:
        c.drawString(4*cm, y, msg.replace(":red[","").replace(":green[","").replace("].",""))
        y -= 0.7*cm
    
    # AI Recommendations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(3*cm, y-0.3*cm, "AI Recommendations:")
    c.setFont("Helvetica", 12)
    y -= 1*cm
    for rec in report['recommendations']:
        c.drawString(4*cm, y, rec.replace("**",""))
        y -= 0.7*cm
        if y < 2*cm:
            c.showPage(); y = height-3*cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------- STREAMLIT USER INTERFACE --------
st.title("🧒 Hybrid AI Child Growth Advisor")
st.markdown("Enter a child's measurements for a growth analysis based on WHO standards and an AI-powered recommendation engine.")

growth_model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, PARAMS_PATH)

with st.sidebar:
    st.header("Child's Measurements")
    child_name = st.text_input("Child's Name", value="John Doe")
    sex_options = {"Male": "M", "Female": "F"}
    sex_label = st.radio("Sex", options=sex_options.keys(), horizontal=True)
    sex = sex_options[sex_label]

    age_months = st.number_input("Age in Months", min_value=0, max_value=60, value=24, step=1)
    height_cm = st.number_input("Height (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1, format="%.1f")
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=40.0, value=12.0, step=0.1, format="%.1f")
    
    generate_button = st.button("Generate Report", type="primary", use_container_width=True)

if generate_button and growth_model and scaler:
    with st.spinner('Analyzing...'):
        report = generate_report(int(age_months), float(height_cm), float(weight_kg), sex, growth_model, scaler)
    if report:
        st.header("Growth Report Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Age", f"{int(age_months) // 12}y {int(age_months) % 12}m")
        col2.metric("Height Percentile", f"P{report['hfa_p']:.1f}")
        col3.metric("Weight/Ht Percentile", f"P{report['wfh_p']:.1f}")
        col4.metric("BMI", f"{report['bmi']:.1f}")
        st.markdown("---")
        
        col_who, col_ai = st.columns(2)
        with col_who:
            st.subheader("📈 WHO Assessment")
            for msg in report['who_msgs']: st.markdown(f"- {msg}")
        with col_ai:
            st.subheader(f"🤖 AI Recommendations")
            st.caption(f"Final Status: **{report['ai_status']}** | Model Confidence: **{report['confidence']:.1%}**")
            for tip in report['recommendations']: st.markdown(f"- {tip}")

        # PDF download
        pdf_buffer = create_pdf_report(child_name, int(age_months), report)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{child_name.replace(' ', '_')}_Growth_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("Could not generate report. Please check that all data files are present and measurements are realistic.")

elif not (growth_model and scaler):
    st.warning("Cannot generate report because AI model or scaler is not loaded. Please ensure required files are in the project folder.")
