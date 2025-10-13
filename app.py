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
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from textwrap import wrap

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

# -------- CACHED FUNCTIONS --------
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
        st.error(f"Dataset file not found: '{path}'.")
        return None, None

# -------- CALCULATION LOGIC --------
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
        recs.append(f"Status: {status} (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})")
        if bmi >= 35: recs.append("- Immediate pediatric consultation is critical.")
        else: recs.append("- Pediatric consultation recommended.")
        if age_m < 24: recs.append("- Nutrition: Avoid sugary drinks/snacks. Prioritize whole foods.")
        else: recs.append("- Encourage ≥60 minutes of active play daily; limit screen time.")
        if hfa_p < 5: recs.append("- Special Note: Child is overweight & stunted. Focus on nutrient-dense foods.")
    elif status == "Underweight":
        recs.append(f"Status: Underweight (Weight-for-Height: P{wfh_p:.1f})")
        if wfh_p < 1: recs.append("- Severe Wasting: Medical evaluation urgently needed.")
        else: recs.append("- Nutrition: Increase intake of energy-dense foods.")
        if age_m <= 12: recs.append("- Offer nutrient-rich first foods; do not restrict healthy fats.")
        else: recs.append("- Offer frequent, small meals rich in protein and healthy fats.")
    elif status == "Stunted":
        recs.append(f"Status: Stunted (Height-for-Age: P{hfa_p:.1f})")
        recs.append("- Nutrition: Focus on iron, zinc, and vitamin A rich foods.")
        recs.append("- Food Sources: Eggs, dairy, leafy greens, lentils.")
        recs.append("- Next Steps: Medical evaluation for supplements recommended.")
    else:
        recs.append("Status: Healthy Growth Track")
        recs.append("- Continue providing balanced diet and regular meal times.")
        recs.append("- Encourage at least 60 minutes of varied play daily.")
        recs.append("- Maintain regular pediatric check-ups.")
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
    if wfh_p < 3: who_msgs.append(("Wt-for-height risk (P{:.1f})".format(wfh_p), colors.red))
    elif wfh_p > 85: who_msgs.append(("Possible overweight (P{:.1f})".format(wfh_p), colors.red))
    else: who_msgs.append(("Weight-for-height healthy", colors.green))
    if hfa_p < 3: who_msgs.append(("Stunting risk (P{:.1f})".format(hfa_p), colors.red))
    else: who_msgs.append(("Height-for-age healthy", colors.green))
    
    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {"wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi, "who_msgs": who_msgs, 
            "recommendations": recommendations, "ai_status": ai_status, "confidence": confidence}

# -------- PDF GENERATION --------
def save_report_pdf(child_name, age_m, height, weight, sex, report):
    filename = f"{child_name.replace(' ','_')}_Growth_Report.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height_page = A4
    left_margin = 20 * mm
    top_margin = height_page - 30 * mm
    line_height = 10 * mm

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(left_margin, top_margin, f"Child Growth Report: {child_name}")
    c.setFont("Helvetica", 12)
    c.drawString(left_margin, top_margin - 20, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(left_margin, top_margin - 35, f"Age: {int(age_m)//12}y {int(age_m)%12}m | Sex: {'Male' if sex=='M' else 'Female'}")
    c.drawString(left_margin, top_margin - 50, f"Height: {height} cm | Weight: {weight} kg | BMI: {report['bmi']:.1f}")

    y_pos = top_margin - 70

    # WHO Assessment
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y_pos, "WHO Assessment")
    y_pos -= 12
    c.setFont("Helvetica", 12)
    for msg, color in report['who_msgs']:
        c.setFillColor(color)
        wrapped = wrap(msg, 90)
        for line in wrapped:
            c.drawString(left_margin, y_pos, "- " + line)
            y_pos -= 12
    c.setFillColor(colors.black)
    y_pos -= 10

    # AI Recommendations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y_pos, "AI Recommendations")
    y_pos -= 12
    c.setFont("Helvetica", 12)
    for tip in report['recommendations']:
        wrapped = wrap(tip, 90)
        for line in wrapped:
            c.drawString(left_margin, y_pos, "- " + line)
            y_pos -= 12

    c.save()
    return filename

# -------- STREAMLIT UI --------
st.title("🧒 Hybrid AI Child Growth Advisor")
st.markdown("Enter a child's measurements for a growth analysis based on WHO standards and AI recommendations.")

growth_model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, PARAMS_PATH)

with st.sidebar:
    st.header("Child's Details")
    child_name = st.text_input("Child Name", value="Child Name")
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
            for msg, color in report['who_msgs']:
                st.markdown(f"- {msg}")
        with col_ai:
            st.subheader("🤖 AI Recommendations")
            st.caption(f"Final Status: **{report['ai_status']}** | Model Confidence: **{report['confidence']:.1%}**")
            for tip in report['recommendations']:
                st.markdown(f"- {tip}")

        if st.button("Download PDF Report"):
            pdf_file = save_report_pdf(child_name, int(age_months), height_cm, weight_kg, sex, report)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_file)

elif not (growth_model and scaler):
    st.warning("Cannot generate report because AI model or scaler is not loaded.")
