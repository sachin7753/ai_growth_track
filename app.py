# ------------------- IMPORTS -------------------
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import streamlit as st
from functools import lru_cache
import joblib
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
import matplotlib.pyplot as plt
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Child Growth Advisor", page_icon="🧒", layout="wide")

# ------------------- CONFIG & CONSTANTS -------------------
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
SCALER_PATH = "scaler.joblib"
PARAMS_PATH = "best_params.json"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# ------------------- GOOGLE DRIVE SETUP -------------------
SERVICE_ACCOUNT_FILE = "service_account.json"  # your JSON key
DRIVE_FOLDER_ID = "1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP"

scope = ["https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
gauth = GoogleAuth()
gauth.credentials = credentials
drive = GoogleDrive(gauth)

def upload_pdf_to_drive(pdf_buffer, filename):
    pdf_buffer.seek(0)
    file_drive = drive.CreateFile({
        "title": filename,
        "parents": [{"id": DRIVE_FOLDER_ID}]
    })
    file_drive.SetContentBinary(pdf_buffer.read())
    file_drive.Upload()
    # Make file public
    file_drive.InsertPermission({
        "type": "anyone",
        "value": "anyone",
        "role": "reader"
    })
    return file_drive['alternateLink']

# ------------------- AI MODEL -------------------
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

# ------------------- LOAD MODEL & SCALER -------------------
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
        st.error(f"Required file not found: {e}. Please ensure all files are present.")
        return None, None

@st.cache_data
def load_ref(path: str, primary_col_regex: str):
    try:
        df = pd.read_excel(path)
        primary_col = next((c for c in df.columns if re.search(primary_col_regex, str(c), re.I)), None)
        if not primary_col: raise ValueError(f"No primary column found in {path}")
        pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
        df = df[[primary_col] + pcols].copy(); df.columns = ["primary"] + pcols
        return df, pcols
    except FileNotFoundError:
        st.error(f"Dataset file not found: '{path}'")
        return None, None

# ------------------- CORE CALCULATION -------------------
def interp_curve(ref_df: pd.DataFrame, pcols: list, val: float):
    values = ref_df.iloc[:, 0].values.astype(float)
    if val <= values.min(): row = ref_df.iloc[0]
    elif val >= values.max(): row = ref_df.iloc[-1]
    else:
        idx = np.searchsorted(values, val, side="right"); v0, v1 = values[idx-1], values[idx]
        frac = (val - v0) / (v1 - v0); row0, row1 = ref_df.iloc[idx-1], ref_df.iloc[idx]
        return {float(re.findall(r"\d+",c)[0]): row0[c]+frac*(row1[c]-row0[c]) for c in pcols}
    return {float(re.findall(r"\d+",c)[0]): float(row[c]) for c in pcols}

def est_percentile(value: float, curve: dict):
    pts = sorted(curve.items(), key=lambda item: item[1]); values=[v for p,v in pts]; percs=[p for p,v in pts]
    if value <= values[0]: return percs[0]
    if value >= values[-1]: return percs[-1]
    j = np.searchsorted(values, value, side="right"); v0,v1,p0,p1 = values[j-1],values[j],percs[j-1],percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

def ai_predict(model, scaler, age_m: int, ht: float, wt: float, sex: str, wfh_p: float, hfa_p: float):
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

# ------------------- AI RECOMMENDATIONS -------------------
def get_ai_recommendations(status: str, age_m: int, wfh_p: float, hfa_p: float, bmi: float):
    recs = [f"**Status: {status}** (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})"]
    if status in ["Obese", "Overweight"]:
        recs += ["- Balanced meals: vegetables, fruits, lean proteins.",
                 "- Avoid sugary drinks & high-calorie snacks.",
                 "- 60 min physical activity daily.",
                 "- Consult pediatrician if BMI>30."]
    elif status == "Underweight":
        recs += ["- Increase nutrient-dense foods (nuts, dairy, eggs).",
                 "- Frequent small meals.",
                 "- Monitor growth monthly."]
    elif status == "Stunted":
        recs += ["- Focus on iron, zinc, vitamin A-rich foods.",
                 "- Ensure adequate protein intake.",
                 "- Pediatric evaluation recommended."]
    else:
        recs += ["- Continue balanced diet.",
                 "- Encourage daily active play 60+ min.",
                 "- Regular pediatric check-ups."]
    return recs

# ------------------- REPORT GENERATION -------------------
def generate_report(age_m: int, ht: float, wt: float, sex: str, model, scaler):
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
    if wfh_p < 3: who_msgs.append((f"Wasting risk (Wt-for-Ht P{wfh_p:.1f})", colors.red))
    elif wfh_p > 85: who_msgs.append((f"Possible overweight risk (Wt-for-Ht P{wfh_p:.1f})", colors.red))
    else: who_msgs.append(("Weight-for-height healthy", colors.green))
    if hfa_p < 3: who_msgs.append((f"Stunting risk (Ht-for-age P{hfa_p:.1f})", colors.red))
    else: who_msgs.append(("Height-for-age healthy", colors.green))
    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {"wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi, "who_msgs": who_msgs,
            "recommendations": recommendations, "ai_status": ai_status,
            "confidence": confidence, "hfa_curve": hfa_curve, "wfh_curve": wfh_curve,
            "age_d": age_d, "ht": ht, "wt": wt}

# ------------------- PDF REPORT -------------------
def create_pdf_report(child_name: str, age_months: int, report: dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(3*cm, height-3*cm, f"Child Growth Report: {child_name}")
    c.setFont("Helvetica", 12)
    c.drawString(3*cm, height-4*cm, f"Age: {int(age_months)//12}y {int(age_months)%12}m")
    c.drawString(3*cm, height-4.7*cm, f"Height Percentile: P{report['hfa_p']:.1f}")
    c.drawString(3*cm, height-5.4*cm, f"Weight-for-Height Percentile: P{report['wfh_p']:.1f}")
    c.drawString(3*cm, height-6.1*cm, f"BMI: {report['bmi']:.1f}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(3*cm, height-7*cm, "WHO Assessment:")
    c.setFont("Helvetica", 12)
    y = height-7.7*cm
    for msg, color in report['who_msgs']:
        c.setFillColor(color)
        c.drawString(4*cm, y, msg)
        y -= 0.7*cm
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(3*cm, y-0.3*cm, "AI Recommendations:")
    c.setFont("Helvetica", 12)
    y -= 1*cm
    for rec in report['recommendations']:
        c.drawString(4*cm, y, rec.replace("**",""))
        y -= 0.7*cm
        if y < 5*cm:
            c.showPage(); y = height-3*cm

    # Plot charts
    plt.figure(figsize=(6,4))
    hfa_x, hfa_y = list(report['hfa_curve'].keys()), list(report['hfa_curve'].values())
    plt.plot(hfa_x, hfa_y, label='Height-for-age', color='green')
    plt.scatter([report['ht']], [report['ht']], color='blue', label='Child Height')
    plt.xlabel("Percentile"); plt.ylabel("Height (cm)"); plt.title("Height-for-Age Percentile"); plt.legend(); plt.tight_layout()
    plt.savefig("hfa_chart.png"); plt.close()

    plt.figure(figsize=(6,4))
    wfh_x, wfh_y = list(report['wfh_curve'].keys()), list(report['wfh_curve'].values())
    plt.plot(wfh_x, wfh_y, label='Weight-for-height', color='orange')
    plt.scatter([report['ht']], [report['wt']], color='red', label='Child Weight')
    plt.xlabel("Percentile"); plt.ylabel("Weight (kg)"); plt.title("Weight-for-Height Percentile"); plt.legend(); plt.tight_layout()
    plt.savefig("wfh_chart.png"); plt.close()

    c.showPage()
    c.drawImage("hfa_chart.png", 2*cm, height/2, width=16*cm, height=9*cm)
    c.drawImage("wfh_chart.png", 2*cm, 2*cm, width=16*cm, height=9*cm)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------- STREAMLIT UI -------------------
st.title("🧒 Hybrid AI Child Growth Advisor with PDF Charts")
growth_model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, PARAMS_PATH)

with st.sidebar:
    st.header("Child's Measurements")
    child_name = st.text_input("Child's Name", value="John Doe")
    sex_options = {"Male": "M", "Female": "F"}
    sex_label = st.radio("Sex", options=sex_options.keys(), horizontal=True)
    sex = sex_options[sex_label]
    age_months = st.number_input("Age in Months", min_value=0, max_value=60, value=24, step=1)
    height_cm = st.number_input("Height (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)
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
            st.subheader(f"🤖 AI Recommendations")
            st.caption(f"Final Status: **{report['ai_status']}** | Model Confidence: **{report['confidence']:.1%}**")
            for tip in report['recommendations']: st.markdown(f"- {tip}")

        pdf_buffer = create_pdf_report(child_name, int(age_months), report)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{child_name.replace(' ', '_')}_Growth_Report.pdf",
            mime="application/pdf"
        )

        drive_link = upload_pdf_to_drive(pdf_buffer, f"{child_name.replace(' ', '_')}_Growth_Report.pdf")
        st.success(f"PDF uploaded to Google Drive! [View PDF]({drive_link})")

elif not (growth_model and scaler):
    st.warning("Cannot generate report because AI model or scaler is missing.")
