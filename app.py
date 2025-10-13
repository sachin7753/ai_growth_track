# ------------------- IMPORTS -------------------
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
from reportlab.lib import colors
import matplotlib.pyplot as plt

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

# ------------------- CORE CALCULATION -------------------
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

# ------------------- AI RECOMMENDATIONS -------------------
def get_ai_recommendations(status: str, age_m: int, wfh_p: float, hfa_p: float, bmi: float) -> list[str]:
    recs = []
    recs.append(f"**Status: {status}** (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})")
    
    if status in ["Obese", "Overweight"]:
        recs.append("- Encourage balanced meals with vegetables, fruits, and lean proteins.")
        recs.append("- Avoid sugary drinks and high-calorie snacks.")
        recs.append("- Ensure at least 60 minutes of physical activity daily.")
        recs.append("- Schedule pediatric consultation if BMI > 30 or rapid weight gain.")
    elif status == "Underweight":
        recs.append("- Increase intake of nutrient-dense foods such as nuts, dairy, and eggs.")
        recs.append("- Frequent small meals may help gain weight.")
        recs.append("- Monitor growth monthly to track improvement.")
    elif status == "Stunted":
        recs.append("- Focus on iron, zinc, vitamin A-rich foods (eggs, greens, dairy).")
        recs.append("- Ensure adequate protein intake.")
        recs.append("- Pediatric evaluation for possible supplements recommended.")
    else:
        recs.append("- Continue balanced diet and regular meal times.")
        recs.append("- Encourage 60+ minutes of daily active play.")
        recs.append("- Regular pediatric check-ups are important.")
    return recs

# ------------------- REPORT GENERATION -------------------
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
    if wfh_p < 3: who_msgs.append((f"Wasting risk (Weight-for-height at P{wfh_p:.1f})", colors.red))
    elif wfh_p > 85: who_msgs.append((f"Possible overweight risk (Weight-for-height at P{wfh_p:.1f})", colors.red))
    else: who_msgs.append(("Weight-for-height is in a healthy range.", colors.green))
    if hfa_p < 3: who_msgs.append((f"Stunting risk (Height-for-age at P{hfa_p:.1f})", colors.red))
    else: who_msgs.append(("Height-for-age is in a healthy range.", colors.green))
    
    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {"wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi, "who_msgs": who_msgs, "recommendations": recommendations, "ai_status": ai_status, "confidence": confidence, "hfa_curve": hfa_curve, "wfh_curve": wfh_curve, "age_d": age_d, "ht": ht, "wt": wt}
