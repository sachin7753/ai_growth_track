# ------------------- IMPORTS -------------------
import streamlit as st
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import re
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Child Growth Advisor", page_icon="🧒", layout="wide")

# ------------------- CONSTANTS -------------------
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
SCALER_PATH = "scaler.joblib"
PARAMS_PATH = "best_params.json"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

GOOGLE_DRIVE_FOLDER_ID = "1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP"  # Your folder ID

# ------------------- AI MODEL -------------------
class GrowthNet(nn.Module):
    def __init__(self, n_layers=2, n_units=64, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_features = 4
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# ------------------- LOAD MODEL & SCALER -------------------
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path, params_path):
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

# ------------------- DRIVE AUTH -------------------
@st.cache_resource
def drive_auth():
    service_account_info = st.secrets["google_service_account"]
    gauth = GoogleAuth()
    gauth.settings['client_config_backend'] = 'service'
    gauth.settings['client_config'] = service_account_info
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

def upload_pdf_to_drive(pdf_buffer: io.BytesIO, filename: str, folder_id: str) -> str:
    drive = drive_auth()
    pdf_buffer.seek(0)
    file_drive = drive.CreateFile({
        "title": filename,
        "parents": [{"id": folder_id}]
    })
    file_drive.SetContentBinary(pdf_buffer.read())
    file_drive.Upload()
    file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
    return file_drive['alternateLink']

# ------------------- PDF GENERATION -------------------
def create_pdf_report(child_name: str, age_months: int) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(3*cm, height-3*cm, f"Child Growth Report: {child_name}")
    c.setFont("Helvetica", 12)
    c.drawString(3*cm, height-4*cm, f"Age: {age_months // 12}y {age_months % 12}m")
    c.drawString(3*cm, height-4.7*cm, "Sample Data Here")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------- STREAMLIT INTERFACE -------------------
st.title("🧒 Child Growth PDF to Google Drive")

child_name = st.text_input("Child's Name", value="John Doe")
age_months = st.number_input("Age in Months", min_value=0, max_value=60, value=24, step=1)

if st.button("Generate & Upload PDF"):
    with st.spinner("Generating PDF and uploading to Google Drive..."):
        pdf_buffer = create_pdf_report(child_name, age_months)
        drive_link = upload_pdf_to_drive(pdf_buffer, f"{child_name.replace(' ', '_')}_Growth_Report.pdf", GOOGLE_DRIVE_FOLDER_ID)
    st.success(f"✅ PDF uploaded successfully! [Open PDF]({drive_link})")
