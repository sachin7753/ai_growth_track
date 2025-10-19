import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# =======================
# GOOGLE DRIVE AUTH
# =======================
@st.cache_resource
def drive_auth():
    gauth = GoogleAuth()
    service_account_info = st.secrets["google_service_account"]
    gauth.settings = {"service_config": service_account_info}
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)
    return drive

# =======================
# PDF UPLOAD FUNCTION
# =======================
def upload_pdf_to_drive(pdf_buffer, filename, folder_id=None):
    drive = drive_auth()
    file_drive = drive.CreateFile(
        {"title": filename, "parents": [{"id": folder_id}]} if folder_id else {"title": filename}
    )
    pdf_buffer.seek(0)
    file_drive.SetContentBinary(pdf_buffer.read())
    file_drive.Upload()
    file_drive.InsertPermission({
        'type': 'anyone',
        'value': 'anyone',
        'role': 'reader'
    })
    return file_drive['alternateLink']

# =======================
# PDF GENERATION FUNCTION
# =======================
def generate_pdf(child_name, age, weight, height):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, f"Growth Report: {child_name}")
    c.setFont("Helvetica", 12)
    c.drawString(100, 760, f"Age: {age} years")
    c.drawString(100, 740, f"Weight: {weight} kg")
    c.drawString(100, 720, f"Height: {height} cm")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# =======================
# STREAMLIT UI
# =======================
st.title("Child Nutrition Growth Report")

child_name = st.text_input("Child Name")
age = st.number_input("Age (years)", min_value=0, max_value=18)
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=100.0, format="%.2f")
height = st.number_input("Height (cm)", min_value=0.0, max_value=200.0, format="%.2f")
folder_id = st.text_input("Google Drive Folder ID (optional)")

if st.button("Generate & Upload PDF"):
    if not child_name:
        st.error("Please enter the child's name.")
    else:
        pdf_buffer = generate_pdf(child_name, age, weight, height)
        try:
            drive_link = upload_pdf_to_drive(
                pdf_buffer, f"{child_name.replace(' ', '_')}_Growth_Report.pdf", folder_id or None
            )
            st.success("PDF uploaded successfully!")
            st.markdown(f"[Click here to view PDF]({drive_link})")
        except Exception as e:
            st.error(f"Error uploading PDF: {e}")
