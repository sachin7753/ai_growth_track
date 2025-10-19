import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import io

# ---------------------------
# CONFIG
# ---------------------------
SERVICE_ACCOUNT_FILE = "service_account.json"
FOLDER_ID = "https://drive.google.com/drive/folders/1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP?usp=drive_link"  # replace with your folder ID

# ---------------------------
# AUTH FUNCTION
# ---------------------------
@st.cache_resource
def drive_auth():
    gauth = GoogleAuth()
    gauth.settings = {'service_config': {'service_account_json': SERVICE_ACCOUNT_FILE}}
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)
    return drive

# ---------------------------
# UPLOAD FUNCTION
# ---------------------------
def upload_pdf_to_drive(pdf_buffer, file_name, folder_id):
    drive = drive_auth()
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': folder_id}]
    })
    file_drive.SetContentBinary(pdf_buffer.read())
    file_drive.Upload()
    return file_drive['alternateLink']

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Upload PDF to Google Drive")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_name = uploaded_file.name
    pdf_buffer = io.BytesIO(uploaded_file.read())

    try:
        st.info("Uploading PDF to Google Drive...")
        link = upload_pdf_to_drive(pdf_buffer, file_name, FOLDER_ID)
        st.success("PDF uploaded successfully!")
        st.markdown(f"[Open PDF in Google Drive]({link})")
    except Exception as e:
        st.error(f"Error uploading PDF: {e}")
