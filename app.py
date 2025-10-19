from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import io

SERVICE_ACCOUNT_FILE = "service_account.json"
FOLDER_ID = "https://drive.google.com/drive/folders/1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP?usp=drive_link"

def drive_auth():
    gauth = GoogleAuth()
    gauth.ServiceAuth(service_file=SERVICE_ACCOUNT_FILE)
    drive = GoogleDrive(gauth)
    return drive

def upload_pdf_to_drive(pdf_buffer, file_name, folder_id):
    drive = drive_auth()
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': folder_id}]
    })
    file_drive.SetContentBinary(pdf_buffer.read())
    file_drive.Upload()
    return file_drive['alternateLink']

# Example usage:
import streamlit as st

st.title("Test PDF Upload to Google Drive")

pdf_buffer = io.BytesIO(b"Hello world! This is a test PDF.")  # replace with actual PDF bytes
try:
    link = upload_pdf_to_drive(pdf_buffer, "TestReport.pdf", FOLDER_ID)
    st.success(f"PDF uploaded! [Open Link]({link})")
except Exception as e:
    st.error(f"Error uploading PDF: {e}")
