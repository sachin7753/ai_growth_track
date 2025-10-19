import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from io import BytesIO
from reportlab.pdfgen import canvas

# --- Streamlit App ---
st.title("PDF Upload to Google Drive Demo")

child_name = st.text_input("Enter Child Name", "John Doe")
folder_id = st.text_input("Enter Google Drive Folder ID","https://drive.google.com/drive/folders/1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP?usp=drive_link")

if st.button("Generate & Upload PDF"):
    if not child_name or not folder_id:
        st.error("Please provide both child name and folder ID.")
    else:
        # --- Generate PDF in memory ---
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.drawString(100, 750, f"Growth Report for {child_name}")
        c.drawString(100, 730, "Height: 120 cm")
        c.drawString(100, 710, "Weight: 25 kg")
        c.save()
        pdf_buffer.seek(0)

        try:
            # --- Authenticate PyDrive2 with Service Account ---
            service_account_info = st.secrets["google_service_account"]

            gauth = GoogleAuth()
            gauth.settings = {
                "client_config_backend": "service",
                "service_config": service_account_info
            }
            gauth.ServiceAuth()
            drive = GoogleDrive(gauth)

            # --- Upload PDF ---
            file_name = f"{child_name.replace(' ', '_')}_Growth_Report.pdf"
            file_drive = drive.CreateFile({
                "title": file_name,
                "parents": [{"id": folder_id}]
            })
            file_drive.SetContentBinary(pdf_buffer.read())
            file_drive.Upload()

            # --- Make it shareable ---
            file_drive.InsertPermission({
                "type": "anyone",
                "value": "anyone",
                "role": "reader"
            })
            drive_link = file_drive['alternateLink']
            st.success("PDF uploaded successfully!")
            st.write(f"[Open PDF]({drive_link})")
        except Exception as e:
            st.error(f"Error uploading PDF: {e}")
