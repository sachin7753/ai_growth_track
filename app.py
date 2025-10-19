import io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

SERVICE_ACCOUNT_JSON = "service_account.json"  # path to your JSON
FOLDER_ID = "https://drive.google.com/drive/folders/1u9uDdhJ0Q8GolChkIk6-otOi6pYggTxP?usp=drive_link"  # put your folder ID here

def drive_auth():
    gauth = GoogleAuth()
    gauth.ServiceAuth()  # authenticate using service account JSON
    gauth.LoadServiceConfigSettings()  # loads default service settings
    gauth.LoadServiceConfigFile(SERVICE_ACCOUNT_JSON)
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

# Example usage
pdf_buffer = io.BytesIO(b"Hello, this is a test PDF content")  # replace with real PDF
link = upload_pdf_to_drive(pdf_buffer, "TestReport.pdf", FOLDER_ID)
print("PDF uploaded! Link:", link)
