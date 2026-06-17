import streamlit as st
import os
import tempfile
import time
import requests
import io
import markdown
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google import genai
from google.genai import types

load_dotenv()

# --- Config & Setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_config(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None

GEMINI_API_KEY = get_config("GEMINI_API_KEY")
DRIVE_FOLDER_ID = get_config("DRIVE_FOLDER_ID")
CREDENTIALS_FILE = "credentials.json"

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

# --- Google Drive Helpers ---
@st.cache_resource
def get_drive_service():
    """Initialize and return the Google Drive API service."""
    creds = None
    try:
        # 1. Streamlit Cloud Secrets
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=SCOPES)
        # 2. Local credentials.json file
        elif os.path.exists(CREDENTIALS_FILE):
            creds = service_account.Credentials.from_service_account_file(
                CREDENTIALS_FILE, scopes=SCOPES)
        else:
            return None
            
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Failed to connect to Google Drive: {e}")
        return None

def get_folders_in_drive(service, parent_id):
    """Fetch class folders from the main drive folder."""
    try:
        query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error fetching folders: {e}")
        return []

def get_pdfs_in_folder(service, folder_id):
    """Fetch Subject PDFs from a class folder."""
    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error fetching PDFs: {e}")
        return []

def download_pdf_from_drive(service, file_id, file_name):
    """Download a PDF from Drive to a local temporary file with retries."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file_name)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            # Use 10MB chunk size to avoid SSL timeouts on large files
            downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*10)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            
            with open(temp_path, 'wb') as f:
                f.write(fh.read())
            return temp_path
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download from Drive after {max_retries} attempts: {e}")
            time.sleep(2)
    return temp_path

# --- Gemini Generation ---
def generate_paper(instructions, language, book_path, blueprint_path=None):
    """Upload files to Gemini and generate the exam paper."""
    if not client:
        st.error("Gemini API Client is not initialized. Please check your API key.")
        return None
        
    uploaded_files = []
    try:
        with st.spinner("Uploading book to Gemini (this may take a moment)..."):
            gemini_book = client.files.upload(
                file=book_path, 
                config=types.UploadFileConfig(mime_type="application/pdf")
            )
            uploaded_files.append(gemini_book)
            
            while gemini_book.state == "PROCESSING":
                time.sleep(2)
                gemini_book = client.files.get(name=gemini_book.name)
                
        if blueprint_path:
            with st.spinner("Uploading blueprint to Gemini..."):
                gemini_blueprint = client.files.upload(
                    file=blueprint_path, 
                    config=types.UploadFileConfig(mime_type="application/pdf")
                )
                uploaded_files.append(gemini_blueprint)
                while gemini_blueprint.state == "PROCESSING":
                    time.sleep(2)
                    gemini_blueprint = client.files.get(name=gemini_blueprint.name)
                    
        prompt = f"""
You are an expert exam paper generator. 
Create an exam paper based primarily on the uploaded textbook.
Language: {language}
Instructions/Requirements: {instructions}
"""
        if blueprint_path:
            prompt += "\nA Board Blueprint has also been uploaded. Analyze the blueprint to determine marks distribution, difficulty levels, and structure, and apply this to the exam paper."
            
        prompt += "\nFormat the output clearly, line-by-line, suitable for a professional black & white print. Provide ONLY the exam paper content without any conversational filler."
        
        contents = [f for f in uploaded_files]
        contents.append(prompt)
        
        with st.spinner("Generating exam paper..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents
            )
            
        return response.text
        
    except Exception as e:
        if "429" in str(e):
            st.error("API Limit reached. Please wait 60 seconds.")
        else:
            st.error(f"Error generating paper: {e}")
        return None
    finally:
        # Cleanup uploaded files from Gemini to save storage
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except:
                pass

# --- Main App UI ---
def main():
    st.set_page_config(page_title="AI-Powered Exam Paper Generator", layout="wide")
    
    # Initialize session state for retaining paper
    if "paper_text" not in st.session_state:
        st.session_state.paper_text = None
        
    st.title("📄 AI-Powered Exam Paper Generator")
    
    # Run setup tasks
    service = get_drive_service()
    
    # Verify configurations
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is missing in the .env file.")
        return
        
    if not service:
        st.warning("Google Drive credentials not found or invalid. Please check your credentials.json and .env file.")
        return
        
    if not DRIVE_FOLDER_ID:
        st.error("DRIVE_FOLDER_ID is missing in the .env file.")
        return
        
    # --- Sidebar ---
    st.sidebar.header("1. Select Source Material")
    
    classes = get_folders_in_drive(service, DRIVE_FOLDER_ID)
    if not classes:
        st.sidebar.error("No Class folders found in the specified Drive folder.")
        selected_subject_id = None
        selected_subject_name = None
    else:
        class_names = [c['name'] for c in classes]
        selected_class_name = st.sidebar.selectbox("Select Class", class_names)
        selected_class_id = next(c['id'] for c in classes if c['name'] == selected_class_name)
        
        subjects = get_pdfs_in_folder(service, selected_class_id)
        if not subjects:
            st.sidebar.warning("No subject PDFs found in this class folder.")
            selected_subject_id = None
            selected_subject_name = None
        else:
            subject_names = [s['name'] for s in subjects]
            selected_subject_name = st.sidebar.selectbox("Select Subject (Book PDF)", subject_names)
            selected_subject_id = next(s['id'] for s in subjects if s['name'] == selected_subject_name)
            
    # --- Main Area ---
    st.header("2. Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox("Language", ["English", "Hindi", "Both (Bilingual)"])
        blueprint_file = st.file_uploader("Upload Board Blueprint PDF (Optional)", type=['pdf'])
        
    with col2:
        instructions = st.text_area(
            "Instructions", 
            placeholder="e.g., Chapter 5, 50 marks, Hard difficulty, 5 MCQs, 3 Long Qs",
            height=150
        )
        
    # Process blueprint upload
    blueprint_path = None
    if blueprint_file is not None:
        temp_bp_dir = tempfile.gettempdir()
        blueprint_path = os.path.join(temp_bp_dir, blueprint_file.name)
        with open(blueprint_path, "wb") as f:
            f.write(blueprint_file.getbuffer())

    # Generate button
    if st.button("Generate Paper", type="primary"):
        if not selected_subject_id:
            st.error("Please select a subject PDF from the sidebar.")
            return
        if not instructions:
            st.error("Please provide some instructions for the paper.")
            return
            
        with st.spinner("Downloading textbook from Google Drive..."):
            book_temp_path = download_pdf_from_drive(service, selected_subject_id, selected_subject_name)
            
        paper_text = generate_paper(instructions, language, book_temp_path, blueprint_path)
        
        # Cleanup temporary local files
        if os.path.exists(book_temp_path):
            os.remove(book_temp_path)
        if blueprint_path and os.path.exists(blueprint_path):
            os.remove(blueprint_path)
            
        if paper_text:
            st.session_state.paper_text = paper_text

    # Show paper if it exists in session state
    if st.session_state.paper_text:
        st.markdown("---")
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader("Generated Paper Preview")
        with col2:
            import streamlit.components.v1 as components
            import json
            
            # Convert markdown to HTML for printing
            html_paper = markdown.markdown(st.session_state.paper_text, extensions=['tables'])
            
            # Safely pass HTML to JavaScript using JSON serialization
            js_html_str = json.dumps(html_paper)
            
            # HTML button that creates a hidden iframe and prints ONLY the paper HTML
            components.html(
                f"""
                <div style="text-align: right;">
                    <button onclick="printPaper()" style="padding: 10px 15px; font-size: 14px; font-weight: bold; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        🖨️ Print to PDF
                    </button>
                </div>
                <script>
                function printPaper() {{
                    var htmlContent = {js_html_str};
                    
                    var printIframe = document.createElement('iframe');
                    printIframe.style.position = 'absolute';
                    printIframe.style.width = '0px';
                    printIframe.style.height = '0px';
                    printIframe.style.border = 'none';
                    document.body.appendChild(printIframe);
                    
                    var doc = printIframe.contentWindow.document;
                    doc.open();
                    doc.write(`
                        <html>
                        <head>
                            <title>Exam Paper</title>
                            <style>
                                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; color: black; line-height: 1.6; max-width: 1000px; margin: auto; }}
                                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                                th, td {{ border: 1px solid #000; padding: 8px; text-align: left; }}
                                h1, h2, h3, h4, h5, h6 {{ margin-top: 20px; margin-bottom: 10px; color: black; }}
                                p, li {{ margin-bottom: 10px; color: black; }}
                                @media print {{
                                    body {{ padding: 0; }}
                                }}
                            </style>
                        </head>
                        <body>
                            ${{htmlContent}}
                        </body>
                        </html>
                    `);
                    doc.close();
                    
                    setTimeout(function() {{
                        printIframe.contentWindow.focus();
                        printIframe.contentWindow.print();
                        setTimeout(function() {{
                            document.body.removeChild(printIframe);
                        }}, 1000);
                    }}, 500);
                }}
                </script>
                """,
                height=50
            )

        st.markdown(st.session_state.paper_text)

if __name__ == "__main__":
    main()
