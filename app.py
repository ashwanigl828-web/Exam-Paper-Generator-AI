import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import time
import requests
import io
import json
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
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None
else:
    client = None

# --- Helper: Retry Logic ---
def execute_with_retry(func, *args, **kwargs):
    """Executes a function with a try-except wrapper and one retry with exponential backoff for API errors."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_str = str(e).lower()
        # Checking for common transient errors
        if any(keyword in error_str for keyword in ["429", "connection", "quota", "timeout", "unavailable", "internal", "error"]):
            time.sleep(2)  # Wait 2 seconds before retry (exponential backoff approach)
            try:
                return func(*args, **kwargs)
            except Exception as retry_e:
                raise Exception(f"Operation failed after retry. Please check your connection or try again later.")
        else:
            raise Exception(f"An unexpected error occurred: {e}")

# --- Google Drive Helpers ---
@st.cache_resource
def get_drive_service():
    """Initialize and return the Google Drive API service."""
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=SCOPES)
        elif os.path.exists(CREDENTIALS_FILE):
            creds = service_account.Credentials.from_service_account_file(
                CREDENTIALS_FILE, scopes=SCOPES)
        else:
            return None
            
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Friendly Warning: Failed to connect to Google Drive. Please ensure credentials are correct.")
        return None

def _fetch_drive_files(service, query):
    results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
    return results.get('files', [])

def get_folders_in_drive(service, parent_id):
    """Fetch class folders from the main drive folder."""
    try:
        query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        return execute_with_retry(_fetch_drive_files, service, query)
    except Exception as e:
        st.error(f"Friendly Warning: Could not fetch folders from Drive. {e}")
        return []

def get_pdfs_in_folder(service, folder_id):
    """Fetch Subject PDFs from a class folder."""
    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        return execute_with_retry(_fetch_drive_files, service, query)
    except Exception as e:
        st.error(f"Friendly Warning: Could not fetch PDFs from Drive. {e}")
        return []

def download_pdf_from_drive(service, file_id, file_name):
    """Download a PDF from Drive to a local temporary file with retries."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file_name)
    
    def _download():
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*10)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(fh.read())
        return temp_path

    try:
        return execute_with_retry(_download)
    except Exception as e:
        st.error(f"Friendly Warning: Failed to download the selected file. It might be inaccessible. {e}")
        return None

# --- Gemini Generation ---
def process_with_gemini(task_mode, instructions, language, book_path, blueprint_path=None):
    """Upload files to Gemini and generate content based on the task mode."""
    if not client:
        st.error("Gemini API Client is not initialized. Please check your API key.")
        return None
        
    uploaded_files = []
    
    def _upload_file(path, mime_type="application/pdf"):
        gemini_file = client.files.upload(
            file=path, 
            config=types.UploadFileConfig(mime_type=mime_type)
        )
        while gemini_file.state == "PROCESSING":
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)
        return gemini_file

    def _generate_content(model, contents):
        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        return response.text

    try:
        with st.spinner("Uploading document to Gemini (this may take a moment)..."):
            gemini_book = execute_with_retry(_upload_file, book_path)
            uploaded_files.append(gemini_book)
            
        if blueprint_path and task_mode == "Generate Exam Paper":
            with st.spinner("Uploading blueprint to Gemini..."):
                gemini_blueprint = execute_with_retry(_upload_file, blueprint_path)
                uploaded_files.append(gemini_blueprint)
                
        # Determine Prompt based on Task Mode
        if task_mode == "Generate Exam Paper":
            prompt = f"""
You are an expert exam paper generator. 
Create an exam paper based primarily on the uploaded textbook.
Language: {language}
Instructions/Requirements: {instructions}
"""
            if blueprint_path:
                prompt += "\nA Board Blueprint has also been uploaded. Analyze the blueprint to determine marks distribution, difficulty levels, and structure, and apply this to the exam paper."
            prompt += "\nFormat the output clearly, line-by-line, suitable for a professional black & white print. Provide ONLY the exam paper content without any conversational filler."
            
        elif task_mode == "Summarize Chapter":
            prompt = f"""
You are an expert teacher and summarizer.
Create a concise, easy-to-understand explanation or summary based on the uploaded document.
Instructions/Requirements: {instructions}
Format the output nicely with headings and bullet points where appropriate. Ensure it is very easy to read and understand.
"""
        elif task_mode == "Ask a Question":
            prompt = f"""
You are a helpful teaching assistant.
Search the uploaded document to find the exact answer to the user's question.
Question/Instructions: {instructions}
Provide a clear, direct answer based ONLY on the provided document context. If the answer is not in the document, say so politely.
"""
        else:
            prompt = instructions

        contents = [f for f in uploaded_files]
        contents.append(prompt)
        
        with st.spinner(f"Processing '{task_mode}' with Gemini..."):
            # Forced to use gemini-1.5-flash for performance optimization
            result_text = execute_with_retry(_generate_content, 'gemini-1.5-flash', contents)
            
        return result_text
        
    except Exception as e:
        st.error(f"Friendly Warning: Failed to process your request with AI. {e}")
        return None
    finally:
        # Cleanup uploaded files from Gemini to save storage (Efficient Memory Management)
        for f in uploaded_files:
            try:
                execute_with_retry(client.files.delete, name=f.name)
            except:
                pass
        
        # Cleanup local temporary files immediately (Efficient Memory Management)
        try:
            if book_path and os.path.exists(book_path):
                os.remove(book_path)
            if blueprint_path and os.path.exists(blueprint_path):
                os.remove(blueprint_path)
        except Exception:
            pass

# --- Main App UI ---
def main():
    st.set_page_config(page_title="Smart School Learning Suite", layout="wide", page_icon="🎓")
    
    # Initialize session state
    if "result_text" not in st.session_state:
        st.session_state.result_text = None
        
    st.title("🎓 Smart School Learning Suite")
    st.markdown("---")
    
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
    st.sidebar.header("1. Select Task Mode")
    task_mode = st.sidebar.selectbox(
        "Task Mode", 
        ["Generate Exam Paper", "Summarize Chapter", "Ask a Question"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Select Source Material")
    
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
            
    # Branding in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.9em; padding-top: 20px;">
            <p><b>© 2026 GGSSS Morak Station - Khairabad</b></p>
            <p>Developed by Ashwani Goyal</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # --- Main Area ---
    st.header("3. Configuration")
    
    # Dynamic UI based on Task Mode
    language = "English"
    blueprint_file = None
    blueprint_path = None

    if task_mode == "Generate Exam Paper":
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
    else:
        # For 'Summarize Chapter' or 'Ask a Question'
        instructions = st.text_area(
            f"Instructions / {'Prompt' if task_mode == 'Summarize Chapter' else 'Question'}", 
            placeholder="Type your prompt or question here...",
            height=150
        )
        
    # Process blueprint upload safely
    if blueprint_file is not None and task_mode == "Generate Exam Paper":
        try:
            temp_bp_dir = tempfile.gettempdir()
            blueprint_path = os.path.join(temp_bp_dir, blueprint_file.name)
            with open(blueprint_path, "wb") as f:
                f.write(blueprint_file.getbuffer())
        except Exception as e:
            st.error(f"Friendly Warning: Error saving blueprint file. {e}")
            blueprint_path = None

    # Generate button
    button_text = "Generate Paper" if task_mode == "Generate Exam Paper" else ("Summarize" if task_mode == "Summarize Chapter" else "Find Answer")
    
    if st.button(button_text, type="primary"):
        if not selected_subject_id:
            st.error("Friendly Warning: Please select a subject PDF from the sidebar.")
        elif not instructions.strip():
            st.error(f"Friendly Warning: Please provide some instructions/question for the {task_mode.lower()}.")
        else:
            with st.spinner("Downloading document from Google Drive..."):
                book_temp_path = download_pdf_from_drive(service, selected_subject_id, selected_subject_name)
                
            if book_temp_path:
                result = process_with_gemini(task_mode, instructions, language, book_temp_path, blueprint_path)
                if result:
                    st.session_state.result_text = result

    # Show result if it exists in session state
    if st.session_state.result_text:
        st.markdown("---")
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader(f"{task_mode} Result")
        with col2:
            # Convert markdown to HTML for printing
            html_content = markdown.markdown(st.session_state.result_text, extensions=['tables'])
            js_html_str = json.dumps(html_content)
            
            # HTML button that creates a hidden iframe and prints ONLY the HTML
            components.html(
                f"""
                <div style="text-align: right;">
                    <button onclick="printContent()" style="padding: 10px 15px; font-size: 14px; font-weight: bold; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        🖨️ Print to PDF
                    </button>
                </div>
                <script>
                function printContent() {{
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
                            <title>Print Document</title>
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

        st.markdown(st.session_state.result_text)
        
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.9em;">
            <p><b>© 2026 GGSSS Morak Station - Khairabad</b> | Developed by Ashwani Goyal</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
