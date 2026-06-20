import os

# Fix gRPC and FAISS malloc crash issues in Streamlit Cloud
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import tempfile
import io
import random
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# LangChain & FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# AI Engines
from groq import Groq

# ReportLab for PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ReportLab for PDF

load_dotenv()

# --- Config & Setup ---
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = "credentials.json"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "models/text-embedding-004"  # ✓ Verified working model with Google Generative AI v1beta
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Register Hindi Font if available
try:
    font_path = "NotoSansDevanagari-Regular.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('HindiFont', font_path))
except Exception as e:
    pass

def get_config(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, None)
    except Exception:
        return None

def get_drive_folder_id():
    val = get_config("DRIVE_FOLDER_ID")
    if val:
        if "folders/" in val:
            return val.split("folders/")[-1].split("?")[0].strip()
        elif "id=" in val:
            return val.split("id=")[-1].split("&")[0].strip()
        return val.strip()
    return None

def get_gemini_keys():
    keys_str = get_config("GEMINI_KEYS")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single_key = get_config("GEMINI_API_KEY")
    if single_key:
        return [k.strip() for k in single_key.split(",") if k.strip()]
    return []

def get_groq_key():
    return get_config("GROQ_API_KEY")

# --- Helper: Retry Logic ---
def execute_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["429", "connection", "quota", "timeout", "unavailable", "internal", "error"]):
            time.sleep(2)
            try:
                return func(*args, **kwargs)
            except Exception as retry_e:
                raise Exception(f"Operation failed after retry. ({retry_e})")
        else:
            raise Exception(f"An unexpected error occurred: {e}")

# --- Helper: Embedding Error Handler ---
def handle_embedding_error(error):
    """Handle embedding-related errors with helpful messages"""
    error_str = str(error)
    if "404" in error_str and "NOT_FOUND" in error_str:
        return f"❌ Embedding Model Error: The model '{EMBEDDING_MODEL}' is not available. This might be because: 1) Old FAISS vector stores need to be regenerated 2) API key doesn't have access to embeddings 3) Please clear cache and try again. Error: {error}"
    elif "PERMISSION_DENIED" in error_str or "permission" in error_str.lower():
        return f"❌ API Permission Error: Your API key doesn't have permission to use embeddings. Please check your Google Cloud credentials."
    elif "401" in error_str or "unauthorized" in error_str.lower():
        return f"❌ API Authentication Error: Invalid or missing API key. Please check GEMINI_API_KEY configuration."
    else:
        return f"❌ Embedding Error: {error}"

# --- Google Drive Helpers ---
@st.cache_resource(show_spinner=False)
def get_drive_service():
    creds = None
    try:
        if get_config("gcp_service_account"):
            creds_dict = dict(get_config("gcp_service_account"))
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
        st.error(f"Failed to connect to Google Drive: {e}")
        return None

def _fetch_drive_files(service, query):
    results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
    return results.get('files', [])

def get_folders_in_drive(service, parent_id):
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    return execute_with_retry(_fetch_drive_files, service, query)

def get_zips_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/zip' and trashed=false"
    return execute_with_retry(_fetch_drive_files, service, query)

def create_folder_in_drive(service, parent_id, folder_name):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    def _create():
        return service.files().create(body=file_metadata, fields='id').execute()
    folder = execute_with_retry(_create)
    return folder.get('id')

def upload_file_to_drive(service, parent_id, file_path, file_name, mime_type):
    file_metadata = {
        'name': file_name,
        'parents': [parent_id]
    }
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    def _upload():
        return service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file = execute_with_retry(_upload)
    return file.get('id')

def download_file_from_drive(service, file_id, file_name):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file_name)
    
    def _download():
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*5)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(fh.read())
        return temp_path

    return execute_with_retry(_download)

# --- RAG & Vector Store Logic ---
def create_and_save_faiss(pdf_path, save_dir):
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        raise ValueError("GEMINI_API_KEY is required for embeddings.")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=gemini_keys[0]
        )
    except Exception as e:
        raise Exception(handle_embedding_error(e))
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    try:
        db = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise Exception(handle_embedding_error(e))
    
    os.makedirs(save_dir, exist_ok=True)
    db.save_local(save_dir)
    return db

@st.cache_resource(show_spinner=False)
def load_faiss_from_zip(_zip_path, extract_dir):
    with zipfile.ZipFile(_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    gemini_keys = get_gemini_keys()
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=gemini_keys[0]
        )
    except Exception as e:
        raise Exception(handle_embedding_error(e))
    
    # allow_dangerous_deserialization=True is required to load FAISS indices, 
    # but since we generated them ourselves, it is safe.
    try:
        db = FAISS.load_local(extract_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise Exception(handle_embedding_error(e))
    
    return db

# --- Hybrid AI Engine ---
def process_hybrid_ai(task_mode, instructions, language, context_text):
    groq_key = get_groq_key()
    gemini_keys = get_gemini_keys()
    
    if task_mode == "Generate Exam Paper":
        prompt = f"""
You are an expert exam paper generator. 
Create an exam paper based primarily on the provided context.
Language: {language}
Instructions/Requirements: {instructions}

Format the output clearly. Use simple headings (e.g., Section A, Q1, Q2) without excessive markdown. 
Do not include conversational filler, only the exam paper content.

Context:
{context_text}
"""
    elif task_mode == "Summarize Chapter":
        prompt = f"""
You are an expert teacher and summarizer.
Create a concise, easy-to-understand explanation or summary based on the provided context.
Language: {language}
Instructions/Requirements: {instructions}

Context:
{context_text}
"""
    else:
        prompt = f"""
You are a helpful teaching assistant.
Answer the user's question based ONLY on the provided context.
Language: {language}
Question/Instructions: {instructions}

Context:
{context_text}
"""

    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000,
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.warning(f"Groq API failed ({e}). Falling back to Gemini...")
    else:
        st.info("No Groq API key found. Defaulting to Gemini...")

    if not gemini_keys:
        raise ValueError("No Gemini keys configured for fallback.")
        
    random.shuffle(gemini_keys)
    last_error = None
    
    for key in gemini_keys:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash", 
                google_api_key=key, 
                temperature=0.7
            )
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            last_error = e
            continue
            
    raise Exception(f"All AI engines failed. Last error: {last_error}")

# --- PDF Generation (ReportLab) ---
def create_pdf_from_text(text, filename="output.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Use Hindi font if registered, else fallback to standard fonts
    font_name = 'HindiFont' if 'HindiFont' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontName=font_name, alignment=TA_CENTER, fontSize=18, spaceAfter=14)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceAfter=10, spaceBefore=10)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontName=font_name, fontSize=11, spaceAfter=8, leading=14)
    
    story = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 10))
            continue
            
        if line.startswith('# '):
            story.append(Paragraph(line[2:], title_style))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], heading_style))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], heading_style))
        elif line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(f"<b>{line[2:-2]}</b>", body_style))
        else:
            while '**' in line:
                line = line.replace('**', '<b>', 1).replace('**', '</b>', 1)
            story.append(Paragraph(line, body_style))
            
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --- Main App UI ---
def main():
    st.set_page_config(page_title="Smart School Learning Suite", layout="wide", page_icon="🎓")
    
    if "result_text" not in st.session_state:
        st.session_state.result_text = None
        
    if not get_gemini_keys() and not get_groq_key():
        st.error("No API keys configured. Please add GEMINI_KEYS and/or GROQ_API_KEY to your .env file or Streamlit Secrets.")
        return

    drive_folder_id = get_drive_folder_id()
    if not drive_folder_id:
        st.error("DRIVE_FOLDER_ID is missing in the .env file or Streamlit Secrets.")
        return

    service = get_drive_service()
    if not service:
        st.warning("Google Drive credentials not found or invalid. Please check your credentials.json and secrets.")
        return

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", ["🎓 Learning Suite", "⚙️ Manage Books"])
    st.sidebar.markdown("---")

    if app_mode == "⚙️ Manage Books":
        st.title("⚙️ Manage Books & Upload")
        st.markdown("Upload a PDF book. The bot will automatically vectorize it and save the FAISS index to Google Drive so it loads instantly next time.")
        
        st.subheader("1. Class Details")
        classes = get_folders_in_drive(service, drive_folder_id)
        class_names = [c['name'] for c in classes] if classes else []
        
        col1, col2 = st.columns(2)
        with col1:
            class_action = st.radio("Class Action", ["Select Existing Class", "Create New Class"])
        
        with col2:
            selected_class_id = None
            if class_action == "Select Existing Class":
                if not class_names:
                    st.warning("No existing classes found.")
                else:
                    sel_class = st.selectbox("Select Class", class_names)
                    selected_class_id = next(c['id'] for c in classes if c['name'] == sel_class)
            else:
                new_class_name = st.text_input("New Class Name (e.g., Class 10)")
                
        st.subheader("2. Book Details")
        book_name = st.text_input("Book Name (e.g., Science, Mathematics Vol 1)")
        uploaded_pdf = st.file_uploader("Upload PDF Book", type=["pdf"])
        
        if st.button("Process & Upload to Drive", type="primary"):
            if class_action == "Create New Class" and new_class_name:
                with st.spinner(f"Creating class folder '{new_class_name}' in Drive..."):
                    selected_class_id = create_folder_in_drive(service, drive_folder_id, new_class_name)
            
            if not selected_class_id:
                st.error("Please select or create a valid Class.")
            elif not book_name.strip():
                st.error("Please enter a valid Book Name.")
            elif not uploaded_pdf:
                st.error("Please upload a PDF file.")
            else:
                try:
                    book_clean_name = book_name.strip().replace(" ", "_")
                    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{book_clean_name}_{int(time.time())}.pdf")
                    
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    
                    faiss_dir = os.path.join(VECTOR_STORE_DIR, book_clean_name)
                    
                    with st.spinner("Processing PDF: Chunking and creating FAISS Vector Text..."):
                        create_and_save_faiss(temp_pdf_path, faiss_dir)
                        
                    with st.spinner("Zipping Vector Text..."):
                        zip_base_path = os.path.join(tempfile.gettempdir(), f"{book_clean_name}_faiss")
                        shutil.make_archive(zip_base_path, 'zip', faiss_dir)
                        zip_file_path = f"{zip_base_path}.zip"
                        
                    with st.spinner("Uploading Vector Text (FAISS Zip) to Google Drive..."):
                        upload_file_to_drive(service, selected_class_id, zip_file_path, f"{book_clean_name}_faiss.zip", "application/zip")
                        
                    st.success(f"Successfully processed '{book_name}' and uploaded its vector text to Drive!")
                    
                    # Cleanup local temp files
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    if os.path.exists(zip_file_path):
                        os.remove(zip_file_path)
                    if os.path.exists(faiss_dir):
                        shutil.rmtree(faiss_dir)
                        
                except Exception as e:
                    error_msg = str(e)
                    if "embedding" in error_msg.lower() or "404" in error_msg or "NOT_FOUND" in error_msg:
                        st.error(error_msg)
                    else:
                        st.error(f"An error occurred during upload: {e}")

    else:
        st.title("🎓 Smart School Learning Suite")
        st.markdown("---")
        
        # --- Sidebar ---
        st.sidebar.header("1. Select Task Mode")
        task_mode = st.sidebar.selectbox(
            "Task Mode", 
            ["Generate Exam Paper", "Summarize Chapter", "Ask a Question"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("2. Select Source Material")
        
        classes = get_folders_in_drive(service, drive_folder_id)
        selected_subject_id = None
        selected_subject_name = None
        
        if not classes:
            st.sidebar.error("No Class folders found in the specified Drive folder.")
        else:
            class_names = [c['name'] for c in classes]
            selected_class_name = st.sidebar.selectbox("Select Class", class_names)
            selected_class_id = next(c['id'] for c in classes if c['name'] == selected_class_name)
            
            # Now fetch ZIP files instead of PDFs
            subjects = get_zips_in_folder(service, selected_class_id)
            if not subjects:
                st.sidebar.warning("No processed books (Vector Zips) found in this class. Please upload using 'Manage Books'.")
            else:
                # Display name without _faiss.zip
                subject_display_names = [s['name'].replace("_faiss.zip", "").replace(".zip", "").replace("_", " ") for s in subjects]
                sel_subject_disp = st.sidebar.selectbox("Select Subject/Book", subject_display_names)
                
                # Find the matching original zip file name
                original_zip_name = next(s['name'] for s in subjects if s['name'].replace("_faiss.zip", "").replace(".zip", "").replace("_", " ") == sel_subject_disp)
                selected_subject_id = next(s['id'] for s in subjects if s['name'] == original_zip_name)
                selected_subject_name = original_zip_name
                
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            <div style="text-align: center; color: gray; font-size: 0.9em; padding-top: 20px;">
                <p><b>© 2026 GGSSS Morak Station | Developed by Ashwani Goyal</b></p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # --- Main Area ---
        st.header("3. Configuration")
        
        language = "English"
        if task_mode == "Generate Exam Paper":
            col1, col2 = st.columns([1, 2])
            with col1:
                language = st.selectbox("Language", ["English", "Hindi", "Both (Bilingual)"])
            with col2:
                instructions = st.text_area(
                    "Instructions", 
                    placeholder="e.g., Chapter 5, 50 marks, Hard difficulty, 5 MCQs, 3 Long Qs",
                    height=150
                )
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                language = st.selectbox("Language", ["English", "Hindi", "Both (Bilingual)"])
            with col2:
                instructions = st.text_area(
                    f"Instructions / {'Prompt' if task_mode == 'Summarize Chapter' else 'Question'}", 
                    placeholder="Type your prompt or question here...",
                    height=150
                )
            
        button_text = "Generate Paper" if task_mode == "Generate Exam Paper" else ("Summarize" if task_mode == "Summarize Chapter" else "Find Answer")
        
        if st.button(button_text, type="primary"):
            if not selected_subject_id:
                st.error("Please select a valid subject/book from the sidebar.")
            elif not instructions.strip():
                st.error("Please provide instructions or a question.")
            else:
                try:
                    book_folder_name = selected_subject_name.replace(".zip", "")
                    extract_dir = os.path.join(VECTOR_STORE_DIR, book_folder_name)
                    
                    # 1. Check if FAISS exists locally, otherwise download ZIP and extract
                    if not (os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, "index.faiss"))):
                        with st.spinner("Downloading pre-processed vector text from Drive..."):
                            zip_path = download_file_from_drive(service, selected_subject_id, selected_subject_name)
                        
                        with st.spinner("Extracting vector store..."):
                            os.makedirs(extract_dir, exist_ok=True)
                            db = load_faiss_from_zip(zip_path, extract_dir)
                            
                        # Clean up temp zip file
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                    else:
                        with st.spinner("Loading local vector store..."):
                            gemini_keys = get_gemini_keys()
                            try:
                                embeddings = GoogleGenerativeAIEmbeddings(
                                    model=EMBEDDING_MODEL,
                                    google_api_key=gemini_keys[0]
                                )
                                db = FAISS.load_local(extract_dir, embeddings, allow_dangerous_deserialization=True)
                            except Exception as e:
                                raise Exception(handle_embedding_error(e))
                        
                    # 2. Retrieve Relevant Chunks
                    with st.spinner("Retrieving relevant context..."):
                        k_val = 15 if task_mode != "Ask a Question" else 5
                        docs = db.similarity_search(instructions, k=k_val)
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # 3. Generate Output using Hybrid Engine
                    with st.spinner("Generating response with AI Engine..."):
                        result = process_hybrid_ai(task_mode, instructions, language, context_text)
                        if result:
                            st.session_state.result_text = result
                            
                except Exception as e:
                    error_msg = str(e)
                    if "embedding" in error_msg.lower() or "404" in error_msg or "NOT_FOUND" in error_msg:
                        st.error(error_msg)
                    else:
                        st.error(f"An error occurred: {e}")

        # Show result
        if st.session_state.result_text:
            st.markdown("---")
            st.subheader(f"{task_mode} Result")
            
            try:
                pdf_bytes = create_pdf_from_text(st.session_state.result_text)
                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_bytes,
                    file_name="generated_document.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.warning(f"Could not generate PDF correctly. Ensure 'NotoSansDevanagari-Regular.ttf' is in the root directory. Error: {e}")
            
            st.markdown(st.session_state.result_text)

if __name__ == "__main__":
    main()
