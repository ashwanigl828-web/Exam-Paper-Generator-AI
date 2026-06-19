import os
import time
import tempfile
import io
import random
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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

# Fix gRPC malloc crash issues in Streamlit Cloud
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

load_dotenv()

# --- Config & Setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = "credentials.json"
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

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

def get_pdfs_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    return execute_with_retry(_fetch_drive_files, service, query)

def download_pdf_from_drive(service, file_id, file_name):
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
@st.cache_resource(show_spinner=False)
def load_or_create_faiss_index(file_id, pdf_path):
    """
    Checks if a FAISS index exists for the given Drive file ID.
    If not, processes the local temp PDF, chunks it, creates embeddings, and saves to local FAISS.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, file_id)
    
    # Use Gemini Embeddings to avoid memory corruption (PyTorch crashes on Streamlit Cloud)
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        raise ValueError("GEMINI_API_KEY is required for embeddings.")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_keys[0]
    )
    
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return db
    else:
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF file was not downloaded correctly.")
            
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_path)
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
Instructions/Requirements: {instructions}

Context:
{context_text}
"""
    else:
        prompt = f"""
You are a helpful teaching assistant.
Answer the user's question based ONLY on the provided context.
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
                model="gemini-1.5-flash", 
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
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18, spaceAfter=14)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=10)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11, spaceAfter=8, leading=14)
    
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
        
    st.title("🎓 Smart School Learning Suite")
    st.markdown("---")
    
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
        
        subjects = get_pdfs_in_folder(service, selected_class_id)
        if not subjects:
            st.sidebar.warning("No subject PDFs found in this class folder.")
        else:
            subject_names = [s['name'] for s in subjects]
            selected_subject_name = st.sidebar.selectbox("Select Subject (Book PDF)", subject_names)
            selected_subject_id = next(s['id'] for s in subjects if s['name'] == selected_subject_name)
            
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
        instructions = st.text_area(
            f"Instructions / {'Prompt' if task_mode == 'Summarize Chapter' else 'Question'}", 
            placeholder="Type your prompt or question here...",
            height=150
        )
        
    button_text = "Generate Paper" if task_mode == "Generate Exam Paper" else ("Summarize" if task_mode == "Summarize Chapter" else "Find Answer")
    
    if st.button(button_text, type="primary"):
        if not selected_subject_id:
            st.error("Please select a valid subject PDF from the sidebar.")
        elif not instructions.strip():
            st.error("Please provide instructions or a question.")
        else:
            try:
                # 1. Check if FAISS exists, otherwise download and process
                index_path = os.path.join(VECTOR_STORE_DIR, selected_subject_id)
                book_temp_path = None
                
                # We only need to download if the index doesn't exist yet
                if not (os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss"))):
                    with st.spinner("Downloading PDF from Google Drive..."):
                        book_temp_path = download_pdf_from_drive(service, selected_subject_id, selected_subject_name)
                
                with st.spinner("Preparing vector store (this runs once per book)..."):
                    db = load_or_create_faiss_index(selected_subject_id, book_temp_path)
                    
                # Clean up temp file
                if book_temp_path and os.path.exists(book_temp_path):
                    os.remove(book_temp_path)
                    
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
            st.warning(f"Could not generate PDF: {e}")
        
        st.markdown(st.session_state.result_text)
        
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.9em;">
            <p><b>© 2026 GGSSS Morak Station | Developed by Ashwani Goyal</b></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
