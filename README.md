# AI-Powered Exam Paper Generator

This project is an AI-powered exam paper generator that utilizes the Google Gemini API (1.5 Flash), Streamlit, and Google Drive API. It is designed to be highly memory-efficient by utilizing the Gemini File API and downloading large PDF textbooks only temporarily during processing.

## Prerequisites
- Python 3.9+
- A Google Cloud Platform (GCP) Account
- A Google Gemini API Key

## Setup Instructions

### 1. Installation
Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Google Service Account
To allow the app to read files from Google Drive:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Enable the **Google Drive API** for this project.
4. Go to **APIs & Services** > **Credentials**.
5. Click **Create Credentials** > **Service Account**.
6. Provide a name and create the service account. You do not need to assign it any specific GCP roles.
7. Click on the newly created Service Account, go to the **Keys** tab.
8. Click **Add Key** > **Create new key** and select **JSON**.
9. Save the downloaded JSON file into the root of this project and rename it to `credentials.json`.

### 3. Share Your Google Drive Folder
1. Open Google Drive and locate or create the folder containing your Class subfolders and Subject PDFs.
   - The structure should be: `Main Folder` -> `Class Name` -> `Book.pdf`
2. Right-click the **Main Folder** and select **Share**.
3. In the "Add people and groups" field, paste the email address of your Service Account (found in `credentials.json` under `"client_email"`).
4. Assign it "Viewer" access and click **Share**.
5. Get the Folder ID of the Main Folder. You can find this in the URL when you open the folder in your browser (e.g., `drive.google.com/drive/folders/YOUR_FOLDER_ID`).

### 4. Create `.env` file
Create a `.env` file in the root directory and add the following keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
DRIVE_FOLDER_ID=your_main_drive_folder_id_here
```

### 5. Running the Application
Run the Streamlit app locally:

```bash
streamlit run app.py
```

## Deployment Notes (Render Free Tier)
This application is optimized for environments with low memory (like Render's 512MB free tier limit):
- It uses the `google-generativeai` File Upload API (`genai.upload_file`) which passes the file directly to Gemini's storage, bypassing the need to load the entire PDF content into the app's local memory.
- Downloaded Drive PDFs are saved as temporary files and immediately deleted after uploading to Gemini.
- Ensure you set the Environment Variables in your Render dashboard, just as you did in the `.env` file. Copy the content of `credentials.json` into a Render Secret File if needed, or point `GOOGLE_APPLICATION_CREDENTIALS` to its path.
