
```markdown
# ✨ AI Note System & Multi-Agent Platform 📝🤖🗣️

This Flask-based web application provides a suite of tools powered by Google's Gemini AI, focusing on text analysis, interactive chat, PDF processing, and specialized AI agents, including a multilingual voice assistant.

## 🚀 Key Features

*   🔐 **Secure Authentication:** Supports both traditional username/password registration/login and Google OAuth 2.0 for seamless access.
*   📄 **Text Analysis & Reporting:** Submit text for analysis by Gemini, generating structured reports and optional chart data visualizations.
*   💬 **Real-time Chat Interfaces:**
    *   **Report Chat:** Discuss generated reports directly with the AI for clarification or follow-up questions.
    *   **General Chat:** Engage in general conversation with the AI within the user dashboard.
    *   **PDF Chat:** Upload PDF documents, extract text, and chat with the AI specifically about the document's content.
*   🗣️ **Multilingual Voice Assistant:** Interact with the AI using your voice in multiple languages (e.g., English, Hindi, German). The system handles transcription, language detection (passed from frontend), multilingual AI response generation, and prepares the response for Text-to-Speech (TTS) output in the detected language.
*   🧠 **Specialized AI Agents:** Dedicated interfaces for specific domains:
    *   **🎓 Education Agent:** Get assistance with educational queries.
    *   **⚕️ Healthcare Agent:** Ask health-related questions (with appropriate disclaimers).
    *   **🏗️ Construction Agent:** Provide data and queries for construction-related analysis and insights, including potential chart generation.
*   💾 **Persistent Storage:** Uses MongoDB to store user data, uploaded PDF metadata, chat histories, generated reports, and voice conversation logs.
*   ⚙️ **Asynchronous Operations:** Leverages Flask-SocketIO and Eventlet for real-time communication and efficient handling of concurrent users.

## 🛠️ Technology Stack

*   **Backend:** Python 🐍, Flask, Flask-SocketIO, Eventlet ⚡
*   **AI:** Google Generative AI API (Gemini 1.5 Flash) 💎
*   **Database:** MongoDB 🍃
*   **PDF Processing:** PyMuPDF (fitz) 📄
*   **Authentication:** Werkzeug Security, Flask-Dance (Google OAuth) 🔑
*   **Frontend:** HTML, CSS, JavaScript (interacting via Fetch API and SocketIO) 🎨
*   **Deployment Considerations:** Werkzeug ProxyFix (for reverse proxy compatibility) ☁️

## 📊 High-Level Conceptual Flow

This diagram illustrates the general interaction flow:

Diagram 1: Overall System Architecture & Request Flow
  
![Screenshot (224)](https://github.com/user-attachments/assets/b298ba58-35ba-40df-aa1d-b462924760c0)

Diagram 2: Authentication Flow (Password & Google OAuth)

![Screenshot (225)](https://github.com/user-attachments/assets/d7ad95b3-6111-4ce5-97a5-b9897e0b1950)

Diagram 3: PDF Upload and Chat Interaction Workflow

![Screenshot (226)](https://github.com/user-attachments/assets/500b2e5a-a565-4a50-b624-ed9579d5c22f)

Diagram 4: Multilingual Voice Assistant Interaction Flow

![Screenshot (227)](https://github.com/user-attachments/assets/de1fe470-acae-4c78-9ed0-f01c8a5935c7)

Diagram 5: Data Flow for Specific Agent (e.g., Construction)

![Screenshot (228)](https://github.com/user-attachments/assets/8e2b68c2-9ea7-4f38-9e6c-caad6db736a1)


## ⚙️ Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   MongoDB Atlas account (or local MongoDB instance)
    *   Google Cloud Project with Generative AI API enabled
    *   Google OAuth Credentials (Client ID & Secret) if enabling Google Login

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *(Note: A `requirements.txt` file should be created from the project's dependencies. If it doesn't exist, you can generate one while the virtual environment is active using `pip freeze > requirements.txt` after installing necessary packages manually or from the imports.)*
    ```bash
    pip install -r requirements.txt
    # Or install manually based on imports:
    # pip install Flask Flask-SocketIO eventlet python-dotenv google-generativeai pymongo flask-dance Werkzeug PyMuPDF
    ```

5.  **Create Uploads Directory:**
    The application saves uploaded PDFs here.
    ```bash
    mkdir uploads
    ```

6.  **Configure Environment Variables:**
    Create a `.env` file in the project root and add the following variables:

    ```dotenv
    # --- Core Secrets ---
    FLASK_SECRET_KEY='a_very_strong_and_random_secret_key_here' # CHANGE THIS! Generate a secure random key.
    GEMINI_API_KEY='YOUR_GOOGLE_GENERATIVE_AI_API_KEY'

    # --- MongoDB Configuration ---
    MONGODB_URI='YOUR_MONGODB_CONNECTION_STRING' # e.g., mongodb+srv://user:password@cluster.mongodb.net/
    MONGODB_DB_NAME='YOUR_DATABASE_NAME' # e.g., ai_note_system

    # --- Google OAuth (Optional but Recommended for Google Login) ---
    GOOGLE_OAUTH_CLIENT_ID='YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com'
    GOOGLE_OAUTH_CLIENT_SECRET='YOUR_GOOGLE_CLIENT_SECRET'

    # --- OAuth Development Setting (Set to '0' for Production!) ---
    # Allows OAuth over HTTP for local development if needed. MUST be '0' in production (HTTPS).
    OAUTHLIB_INSECURE_TRANSPORT='1' # Use '1' ONLY for local testing if not using HTTPS
    ```

    *   **IMPORTANT:** Ensure `FLASK_SECRET_KEY` is strong and unique.
    *   Get your `GEMINI_API_KEY` from Google AI Studio or Google Cloud Console.
    *   Configure `MONGODB_URI` and `MONGODB_DB_NAME` for your MongoDB deployment.
    *   Generate `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` in the Google Cloud Console under APIs & Services -> Credentials. **Crucially,** you MUST configure the authorized redirect URIs in Google Cloud to match the `forced_redirect_uri` specified in `app.py` (e.g., `https://your-deployed-domain.com/login/google/authorized` or `http://127.0.0.1:5000/login/google/authorized` for local testing).

7.  **Eventlet Monkey Patching:**
    The code includes `eventlet.monkey_patch()` at the very top. This is crucial for `Flask-SocketIO`'s asynchronous operations with Eventlet. Ensure it remains one of the first imports.

## ▶️ Running the Application

1.  **Activate the Virtual Environment** (if not already active):
    ```bash
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Start the Flask Server:**
    ```bash
    python app.py
    ```

3.  **Access the Application:**
    Open your web browser and navigate to `http://127.0.0.1:5000` (or the host/port configured if changed).

## 🔄 Core Processing Flows (Step-by-Step)

Here's how key features process requests:

### 1. User Authentication (Password)

1.  **Request:** User submits username/password via Login form (`/login`).
2.  **Flask:** Route `@app.route('/login', methods=['POST'])` receives the data.
3.  **Validation:** Checks if username and password exist.
4.  **Database:** Queries `registrations_collection` in MongoDB for the username.
5.  **Verification:** If user found, uses `check_password_hash` to compare the submitted password against the stored hash.
6.  **Session:** If verification succeeds, calls `login_user` helper to store `user_id`, `username`, and `login_method` in the Flask session.
7.  **Response:** Redirects the user to the `/dashboard`. On failure, re-renders the login page with an error flash message.

### 2. User Authentication (Google OAuth)

1.  **Request:** User clicks "Login with Google" button, linking to `/login/google`.
2.  **Flask-Dance:** The `google_bp` blueprint initiates the OAuth flow, redirecting the user to Google's authentication page.
3.  **Google:** User authenticates with Google and authorizes the application.
4.  **Redirect:** Google redirects the user back to the `redirect_uri` configured in the blueprint (`/login/google/authorized`).
5.  **Flask:** Route `@app.route('/google/authorized')` handles the callback.
6.  **Flask-Dance:** Verifies the response and fetches user info (ID, email, name) from Google using `google.get()`.
7.  **Database:**
    *   Checks `registrations_collection` if a user exists with the `google_id`.
    *   If not found by ID, checks by `email`.
    *   If found, updates `last_login_at`, `login_method`, and potentially adds missing `google_id` or `name`.
    *   If not found, creates a new user record with Google info.
8.  **Session:** Calls `login_user` to log the user in.
9.  **Response:** Redirects the user to the `/dashboard`. On failure, redirects to login with an error.

### 3. Report Generation (`/generate_report`)

1.  **Request:** Frontend sends text input via `fetch` POST request (JSON) to `/generate_report`.
2.  **Flask:** Route receives JSON data, extracts `text`.
3.  **Database (Input):** Saves the original `input_text` and `user_id` (if logged in) to `input_prompts_collection`.
4.  **AI Prompt:** Formats a prompt including the user's text, asking Gemini for analysis and potentially structured JSON chart data within specific markers (e.g., ` ```json_chart_data...``` `).
5.  **Gemini API:** Calls `model.generate_content(prompt)`.
6.  **Flask (Processing):**
    *   Receives the AI response.
    *   Extracts the main report text.
    *   Searches for the JSON markers (` ```json_chart_data...``` `).
    *   If found, parses the enclosed JSON string into `chart_data`.
    *   Removes the JSON block from the main `report_content`.
7.  **Database (Output):** Saves the `report_content`, `chart_data`, `user_id`, `model_used`, etc., to the `documentation_collection`. Links it back to the original prompt document if saved successfully.
8.  **Response:** Returns a JSON object containing `report_html`, `chart_data`, `report_context_for_chat` (truncated report), and `documentation_id` to the frontend.
9.  **Frontend:** JavaScript receives the JSON and dynamically renders the report and chart (if data exists).

### 4. PDF Analysis & Chat

1.  **Upload (`/upload_pdf`):**
    *   **Request:** User uploads a PDF file via the form on `/pdf_analyzer`.
    *   **Flask:** Route receives the file, validates (`allowed_file`), generates a secure filename.
    *   **Storage:** Saves the PDF to the `UPLOAD_FOLDER`.
    *   **Extraction:** Calls `extract_text_from_pdf` (using PyMuPDF) to get text content and page count.
    *   **Database:** Saves metadata (filename, path, user info, text preview, page count) to `pdf_analysis_collection`.
    *   **Response:** Returns JSON success message with `analysis_id`, filename, and text preview.
    *   **Frontend:** Updates the UI, potentially displaying the preview and making the chat available for this PDF.
2.  **Chat (`/pdf_chat` Namespace):**
    *   **Request:** User sends a message via the PDF chat interface (associated with a specific `analysis_id`). Frontend emits `send_pdf_chat_message` via SocketIO.
    *   **SocketIO Handler:** `handle_pdf_chat_message` receives the message (`text`) and `analysis_id`.
    *   **Database (Save User Msg):** Appends the user's message (`role: 'user'`) to the `messages` array in the corresponding document in `pdf_chats_collection` (identified by `analysis_id`).
    *   **Database (Get Context):** Fetches the `extracted_text_preview` from the `pdf_analysis_collection` document matching the `analysis_id`.
    *   **Database (Get History):** Fetches recent messages from the `pdf_chats_collection` document.
    *   **AI Prompt:** Constructs a prompt containing:
        *   The PDF text context.
        *   The recent chat history.
        *   The user's current message.
        *   Instructions to answer based *only* on the provided context and history.
    *   **Gemini API:** Calls `model.generate_content(prompt)`.
    *   **Database (Save AI Msg):** Appends the AI's response (`role: 'AI'`) to the `messages` array in `pdf_chats_collection`.
    *   **Response:** Emits `receive_pdf_chat_message` back to the specific client (room=sid) with the AI's response text.
    *   **Frontend:** JavaScript listens for `receive_pdf_chat_message` and displays the AI's answer in the chat window.

### 5. Voice Chat (`/voice_chat` Namespace)

1.  **User Interaction:** User clicks "Start Listening", speaks.
2.  **Frontend (STT):** Browser's SpeechRecognition API captures audio, transcribes it to text, and detects the language (e.g., `en-US`, `hi-IN`).
3.  **Frontend (Emit):** Emits `send_voice_text` via SocketIO to the `/voice_chat` namespace, sending the `text` and detected `lang` code.
4.  **SocketIO Handler:** `handle_send_voice_text` receives the data.
5.  **Validation:** Checks authentication, data validity.
6.  **Database (History):** Fetches recent messages (including `lang` codes) from `voice_conversations_collection` for the user.
7.  **AI Prompt (Multilingual):** Constructs a prompt that:
    *   Specifies the user's input language (`lang` code received).
    *   Includes the chat history (optional).
    *   Includes the user's current transcribed `text`.
    *   **Crucially:** Instructs Gemini to respond *in the same language* as the user's input.
8.  **Gemini API:** Calls `model.generate_content()` with the history and prompt.
9.  **Processing:** Extracts the AI's response text. Assumes the response language (`ai_lang`) matches the requested user language (`user_lang`).
10. **Database (Save Turn):** Appends *both* the user message (with `user_lang`) and the AI response (with `ai_lang`) to the `messages` array in `voice_conversations_collection`, ensuring language context is saved.
11. **Response (Emit):** Emits `receive_ai_voice_text` back to the specific client, sending the AI's `text` and the determined `ai_lang` code.
12. **Frontend (TTS):** JavaScript receives the message, uses the `ai_lang` code to select the correct voice/language in the browser's SpeechSynthesis API, and speaks the AI's response aloud.

## 📁 Directory Structure (Simplified)

```
.
├── .env                  # <-- **SECRET** Environment variables (Create this!)
├── app.py                # <-- Main Flask application logic
├── uploads/              # <-- Directory for uploaded PDF files (Created by app or manually)
├── src/
│   ├── static/           # <-- CSS, JavaScript, Images
│   │   └── css/
│   │   └── js/
│   │   └── img/
│   └── templates/        # <-- HTML Files (Jinja2)
│       ├── landing.html
│       ├── login.html
│       ├── register.html
│       ├── dashboard.html
│       ├── index.html        # (Report page)
│       ├── education_agent.html
│       ├── healthcare_agent.html
│       ├── construction_agent.html
│       ├── pdf_analyzer.html
│       ├── voice_agent.html  # <-- Voice assistant page
│       └── base.html         # (Optional base template)
│       └── includes/       # (Optional template partials)
└── requirements.txt      # <-- Python dependencies (Generate this!)
```

## ☁️ Deployment Notes

*   **HTTPS:** Production deployments **MUST** use HTTPS, especially when handling user authentication and OAuth.
*   **ProxyFix:** The `ProxyFix` middleware is included. This is essential if deploying behind a reverse proxy (like Nginx or Apache) to ensure Flask correctly interprets headers like `X-Forwarded-For`, `X-Forwarded-Proto`, etc., which is vital for security and URL generation (especially for OAuth redirects).
*   **Google OAuth Redirect URI:** The `forced_redirect_uri` in the Google Blueprint setup is **HARDCODED**. You **MUST** change this to match your production domain's callback URL (`https://your-domain.com/login/google/authorized`) and ensure this exact URI is registered in your Google Cloud OAuth credentials. For local testing, ensure the local URI (`http://127.0.0.1:5000/...`) is registered.
*   **`OAUTHLIB_INSECURE_TRANSPORT`:** Set this environment variable to `'0'` in production to enforce HTTPS for OAuth callbacks.
*   **Web Server:** Use a production-grade web server like Gunicorn or uWSGI with Eventlet workers to run the Flask app, typically placed behind a reverse proxy like Nginx.
    ```bash
    # Example with Gunicorn and Eventlet
    gunicorn --worker-class eventlet -w 1 app:app
    ```
*   **Environment Variables:** Ensure all required variables from the `.env` file are securely set in the production environment.

## 🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflows. Ensure code is well-formatted, includes comments where necessary, and updates relevant documentation.

## 📜 License

(Specify your license here, e.g., MIT License, Apache 2.0, etc.)

Example:
```
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted...
```
```
