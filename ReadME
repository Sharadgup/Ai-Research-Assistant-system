
---

# ✨ AI Note System & Multi-Agent Platform 📝🤖🗣️

A Flask-powered intelligent document and communication platform integrated with Google's Gemini AI. This system provides real-time text analytics, multilingual voice chat, PDF comprehension, and industry-specific AI agents for structured business and educational support.

---

## 📌 Table of Contents

- [🚀 Key Features](#-key-features)  
- [🛠️ Technology Stack](#️-technology-stack)  
- [📊 Conceptual Architecture](#-conceptual-architecture)  
- [⚙️ Setup and Installation](#-setup-and-installation)  
- [▶️ Running the Application](#️-running-the-application)  
- [🔄 Core Processing Flows](#-core-processing-flows)  
- [📁 Directory Structure](#-directory-structure)  

---

## 🚀 Key Features

- 🔐 **Secure Authentication**  
  - Username/password & Google OAuth 2.0 via Flask-Dance.

- 📄 **Text Analysis & AI Report Generation**  
  - Gemini-powered semantic analysis with optional chart data extraction.

- 💬 **Interactive AI Chat Interfaces**  
  - General, Report-based, and PDF-contextual conversations.

- 🗣️ **Multilingual Voice Assistant**  
  - Real-time speech recognition, language detection, and TTS response generation in supported languages (e.g., English, Hindi, German).

- 🧠 **Domain-Specific AI Agents**  
  - 🎓 **Education**, ⚕️ **Healthcare**, 🏗️ **Construction** — each tailored for context-specific use.

- 💾 **Data Persistence via MongoDB**  
  - Stores user sessions, document metadata, chat transcripts, voice interactions.

- ⚡ **Real-Time Interactivity**  
  - Leveraging `Flask-SocketIO` + `Eventlet` for asynchronous, low-latency UI/UX.

---

## 🛠️ Technology Stack

| Layer        | Technology                     |
|--------------|-------------------------------|
| **Backend**  | Python, Flask, Flask-SocketIO, Eventlet |
| **AI Engine**| Google Generative AI (Gemini 1.5 Flash) |
| **Database** | MongoDB (Cloud or Local)       |
| **Frontend** | HTML, CSS, JavaScript (Fetch API + SocketIO) |
| **Authentication** | Werkzeug, Flask-Dance (OAuth2.0) |
| **PDF Parsing** | PyMuPDF (fitz)              |
| **Environment** | `dotenv`, `venv`            |

---

## 📊 Conceptual Architecture

### System Interaction Flow  
![Diagram 1](https://github.com/user-attachments/assets/b298ba58-35ba-40df-aa1d-b462924760c0)

### Authentication Flows  
![Diagram 2](https://github.com/user-attachments/assets/d7ad95b3-6111-4ce5-97a5-b9897e0b1950)

### PDF & Chat Architecture  
![Diagram 3](https://github.com/user-attachments/assets/500b2e5a-a565-4a50-b624-ed9579d5c22f)

### Voice Assistant Flow  
![Diagram 4](https://github.com/user-attachments/assets/de1fe470-acae-4c78-9ed0-f01c8a5935c7)

### Domain Agent (Construction)  
![Diagram 5](https://github.com/user-attachments/assets/8e2b68c2-9ea7-4f38-9e6c-caad6db736a1)

### Diagram For Data Analysis 
![Diagram 1](https://github.com/user-attachments/assets/4a325d91-7ad8-4886-b28f-c182c8253a5a)

### File Upload and Initial Processing Workflow

![Diagram 2](https://github.com/user-attachments/assets/55346e06-6544-4f06-a730-c5092d15edfa)

### Data Cleaning Action Workflow

![Diagram 3](https://github.com/user-attachments/assets/b94344c4-2af9-4e8d-9239-486be45cc66e)

### Visualization Generation Workflow

![Diagram 4](https://github.com/user-attachments/assets/d3d1bdef-6ac3-4fc2-be21-2c216b0cab67)

### AI Insight Generation Workflow

![Diagram 5](https://github.com/user-attachments/assets/77d9add8-d2eb-4334-9ddb-26ec3716d54f)


---

## ⚙️ Setup and Installation

### 1. Prerequisites

- Python 3.8+
- MongoDB URI (Local/Atlas)
- Google Cloud Project w/ Generative AI API
- Google OAuth Credentials (optional for SSO)

### 2. Clone Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 3. Create & Activate Virtual Environment
  
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If `requirements.txt` is missing, use:
```bash
pip install Flask Flask-SocketIO eventlet python-dotenv google-generativeai pymongo flask-dance Werkzeug PyMuPDF
pip freeze > requirements.txt
```

### 5. Setup Upload Directory

```bash
mkdir uploads
```

### 6. Configure Environment Variables

Create a `.env` file in the root directory with:

```dotenv
FLASK_SECRET_KEY='your-secure-secret'
GEMINI_API_KEY='your-gemini-api-key'
MONGODB_URI='your-mongodb-uri'
MONGODB_DB_NAME='ai_note_system'

# Optional: Google OAuth
GOOGLE_OAUTH_CLIENT_ID='your-client-id.apps.googleusercontent.com'
GOOGLE_OAUTH_CLIENT_SECRET='your-client-secret'

# For local development only
OAUTHLIB_INSECURE_TRANSPORT='1'
```

---

## ▶️ Running the Application

### 1. Activate Environment

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Start Flask Server

```bash
python app.py
```

### 3. Visit Application

[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔄 Core Processing Flows

> **Refer to the full technical flow breakdown above** for:
- User Authentication (Password & Google OAuth)
- AI Report Generation
- PDF Upload & Chat Mechanism
- Multilingual Voice Interaction
- Domain-Specific Agent Chat

---

## 📁 Directory Structure

```text
├── app.py                          # Main Flask application entry point
├── config.py                       # Environment config handler
├── /templates                      # HTML templates (Jinja2)
│   ├── login.html
│   ├── dashboard.html
│   └── ...
├── /static                         # CSS, JavaScript, frontend assets
│   ├── css/
│   └── js/
├── /uploads                        # Uploaded PDF files
├── /utils                          # Utility modules (e.g., PDF extraction, chart parsing)
│   ├── pdf_utils.py
│   ├── chart_parser.py
│   └── voice_utils.py
├── /routes                         # Flask route definitions
│   ├── auth_routes.py
│   ├── chat_routes.py
│   ├── report_routes.py
│   └── pdf_routes.py
├── /sockets                        # Socket.IO event handlers
│   ├── voice_handlers.py
│   └── pdf_chat_handlers.py
├── /models                         # MongoDB schema & collections (logical abstraction)
│   ├── user_model.py
│   └── report_model.py
├── requirements.txt                # Python dependency list
├── .env                            # Environment variables (excluded via .gitignore)
└── README.md                       # This file
```

---

## 📣 Final Note

This platform is scalable, real-time capable, and language-agnostic, built for multi-agent communication and AI-powered document intelligence. Ideal for integration with business dashboards, voice-enabled systems, and educational analytics platforms.

For professional deployment, ensure:
- HTTPS is enforced
- OAuth transport is secure
- MongoDB is properly IP-restricted
- API keys are stored in a vault or encrypted secrets manager

---

