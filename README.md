# 📝✨ AI Note Taker & Chat Analyzer ✨📊

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.x%2B-green.svg)](https://flask.palletsprojects.com/)
[![Socket.IO](https://img.shields.io/badge/Socket.IO-✓-brightgreen.svg)](https://socket.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-✓-forestgreen.svg)](https://www.mongodb.com/)
[![Google Gemini API](https://img.shields.io/badge/Gemini%20API-✓-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web application that leverages the power of Google's Gemini AI to analyze text input, generate comprehensive documentation reports, visualize key data, and allow users to interactively chat with the AI about the generated report. All inputs, reports, and chat interactions are stored persistently in a MongoDB database.

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 🚀 Features

*   **📄 Text Analysis & Report Generation:** Input any block of text and get a structured report including:
    *   Summary
    *   Key Findings
    *   Detailed Analysis
    *   Sentiment Analysis
    *   Keywords
    *   Conclusion
*   **📊 Data Visualization:** Automatically generates charts based on the report:
    *   Keyword Frequency (Bar Chart)
    *   Sentiment Score Distribution (Doughnut Chart)
*   **💬 Interactive Chat:** Engage in a real-time chat session with the AI specifically about the generated report content. The AI maintains context based on the report.
*   **💾 Persistent Storage:**
    *   Saves original input text (`input_prompts` collection).
    *   Saves generated reports and chart data (`documentation` collection).
    *   Saves full chat conversation history linked to each report (`chats` collection) in MongoDB.
*   **📄 PDF Export:** Download the generated report (including text and charts) as a PDF document.
*   **🌐 Real-time Communication:** Uses Flask-SocketIO for seamless chat interaction and typing indicators.
*   **🔐 Environment Variable Driven:** Securely configured using a `.env` file for API keys and database URIs.

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 🛠️ System Requirements

*   **Python:** 3.10 or higher
*   **Pip:** Python package installer (usually comes with Python)
*   **MongoDB:**
    *   A running MongoDB instance (local or cloud-based like MongoDB Atlas).
    *   MongoDB Connection String (URI).
*   **Google Cloud Account & Gemini API Key:**
    *   Access to the Google AI Studio or Google Cloud Console to generate a Gemini API key.
*   **Web Browser:** Modern browser supporting JavaScript, WebSockets, Chart.js, and html2pdf.js (Chrome, Firefox, Edge, Safari recommended).
*   **(Optional) Git:** For cloning the repository.

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ai-note-system # Or your repository directory name
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv
    ```
    *   Activate it:
        *   **Windows:** `.venv\Scripts\activate`
        *   **macOS/Linux:** `source .venv/bin/activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure your `requirements.txt` file includes `Flask`, `Flask-SocketIO`, `python-dotenv`, `google-generativeai`, `pymongo`, `eventlet`)*

4.  **Create `.env` File:**
    Create a file named `.env` in the root project directory and add your credentials:
    ```dotenv
    # .env file
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
    FLASK_SECRET_KEY=YOUR_OWN_STRONG_SECRET_KEY # Generate a random string for production
    MONGODB_URI=mongodb+srv://<username>:<password>@<your-cluster-url>/?retryWrites=true&w=majority # URI without DB name is fine
    MONGODB_DB_NAME=ai_note_taker_db # Or your desired database name
    ```
    *   **Replace placeholders** with your actual Gemini API key, a strong secret key, your MongoDB username/password/cluster URL, and your desired database name.
    *   **Important:** Add `.env` to your `.gitignore` file to prevent accidentally committing secrets!

5.  **Ensure Directory Structure:**
    Verify your project has the necessary structure referenced in `app.py`:
    ```
    your-project-root/
    ├── .venv/
    ├── src/
    │   ├── static/
    │   │   ├── script.js
    │   │   └── style.css
    │   └── templates/
    │       └── index.html
    ├── app.py
    ├── requirements.txt
    ├── .env         <-- Your environment variables
    └── README.md    <-- This file
    ```

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## ▶️ Running the Application

1.  **Activate Virtual Environment** (if not already active):
    *   **Windows:** `.venv\Scripts\activate`
    *   **macOS/Linux:** `source .venv/bin/activate`

2.  **Start the Flask Server:**
    ```bash
    python app.py
    ```

3.  **Access the Application:**
    Open your web browser and navigate to:
    `http://127.0.0.1:5000` (or `http://localhost:5000`)

    *(If running in a containerized or cloud environment like Google Cloud Workstations, access it via the provided public/forwarded URL).*

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 🕹️ How to Use

1.  **Enter Text:** Paste or type the text you want to analyze into the input text area.
2.  **Generate Report:** Click the "Generate Report" button.
3.  **View Report & Charts:** The generated report and visualizations (if applicable) will appear.
4.  **Chat about the Report:** Use the chat interface that appears below the report to ask the AI questions *about the content of the report you just generated*.
5.  **Download PDF:** Click the "Download PDF" button to save the report section as a PDF file.

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 📄 Database Collections (MongoDB)

*   **`input_prompts`**: Stores the original text submitted by the user.
    *   `_id`: ObjectId (Primary Key)
    *   `original_text`: String (The user's input)
    *   `timestamp`: ISODate (When the input was submitted)
    *   `related_documentation_id`: ObjectId (Reference to the generated report, added after generation)
*   **`documentation`**: Stores the generated reports and associated data.
    *   `_id`: ObjectId (Primary Key)
    *   `input_prompt_id`: ObjectId (Reference to the original input)
    *   `report_html`: String (The main report content generated by AI)
    *   `chart_data`: Object (JSON data for visualizations, e.g., `keyword_frequencies`, `sentiment_score`)
    *   `timestamp`: ISODate (When the report was generated)
    *   `model_used`: String (e.g., "gemini-1.5-flash")
    *   `finish_reason`: String (Gemini API finish reason, e.g., "STOP")
*   **`chats`**: Stores the conversation history for each report.
    *   `_id`: ObjectId (Primary Key for the chat *session*)
    *   `documentation_id`: ObjectId (Reference to the report being discussed)
    *   `start_timestamp`: ISODate (When the first message for this report was sent)
    *   `initial_sid`: String (SocketIO SID of the client who initiated this chat document)
    *   `messages`: Array of Objects:
        *   `role`: String ("user" or "AI")
        *   `text`: String (The content of the message)
        *   `timestamp`: ISODate (When the message was recorded)

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 💡 Potential Improvements & Future Ideas

*   [ ] **User Authentication:** Implement user accounts to separate reports and chats.
*   [ ] **Error Handling:** More granular error reporting to the user interface.
*   [ ] **Streaming Responses:** Stream AI responses in the chat for a more interactive feel.
*   [ ] **Report History:** Allow users to view and reload previously generated reports.
*   [ ] **Scalable Chat History:** For very long chats, implement pagination or smarter history loading instead of sending the full history to Gemini each time.
*   [ ] **Deployment:** Add instructions/scripts for deploying to platforms like Heroku, Google Cloud Run, etc.
*   [ ] **Unit & Integration Tests:** Add automated tests for reliability.
*   [ ] **UI/UX Enhancements:** Improve styling, add loading states for chat responses, etc.
*   [ ] **Alternative AI Models:** Add support for switching between different Gemini models or other providers.

![Separator](https://via.placeholder.com/800x4/cccccc/ffffff?text=+)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file (if you have one) for details.