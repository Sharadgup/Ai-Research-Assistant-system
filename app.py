# --- START: Apply Eventlet Monkey Patching ---
# This MUST be one of the very first things in your script
import eventlet
eventlet.monkey_patch()
# --- END: Apply Eventlet Monkey Patching ---

import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify # request comes from Flask
from flask_socketio import SocketIO, emit, send # Removed unused imports
from dotenv import load_dotenv
import logging
import json
from threading import Lock
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError # Import ConfigurationError
from datetime import datetime
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__,
            template_folder='src/templates',
            static_folder='src/static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_dev_only!')

# --- MongoDB Initialization (Using MONGODB_DB_NAME) ---
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME") # Read the DB name variable
db = None
input_prompts_collection = None
documentation_collection = None
chats_collection = None

if not MONGO_URI:
    logging.critical("CRITICAL: MONGODB_URI not found in environment variables.")
elif not DB_NAME:
    logging.critical("CRITICAL: MONGODB_DB_NAME not found in environment variables.")
else:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_client.admin.command('ismaster') # Test connection
        db = mongo_client[DB_NAME] # Select DB using the variable
        input_prompts_collection = db["input_prompts"]
        documentation_collection = db["documentation"]
        chats_collection = db["chats"]
        logging.info(f"Successfully connected to MongoDB. Database: '{DB_NAME}'")
    except ConnectionFailure:
        logging.critical(f"MongoDB connection failed: Could not connect to server at {MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI}")
        db = None
    except ConfigurationError as ce:
         logging.critical(f"MongoDB configuration error: {ce}")
         db = None
    except Exception as e:
        logging.critical(f"An unexpected error occurred during MongoDB initialization: {e}")
        logging.error(traceback.format_exc())
        db = None
# --------------------------------------------------------


# --- SocketIO Initialization (with CORS, Pings) ---
allowed_origins_list = [
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "https://5000-idx-ai-note-system-1744087101492.cluster-a3grjzek65cxex762e4mwrzl46.cloudworkstations.dev"
    # "*"
]
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins=allowed_origins_list,
    ping_timeout=20,
    ping_interval=10
    # logger=True,
    # engineio_logger=True
)

# --- Gemini API Configuration ---
api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-1.5-flash"
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [
             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        logging.info(f"Gemini model '{model_name}' initialized.")
    except Exception as e:
        logging.error(f"Error initializing Gemini model '{model_name}': {e}")
        logging.error(traceback.format_exc())
else:
    logging.critical("CRITICAL: GEMINI_API_KEY not found...")

# --- HTTP Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """Handles the report generation request, saves input & output to DB."""
    logging.info("Received request for /generate_report")
    if not model: return jsonify({"error": "AI model not available..."}), 503

    # *** CORRECTED DB CHECK: Using 'is None' ***
    if db is None:
        logging.error("/generate_report: Database connection is not available.")
        return jsonify({"error": "Database connection not available."}), 503
    # ******************************************

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); input_text = data.get('text')
    if not input_text or not isinstance(input_text, str) or len(input_text.strip()) == 0: return jsonify({"error": "No valid text provided"}), 400
    logging.info(f"/generate_report: Processing text (length: {len(input_text)}).")
    logging.info(f"/generate_report: Input text received: '{input_text[:200]}...'")

    # 1. Save Input Prompt to MongoDB
    prompt_doc_id = None
    try:
        prompt_doc = {"original_text": input_text, "timestamp": datetime.utcnow()}
        insert_result = input_prompts_collection.insert_one(prompt_doc)
        prompt_doc_id = insert_result.inserted_id
        logging.info(f"Saved input prompt to DB with ID: {prompt_doc_id}")
    except Exception as db_err:
        logging.error(f"Failed to save input prompt to MongoDB: {db_err}")
        prompt_doc_id = None

    # Define the Prompt for Gemini
    prompt = f"""
    Analyze the following text in detail and generate a comprehensive documentation report suitable for research and analysis. Structure the report clearly.

    Input Text:
    ---
    {input_text}
    ---

    Generate a report with the following sections:

    1.  **## Summary:** Provide a concise overview of the main topic and purpose of the text.
    2.  **## Key Findings:** Extract the most important facts, conclusions, or data points presented. Use bullet points.
    3.  **## Detailed Analysis:** Elaborate on the key findings. Discuss themes, arguments, potential implications, or underlying patterns. Identify any conflicting information or areas needing further investigation.
    4.  **## Sentiment Analysis:** Briefly describe the overall sentiment (positive, negative, neutral) or tone of the text, providing justification.
    5.  **## Potential Keywords:** List the 5-10 most relevant keywords or key phrases.
    6.  **## Conclusion:** Summarize the analysis and offer concluding thoughts or potential next steps based on the text.

    **IMPORTANT - Data for Visualization:**
    After the report, include a section strictly formatted as follows, containing data suggestions for charts. Use JSON format within this section. If no quantifiable data is apparent, provide an empty JSON object like {{}}. Do NOT add any text explanation around the JSON block itself.

    ```json_chart_data
    {{
      "keyword_frequencies": {{
        "keyword1": count1,
        "keyword2": count2,
        "keyword3": count3,
        "keyword4": count4,
        "keyword5": count5
      }},
      "sentiment_score": {{
        "positive": value,
        "negative": value,
        "neutral": value
      }}
    }}
    ```

    Report:
    ---
    """ # Ensure f-string is closed correctly

    generated_text = ""; response = None
    try:
        logging.info("/generate_report: Sending request to Gemini...")
        response = model.generate_content(prompt)
        logging.info("/generate_report: Received response from Gemini.")

        # Validate Gemini Response
        if not response.candidates:
             logging.error("/generate_report: Gemini response has no candidates.")
             if hasattr(response, 'prompt_feedback'): logging.error(f"Prompt Feedback: {response.prompt_feedback}")
             return jsonify({"error": "AI response was empty or blocked..."}), 500
        first_candidate = response.candidates[0]; finish_reason_name = first_candidate.finish_reason.name
        logging.info(f"/generate_report: Gemini finish reason: {finish_reason_name}")
        if finish_reason_name not in ["STOP", "MAX_TOKENS"]:
            logging.warning(f"/generate_report: Generation finished unexpectedly: {finish_reason_name}")
            if hasattr(response, 'prompt_feedback'): logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
            return jsonify({"error": f"AI generation stopped unexpectedly ({finish_reason_name})..."}), 500
        if not hasattr(first_candidate, 'content') or not hasattr(first_candidate.content, 'parts') or not first_candidate.content.parts:
             logging.error("/generate_report: Gemini response candidate has no content parts.")
             return jsonify({"error": "AI response content is missing."}), 500
        generated_text = first_candidate.content.parts[0].text
        logging.info(f"/generate_report: Extracted text (length: {len(generated_text)}).")
        logging.info(f"/generate_report: ==== RAW GEMINI RESPONSE START ====\n{generated_text}\n==== RAW GEMINI RESPONSE END ====")

        # Parse Response Content (Report + JSON)
        report_content = generated_text; chart_data_json = {}
        try:
            json_start_marker = "```json_chart_data"; json_end_marker = "```"
            start_index = generated_text.rfind(json_start_marker)
            if start_index != -1:
                 start_index_content = start_index + len(json_start_marker)
                 end_index = generated_text.find(json_end_marker, start_index_content)
                 if end_index != -1:
                     json_string = generated_text[start_index_content:end_index].strip()
                     try: chart_data_json = json.loads(json_string); report_content = generated_text[:start_index].strip()
                     except json.JSONDecodeError as json_err: logging.error(f"Failed to decode JSON: {json_err}"); report_content = generated_text
                 else: report_content = generated_text
            else: report_content = generated_text
        except Exception as parse_err: logging.error(f"Error parsing AI response: {parse_err}"); report_content = generated_text

        # 2. Save Documentation to MongoDB
        documentation_doc_id = None
        try:
            doc_to_save = {
                "input_prompt_id": prompt_doc_id,
                "report_html": report_content,
                "chart_data": chart_data_json,
                "timestamp": datetime.utcnow(),
                "model_used": model_name,
                "finish_reason": finish_reason_name
            }
            insert_result = documentation_collection.insert_one(doc_to_save)
            documentation_doc_id = insert_result.inserted_id
            logging.info(f"Saved documentation to DB with ID: {documentation_doc_id}")
            # Update the original prompt doc
            if prompt_doc_id is not None:
                input_prompts_collection.update_one({"_id": prompt_doc_id}, {"$set": {"related_documentation_id": documentation_doc_id}})
                logging.info(f"Updated input prompt {prompt_doc_id} with doc ID {documentation_doc_id}")
        except Exception as db_err:
            logging.error(f"Failed to save documentation to MongoDB: {db_err}")
            documentation_doc_id = None

        # Prepare response for client
        report_context_for_chat = report_content[:3000] if report_content else ""
        logging.info("/generate_report: Successfully processed request. Sending response.")
        return jsonify({
            "report_html": report_content,
            "chart_data": chart_data_json,
            "report_context_for_chat": report_context_for_chat,
            "documentation_id": str(documentation_doc_id) if documentation_doc_id else None
        })

    except Exception as e:
        logging.error(f"CRITICAL ERROR processing /generate_report: {e}")
        logging.error(traceback.format_exc())
        if response and hasattr(response, 'prompt_feedback'): logging.warning(f"Prompt Feedback on Error: {response.prompt_feedback}")
        return jsonify({"error": f"An unexpected server error occurred..."}), 500

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    logging.info(f"Client connected successfully: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    logging.info(f"Client disconnected event received for: {sid}")

@socketio.on('send_message')
def handle_send_message(data):
    sid = request.sid
    # *** CORRECTED DB CHECK: Using 'is None' ***
    if db is None:
         logging.error(f"DB connection unavailable for message from {sid}")
         emit('error', {'message': 'Database connection is unavailable. Cannot process message.'}, room=sid)
         return
    # ******************************************

    if not isinstance(data, dict):
        logging.warning(f"Received non-dict data for 'send_message' from {sid}: {data}")
        emit('error', {'message': 'Invalid message format received.'}, room=sid)
        return

    user_message = data.get('text')
    documentation_id_str = data.get('documentation_id')

    if not user_message or not isinstance(user_message, str) or len(user_message.strip()) == 0:
        emit('error', {'message': 'Cannot send empty message.'}, room=sid)
        return
    if not documentation_id_str:
         emit('error', {'message': 'Cannot process chat message without associated documentation ID.'}, room=sid)
         return
    try:
        documentation_id = ObjectId(documentation_id_str)
    except Exception:
         emit('error', {'message': 'Invalid documentation ID format.'}, room=sid)
         return

    logging.info(f"Received message for doc {documentation_id} from {sid}: '{user_message[:50]}...'")

    # Save User Message to Chat Collection
    try:
        user_message_doc = {"role": "user", "text": user_message, "timestamp": datetime.utcnow()}
        chats_collection.update_one(
            {"documentation_id": documentation_id},
            {"$push": {"messages": user_message_doc},
             "$setOnInsert": {"documentation_id": documentation_id, "start_timestamp": datetime.utcnow(), "initial_sid": sid}},
            upsert=True
        )
        logging.info(f"Saved user message for doc {documentation_id} to DB.")
    except Exception as db_err:
        logging.error(f"Failed to save user message for doc {documentation_id} to MongoDB: {db_err}")

    # Get AI Response (rebuilding history from DB)
    ai_response_text = "[Error: Could not get AI response]"
    try:
        emit('typing_indicator', {'isTyping': True}, room=sid)
        logging.info(f"Querying Gemini for doc {documentation_id}, SID {sid}...")

        # Rebuild history from DB for this message
        chat_history_from_db = []
        initial_context_needed = True
        chat_doc = chats_collection.find_one({"documentation_id": documentation_id})
        if chat_doc and "messages" in chat_doc and len(chat_doc["messages"]) > 0:
             initial_context_needed = False
             for msg in chat_doc["messages"]:
                 api_role = 'model' if msg['role'] == 'AI' else msg['role']
                 chat_history_from_db.append({'role': api_role, 'parts': [msg['text']]})

        if initial_context_needed: # Add initial report context if no DB history found
             doc_data = documentation_collection.find_one({"_id": documentation_id})
             if doc_data and "report_html" in doc_data:
                 report_context = doc_data["report_html"][:3000]
                 if report_context:
                      chat_history_from_db.append({'role': 'user', 'parts': [f"This is the report we are discussing:\n\n{report_context}"]})
                      chat_history_from_db.append({'role': 'model', 'parts': ["Understood. I have the report context. How can I assist you with it?"]})

        # Start a temporary chat session
        if model:
            temp_chat = model.start_chat(history=chat_history_from_db)
            response = temp_chat.send_message(user_message)
            logging.info(f"Received chat response from Gemini for doc {documentation_id}, SID {sid}.")
            # Validate Gemini response
            if response and hasattr(response, 'candidates') and response.candidates:
                 first_candidate = response.candidates[0]
                 if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts') and first_candidate.content.parts:
                     finish_reason_name = first_candidate.finish_reason.name
                     if finish_reason_name not in ["STOP", "MAX_TOKENS"]: ai_response_text = f"[AI response may be incomplete...]"
                     else: ai_response_text = response.text if hasattr(response, 'text') else first_candidate.content.parts[0].text
                 else: ai_response_text = "[Error: AI response content missing parts]"
            else: ai_response_text = "[Error: AI response was empty or blocked]"
        else:
             ai_response_text = "[Error: AI Model not available]"

        # Save AI Message to Chat Collection (if not an error)
        if not ai_response_text.startswith("[Error:"):
            try:
                ai_message_doc = {"role": "AI", "text": ai_response_text, "timestamp": datetime.utcnow()}
                chats_collection.update_one({"documentation_id": documentation_id}, {"$push": {"messages": ai_message_doc}})
                logging.info(f"Saved AI response for doc {documentation_id} to DB.")
            except Exception as db_err:
                logging.error(f"Failed to save AI response for doc {documentation_id} to MongoDB: {db_err}")
        else: logging.warning(f"Skipping DB save for AI error response...")

        # Send AI response back to client
        logging.info(f"Sending AI response to {sid}: '{ai_response_text[:50]}...'")
        emit('receive_message', {'user': 'AI', 'text': ai_response_text}, room=sid)

    except Exception as e:
        logging.error(f"Error processing message for doc {documentation_id}, SID {sid}: {e}")
        logging.error(traceback.format_exc())
        emit('error', {'message': f'An server error occurred while communicating with the AI.'}, room=sid)
    finally:
        emit('typing_indicator', {'isTyping': False}, room=sid)


# --- Main Execution ---
if __name__ == '__main__':
    if db is None: # Use 'is None' here too for consistency
         logging.critical("MongoDB connection failed during startup. Aborting.")
         exit(1) # Exit if DB connection failed
    logging.info("Starting Flask-SocketIO server...")
    try:
        # Run on port 5000
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except ValueError as ve:
        if 'Invalid async_mode specified' in str(ve): logging.critical("ASYNC MODE ERROR: 'eventlet' required..."); logging.critical("Please install it: pip install eventlet")
        else: logging.error(f"Error starting server: {ve}"); logging.error(traceback.format_exc())
    except OSError as oe:
         if "Address already in use" in str(oe): logging.critical(f"PORT ERROR: Port 5000 is already in use...")
         else: logging.critical(f"Failed to start server due to OS Error: {oe}"); logging.critical(traceback.format_exc())
    except Exception as e:
        logging.critical(f"Failed to start server: {e}"); logging.critical(traceback.format_exc())