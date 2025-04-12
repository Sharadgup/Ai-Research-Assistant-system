# --- START: Apply Eventlet Monkey Patching ---
# This MUST be one of the very first things in your script
import eventlet
eventlet.monkey_patch()
# --- END: Apply Eventlet Monkey Patching ---

import os
import google.generativeai as genai
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session) # Standard Flask imports
from flask_socketio import SocketIO, emit, send # Keep send if used by client, else remove
from dotenv import load_dotenv
import logging
import json
from threading import Lock # May not be needed with DB approach but keep for now
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, DuplicateKeyError
from datetime import datetime
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash # For passwords

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# Load environment variables
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__,
            template_folder='src/templates',
            static_folder='src/static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
if not app.config['SECRET_KEY']:
    logging.critical("CRITICAL: FLASK_SECRET_KEY not set.")
    app.config['SECRET_KEY'] = 'dev-secret-key-only-not-for-production!'


# --- MongoDB Initialization ---
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
db = None
input_prompts_collection = None
documentation_collection = None
chats_collection = None         # For report-specific chats
registrations_collection = None
general_chats_collection = None # <-- Collection for dashboard chat

if not MONGO_URI: logging.critical("CRITICAL: MONGODB_URI not found.")
elif not DB_NAME: logging.critical("CRITICAL: MONGODB_DB_NAME not found.")
else:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        logging.info("Attempting to connect to MongoDB...")
        mongo_client.admin.command('ismaster') # Test connection
        db = mongo_client[DB_NAME]
        logging.info(f"Connected to MongoDB server. Selecting database: '{DB_NAME}'")

        # Initialize All Collections
        input_prompts_collection = db["input_prompts"]
        documentation_collection = db["documentation"]
        chats_collection = db["chats"]
        registrations_collection = db["registrations"]
        general_chats_collection = db["general_chats"] # <-- Define general chat collection object

        logging.info("MongoDB Collections assigned.")

        # Create unique index for username
        try: registrations_collection.create_index("username", unique=True); logging.info("Ensured unique index on 'username'.")
        except Exception as index_err: logging.warning(f"Username index warn: {index_err}")
        # Create index for general chats
        try: general_chats_collection.create_index("user_id", unique=True); logging.info("Ensured unique index on 'user_id'.")
        except Exception as index_err: logging.warning(f"General chat index warn: {index_err}")

        logging.info(f"Successfully configured MongoDB. Database: '{DB_NAME}'")
    except (ConnectionFailure, ConfigurationError) as ce: logging.critical(f"MongoDB connection/config failed: {ce}"); db = None
    except Exception as e: logging.critical(f"MongoDB initialization error: {e}"); logging.error(traceback.format_exc()); db = None
# --------------------------------------------------------


# --- SocketIO Initialization ---
allowed_origins_list = [
    "http://127.0.0.1:5000", "http://localhost:5000",
    "https://5000-idx-ai-note-system-1744087101492.cluster-a3grjzek65cxex762e4mwrzl46.cloudworkstations.dev", "*"
]
socketio = SocketIO( app, async_mode='eventlet', cors_allowed_origins=allowed_origins_list,
                   ping_timeout=20, ping_interval=10 )

# --- Gemini API Configuration ---
api_key = os.getenv("GEMINI_API_KEY"); model_name = "gemini-1.5-flash"; model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"}for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        model=genai.GenerativeModel(model_name,safety_settings=safety_settings);logging.info(f"Gemini model '{model_name}' init.")
    except Exception as e:logging.error(f"Error init Gemini: {e}")
else:logging.warning("GEMINI_API_KEY not found.")

# --- Authentication Helper ---
def is_logged_in(): return 'user_id' in session

# --- HTTP Routes ---
@app.route('/')
def landing_page():
    return render_template('landing.html', now=datetime.utcnow())

@app.route('/register', methods=['GET', 'POST'])
def register():
    if is_logged_in(): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if db is None: flash("DB error.", "danger"); return render_template('register.html', now=datetime.utcnow())
        username=request.form.get('username','').strip(); password=request.form.get('password',''); confirm=request.form.get('confirm_password','')
        error=None;
        if not username: error="Username required."
        elif not password: error="Password required."
        elif password!=confirm: error="Passwords don't match."
        elif len(password)<6: error="Password min 6 chars."
        if error: flash(error, "warning"); return render_template('register.html', username=username, now=datetime.utcnow())
        hash_val=generate_password_hash(password)
        try: registrations_collection.insert_one({"username": username, "password_hash": hash_val, "created_at": datetime.utcnow()}); flash("Registered!", "success"); return redirect(url_for('login'))
        except DuplicateKeyError: flash("Username exists.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow())
        except Exception as e: logging.error(f"Reg error: {e}"); flash("Registration error.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow())
    return render_template('register.html', now=datetime.utcnow())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in(): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if db is None: flash("DB error.", "danger"); return render_template('login.html', now=datetime.utcnow())
        username=request.form.get('username','').strip(); password=request.form.get('password','')
        if not username or not password: flash("All fields required.", "warning"); return render_template('login.html', username=username, now=datetime.utcnow())
        try:
            user=registrations_collection.find_one({"username": username})
            if user and check_password_hash(user.get('password_hash', ''), password):
                session.clear(); session['user_id'] = str(user['_id']); session['username'] = user['username']; return redirect(url_for('dashboard'))
            else: flash("Invalid credentials.", "danger"); return render_template('login.html', username=username, now=datetime.utcnow())
        except Exception as e: logging.error(f"Login error: {e}"); flash("Login error.", "danger"); return render_template('login.html', username=username, now=datetime.utcnow())
    return render_template('login.html', now=datetime.utcnow())

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown'); session.clear(); flash("Logged out.", "success"); logging.info(f"User '{username}' logged out."); return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    username=session.get('username','User')
    available_models=["Gemini 1.5 Flash", "Gemini Pro"]; usable_models=["Gemini 1.5 Flash"]
    sectors=["Healthcare", "Finance", "Technology", "Education", "Retail", "General"]
    apps=[{"id":"tts","name":"Text-to-Speech","description":"..."},{"id":"ttv","name":"Text-to-Video","description":"..."}]
    services=["Report Analysis", "Visualization", "Chat", "PDF Export"]
    dashboard_data={"username": username, "services": services, "available_models": available_models, "usable_models": usable_models, "sectors": sectors, "apps": apps}
    return render_template('dashboard.html', data=dashboard_data, now=datetime.utcnow())

@app.route('/index')
def report_page():
    return render_template('index.html', now=datetime.utcnow())

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    logging.info("Received request for /generate_report")
    if not model: return jsonify({"error": "AI unavailable."}), 503
    if db is None: return jsonify({"error": "DB unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); input_text = data.get('text')
    if not input_text: return jsonify({"error": "No text provided"}), 400
    logging.info(f"/generate_report: Processing text (length: {len(input_text)}).")

    prompt_doc_id = None; user_id_for_doc = None
    if is_logged_in():
        try: user_id_for_doc = ObjectId(session['user_id'])
        except: logging.warning("Could not get user_id ObjectId.")
    try: # Save Input Prompt
        prompt_doc = {"original_text": input_text, "timestamp": datetime.utcnow(), "user_id": user_id_for_doc}
        prompt_doc_id = input_prompts_collection.insert_one(prompt_doc).inserted_id
    except Exception as db_err: logging.error(f"Failed save input prompt: {db_err}")

    prompt = f"""Analyze...\nInput Text:\n---\n{input_text}\n---\nGenerate report...\n(Your full prompt here)\nReport:\n---""" # Keep your prompt

    generated_text = ""; response = None
    try: # Call Gemini
        response = model.generate_content(prompt)
        if not response.candidates: raise ValueError("AI response empty/blocked.")
        generated_text = response.text
        logging.info(f"/generate_report: Gemini success.")
        # Parse JSON data
        report_content = generated_text; chart_data_json = {}
        try: # Parse JSON
             json_start_marker="```json_chart_data"; json_end_marker="```"; start_index=generated_text.rfind(json_start_marker)
             if start_index!=-1:
                 end_index=generated_text.find(json_end_marker, start_index+len(json_start_marker))
                 if end_index!=-1:
                     json_string=generated_text[start_index+len(json_start_marker):end_index].strip()
                     try: chart_data_json=json.loads(json_string); report_content=generated_text[:start_index].strip()
                     except Exception as json_e: logging.error(f"JSON Parse Error: {json_e}")
        except Exception as parse_e: logging.error(f"Parsing Error: {parse_e}")

        # Save Documentation
        documentation_doc_id = None
        try:
            doc_to_save = { "input_prompt_id": prompt_doc_id, "user_id": user_id_for_doc, "report_html": report_content,
                            "chart_data": chart_data_json, "timestamp": datetime.utcnow(), "model_used": model_name,
                            "finish_reason": response.candidates[0].finish_reason.name if response.candidates else 'UNKNOWN' }
            documentation_doc_id = documentation_collection.insert_one(doc_to_save).inserted_id
            if prompt_doc_id: input_prompts_collection.update_one({"_id": prompt_doc_id}, {"$set": {"related_documentation_id": documentation_doc_id}})
        except Exception as db_err: logging.error(f"Failed save documentation: {db_err}")

        return jsonify({ "report_html": report_content, "chart_data": chart_data_json,
                         "report_context_for_chat": report_content[:3000],
                         "documentation_id": str(documentation_doc_id) if documentation_doc_id else None })
    except Exception as e:
        logging.error(f"ERROR processing /generate_report: {e}"); logging.error(traceback.format_exc())
        return jsonify({"error": f"Server error during report generation."}), 500


# --- SocketIO Event Handlers ---

# == Default Namespace (Report Chat) ==
@socketio.on('connect')
def handle_connect(): logging.info(f"(Report Chat) Client connected: {request.sid}")
@socketio.on('disconnect')
def handle_disconnect(): logging.info(f"(Report Chat) Client disconnected: {request.sid}")
@socketio.on('send_message')
def handle_send_message(data):
    # (Keep existing detailed logic)
    sid = request.sid; logging.info(f"--- handle_send_message START (SID: {sid}) ---")
    if db is None: emit('error', {'message': 'DB unavailable.'}, room=sid); logging.info(f"--- END (SID: {sid}) ---"); return
    logging.debug(f"(SID: {sid}) Received data: {data}")
    if not isinstance(data, dict): emit('error', {'message': 'Invalid format.'}, room=sid); logging.info(f"--- END (SID: {sid}) ---"); return
    user_message=data.get('text'); doc_id_str=data.get('documentation_id')
    logging.info(f"(SID: {sid}) Extracted text: '{user_message[:50]}...', doc_id_str: {doc_id_str}")
    if not user_message or not doc_id_str: emit('error', {'message': 'Missing data.'}, room=sid); logging.info(f"--- END (SID: {sid}) ---"); return
    try: doc_id=ObjectId(doc_id_str)
    except Exception as e: logging.error(f"(SID: {sid}) Invalid doc_id format: {doc_id_str}. Error: {e}"); emit('error', {'message': 'Invalid ID.'}, room=sid); logging.info(f"--- END (SID: {sid}) ---"); return
    logging.info(f"(SID: {sid}) Processing msg for doc {doc_id}")
    try: chats_collection.update_one({"documentation_id": doc_id}, {"$push": {"messages": {"role": "user", "text": user_message, "timestamp": datetime.utcnow()}}}, upsert=True); logging.info(f"(SID: {sid}) Saved user msg")
    except Exception as e: logging.error(f"(SID: {sid}) Failed save user msg: {e}")
    ai_response = "[AI Error]";
    try:
        emit('typing_indicator', {'isTyping': True}, room=sid)
        history = []; chat_doc = chats_collection.find_one({"documentation_id": doc_id})
        if chat_doc and "messages" in chat_doc: # Rebuild history
             for msg in chat_doc["messages"]: history.append({'role': ('model' if msg['role']=='AI' else msg['role']), 'parts': [msg['text']]})
        else: # Add initial context
            doc_data = documentation_collection.find_one({"_id": doc_id})
            if doc_data and "report_html" in doc_data: history.extend([{'role': 'user', 'parts': [f"Report:\n{doc_data['report_html'][:3000]}"]}, {'role': 'model', 'parts': ["OK."]}])
        if model: # Call Gemini
            temp_chat = model.start_chat(history=history); response = temp_chat.send_message(user_message)
            if response.candidates: ai_response = response.text
            else: logging.error("Gemini no candidates")
        else: logging.error("Gemini model unavailable")
        if not ai_response.startswith("[AI Error"): # Save AI Msg
            try: chats_collection.update_one({"documentation_id": doc_id}, {"$push": {"messages": {"role": "AI", "text": ai_response, "timestamp": datetime.utcnow()}}}); logging.info(f"(SID: {sid}) Saved AI response")
            except Exception as e: logging.error(f"Failed save AI msg: {e}")
        emit('receive_message', {'user': 'AI', 'text': ai_response}, room=sid) # Emit to client
    except Exception as e: logging.error(f"Error in report chat: {e}"); emit('error', {'message': 'Server error.'}, room=sid)
    finally: emit('typing_indicator', {'isTyping': False}, room=sid); logging.info(f"--- handle_send_message END (SID: {sid}) ---")

# == Dashboard Namespace (/dashboard_chat) ==
@socketio.on('connect', namespace='/dashboard_chat')
def handle_dashboard_connect():
    if not is_logged_in(): logging.warning(f"Unauth connect /dashboard_chat: {request.sid}"); return False
    logging.info(f"User '{session.get('username')}' connected dashboard chat: {request.sid}")
@socketio.on('disconnect', namespace='/dashboard_chat')
def handle_dashboard_disconnect():
    logging.info(f"User '{session.get('username', 'Unknown')}' disconnected dashboard chat: {request.sid}")

# --- CORRECTED Dashboard Chat Handler with DB saving & Checks ---
@socketio.on('send_dashboard_message', namespace='/dashboard_chat')
def handle_dashboard_chat(data):
    """Handles general chat messages, saves to DB, gets AI response."""
    sid = request.sid
    logging.debug(f"--- handle_dashboard_chat START (SID: {sid}) ---")

    # 1. Authentication Check
    if not is_logged_in():
         logging.warning(f"(Dashboard Chat SID: {sid}) Unauthenticated message. Ignoring.")
         emit('error', {'message': 'Authentication required.'}, room=sid, namespace='/dashboard_chat')
         logging.debug(f"--- handle_dashboard_chat END (SID: {sid}) ---")
         return

    # 2. Database/Collection Availability Check
    # *** Use 'is None' for the collection check ***
    if db is None or general_chats_collection is None:
         logging.error(f"(Dashboard Chat SID: {sid}) DB or general_chats_collection is None. DB: {db is not None}, Collection: {general_chats_collection is not None}")
         emit('error', {'message': 'Chat history database service unavailable.'}, room=sid, namespace='/dashboard_chat')
         logging.debug(f"--- handle_dashboard_chat END (SID: {sid}) ---")
         return
    # ********************************************

    # 3. User Info & Input Validation
    username = session.get('username'); user_id_str = session.get('user_id')
    if not username or not user_id_str:
        logging.error(f"(Dashboard Chat SID: {sid}) Missing username or user_id in session.")
        emit('error', {'message': 'Session error. Please log in again.'}, room=sid, namespace='/dashboard_chat')
        logging.debug(f"--- handle_dashboard_chat END (SID: {sid}) ---")
        return
    try: user_id_object = ObjectId(user_id_str)
    except Exception as e:
         logging.error(f"(Dashboard Chat SID: {sid}) Invalid session user_id ('{user_id_str}') for {username}: {e}")
         emit('error', {'message': 'Internal session error.'}, room=sid, namespace='/dashboard_chat')
         logging.debug(f"--- handle_dashboard_chat END (SID: {sid}) ---")
         return
    if not isinstance(data, dict): logging.warning(f"Non-dict chat data from {username}"); return
    user_message = data.get('text', '').strip();
    if not user_message: logging.debug(f"Empty msg from {username}"); return
    logging.info(f"Dashboard Chat from {username}: '{user_message[:50]}...'")

    # --- 4. Save User Message ---
    try:
        user_message_doc = {"role": "user", "text": user_message, "timestamp": datetime.utcnow()}
        update_result = general_chats_collection.update_one({"user_id": user_id_object},{"$push": {"messages": user_message_doc},"$setOnInsert": {"user_id": user_id_object, "username": username, "start_timestamp": datetime.utcnow()}},upsert=True)
        logging.info(f"(Dash Chat SID: {sid}) User msg save result: {update_result.raw_result}")
    except Exception as db_err: logging.error(f"Failed save general chat user msg: {db_err}"); logging.error(traceback.format_exc())

    # --- 5. Get AI Response (with history) ---
    ai_response_text = "[Error: AI processing failed]"
    try:
        emit('typing_indicator', {'isTyping': True}, room=sid, namespace='/dashboard_chat')
        logging.info(f"(Dashboard Chat SID: {sid}) Querying Gemini for {username}...")
        chat_history_from_db = [] # Rebuild history
        logging.debug(f"(Dashboard Chat SID: {sid}) Fetching history user {user_id_object}...")
        chat_doc = general_chats_collection.find_one({"user_id": user_id_object})
        if chat_doc and "messages" in chat_doc:
             recent_messages = chat_doc["messages"] # Or limit: [-10:]
             for msg in recent_messages: chat_history_from_db.append({'role': ('model' if msg['role']=='AI' else msg['role']), 'parts': [msg['text']]})
             logging.info(f"(Dashboard Chat SID: {sid}) Rebuilt history ({len(chat_history_from_db)} msgs).")
        else: logging.info(f"(Dashboard Chat SID: {sid}) No history found.")

        if model: # Call Gemini
            temp_chat = model.start_chat(history=chat_history_from_db); response = temp_chat.send_message(user_message)
            logging.info(f"(Dashboard Chat SID: {sid}) Got Gemini response.")
            if response and response.candidates: ai_response_text = response.text
            else: ai_response_text = "[Error: AI response empty/blocked]"; logging.error(f"AI response invalid. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        else: ai_response_text = "[Error: AI model unavailable]"; logging.error(f"(SID: {sid}) AI Model is None.")

        # --- 6. Save AI Message ---
        # *** CORRECTED COLLECTION CHECK: Using 'is not None' ***
        if not ai_response_text.startswith("[Error:") and general_chats_collection is not None:
        # ********************************************************
            try:
                ai_message_doc = {"role": "AI", "text": ai_response_text, "timestamp": datetime.utcnow()}
                update_result_ai = general_chats_collection.update_one({"user_id": user_id_object},{"$push": {"messages": ai_message_doc}})
                logging.info(f"(Dash Chat SID: {sid}) AI msg save result: {update_result_ai.raw_result}")
                if not update_result_ai.acknowledged or update_result_ai.matched_count == 0: logging.warning(f"DB did not ack/match AI msg save for {username}.")
            except Exception as db_err: logging.error(f"Failed save general AI response: {db_err}"); logging.error(traceback.format_exc())
        elif general_chats_collection is None: logging.error(f"(Dash Chat SID: {sid}) Cannot save AI rsp, collection is None!")
        else: logging.warning(f"(Dash Chat SID: {sid}) Skipping save for AI error: '{ai_response_text}'")

        # --- 7. Emit AI response back to client ---
        logging.info(f"(Dashboard Chat SID: {sid}) Emitting 'receive_dashboard_message' to {username}: '{ai_response_text[:50]}...'")
        emit('receive_dashboard_message', {'user': 'AI', 'text': ai_response_text}, room=sid, namespace='/dashboard_chat')

    except Exception as e:
        logging.error(f"(Dashboard Chat SID: {sid}) Error processing dashboard message: {e}")
        logging.error(traceback.format_exc())
        emit('error', {'message': f'Server error during chat processing.'}, room=sid, namespace='/dashboard_chat')
    finally:
        emit('typing_indicator', {'isTyping': False}, room=sid, namespace='/dashboard_chat')
        logging.debug(f"--- handle_dashboard_chat END (SID: {sid}) ---")

# --- Main Execution ---
if __name__ == '__main__':
    if db is None: logging.critical("MongoDB connection failed. Aborting."); exit(1)
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'dev-secret-key-only-not-for-production!': logging.warning("WARNING: Running with insecure default FLASK_SECRET_KEY!")
    logging.info("Starting Flask-SocketIO server...")
    try:
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except ValueError as ve:
        if 'Invalid async_mode specified' in str(ve): logging.critical("ASYNC MODE ERROR: 'eventlet' required...")
        else: logging.error(f"Error starting server: {ve}"); logging.error(traceback.format_exc())
    except OSError as oe:
         if "Address already in use" in str(oe): logging.critical(f"PORT ERROR: Port 5000 is already in use...")
         else: logging.critical(f"Failed to start server due to OS Error: {oe}"); logging.critical(traceback.format_exc())
    except Exception as e:
        logging.critical(f"Failed to start server: {e}"); logging.critical(traceback.format_exc())