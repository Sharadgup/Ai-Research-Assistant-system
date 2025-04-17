# --- START: Apply Eventlet Monkey Patching ---
# This MUST be one of the very first things in your script
import eventlet
eventlet.monkey_patch()
# --- END: Apply Eventlet Monkey Patching ---

import os
import google.generativeai as genai
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session, Blueprint,
                   send_from_directory) # Added send_from_directory
from flask_socketio import SocketIO, emit # Removed send as it might conflict
from dotenv import load_dotenv
import logging
import json
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, DuplicateKeyError
from datetime import datetime
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename # For file uploads
import fitz # PyMuPDF library

# --- Flask-Dance Imports (Only Google) ---
from flask_dance.contrib.google import make_google_blueprint, google

# *** ADD ProxyFix Import ***
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# Load environment variables
load_dotenv()

# --- Constants ---
UPLOAD_FOLDER = 'uploads' # Create this directory in your project root
ALLOWED_EXTENSIONS = {'pdf'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload directory: {UPLOAD_FOLDER}")

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

# *** APPLY ProxyFix Middleware FIRST ***
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# ***************************************

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-only-not-for-production!')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Comment out PREFERRED_URL_SCHEME as we are forcing the redirect_uri now
# app.config['PREFERRED_URL_SCHEME'] = 'https'

# --- OAuth Config (Google Only) ---
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = os.getenv('OAUTHLIB_INSECURE_TRANSPORT', '0')
app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
google_enabled = bool(app.config["GOOGLE_OAUTH_CLIENT_ID"] and app.config["GOOGLE_OAUTH_CLIENT_SECRET"])
if not google_enabled: logging.warning("Google OAuth credentials missing.")

# --- Create OAuth Blueprints (Google Only with Forced redirect_uri) ---
if google_enabled:
    try:
        forced_redirect_uri = "https://5000-idx-ai-note-system-1744087101492.cluster-a3grjzek65cxex762e4mwrzl46.cloudworkstations.dev/login/google/authorized"
        # forced_redirect_uri = "http://127.0.0.1:5000/login/google/authorized" # Local
        logging.info(f"Forcing Google redirect_uri to: {forced_redirect_uri}")
        google_bp = make_google_blueprint( scope=["openid", "email", "profile"], redirect_to="google_auth_callback",
                                           offline=False, redirect_uri=forced_redirect_uri )
        app.register_blueprint(google_bp, url_prefix="/login")
        logging.info("Google OAuth Blueprint registered.")
    except Exception as bp_error: logging.error(f"Failed Google Blueprint: {bp_error}"); google_enabled = False


# --- MongoDB Initialization ---
MONGO_URI = os.getenv("MONGODB_URI"); DB_NAME = os.getenv("MONGODB_DB_NAME")
db = None; registrations_collection = None; input_prompts_collection = None;
documentation_collection = None; chats_collection = None; general_chats_collection = None;
education_chats_collection = None; healthcare_chats_collection = None;
construction_agent_interactions_collection = None;
pdf_analysis_collection = None
pdf_chats_collection = None

if not MONGO_URI or not DB_NAME: logging.critical("Missing MongoDB Config")
else:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000); mongo_client.admin.command('ismaster')
        db = mongo_client[DB_NAME]; logging.info(f"MongoDB connected. DB: '{DB_NAME}'")
        # Define collections
        registrations_collection = db["registrations"]; input_prompts_collection = db["input_prompts"]
        documentation_collection = db["documentation"]; chats_collection = db["chats"]; general_chats_collection = db["general_chats"]
        education_chats_collection = db["education_chats"]; healthcare_chats_collection = db["healthcare_chats"]
        construction_agent_interactions_collection = db["construction_agent_interactions"]
        pdf_analysis_collection = db["pdf_analysis"]
        pdf_chats_collection = db["pdf_chats"]
        logging.info("MongoDB Collections assigned.")
        # Ensure Indexes...
        try: registrations_collection.create_index("username", unique=True, sparse=True)
        except: logging.warning("Username index issue")
        try: registrations_collection.create_index("email", unique=True, sparse=True)
        except: logging.warning("Email index issue")
        try: registrations_collection.create_index("google_id", unique=True, sparse=True)
        except: logging.warning("Google ID index issue")
        try: general_chats_collection.create_index("user_id", unique=True)
        except: logging.warning("General chat index issue")
        try: education_chats_collection.create_index("user_id")
        except Exception as idx_err: logging.warning(f"Edu chat index warn: {idx_err}")
        try: healthcare_chats_collection.create_index("user_id")
        except Exception as idx_err: logging.warning(f"Health chat index warn: {idx_err}")
        try: construction_agent_interactions_collection.create_index("user_id")
        except Exception as idx_err: logging.warning(f"Construction chat index warn: {idx_err}")
        try: pdf_analysis_collection.create_index("user_id")
        except Exception as idx_err: logging.warning(f"PDF Analysis index warn: {idx_err}")
        try: pdf_chats_collection.create_index("pdf_analysis_id")
        except Exception as idx_err: logging.warning(f"PDF Chat index warn: {idx_err}")

    except Exception as e: logging.critical(f"MongoDB init error: {e}"); db = None


# --- SocketIO Initialization ---
allowed_origins_list = ["http://127.0.0.1:5000", "http://localhost:5000", "https://5000-idx-ai-note-system-1744087101492.cluster-a3grjzek65cxex762e4mwrzl46.cloudworkstations.dev", "*"]
socketio = SocketIO( app, async_mode='eventlet', cors_allowed_origins=allowed_origins_list, ping_timeout=20, ping_interval=10 )

# --- Gemini API Configuration ---
api_key = os.getenv("GEMINI_API_KEY"); model_name = "gemini-1.5-flash"; model = None
if api_key:
    try: genai.configure(api_key=api_key); safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"}for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]; model=genai.GenerativeModel(model_name,safety_settings=safety_settings); logging.info(f"Gemini model '{model_name}' init.")
    except Exception as e:logging.error(f"Error init Gemini: {e}")
else:logging.warning("GEMINI_API_KEY not found.")


# --- Authentication Helpers ---
def is_logged_in(): return 'user_id' in session
def login_user(user_doc): session.clear(); session['user_id'] = str(user_doc['_id']); session['username'] = user_doc.get('username') or user_doc.get('name') or f"User_{str(user_doc['_id'])[:6]}"; session['login_method'] = user_doc.get('login_method', 'password'); logging.info(f"User '{session['username']}' logged in via {session['login_method']}.")

# --- File Upload Helper ---
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- PDF Text Extraction Helper ---
def extract_text_from_pdf(filepath):
    """Extracts all text from a PDF file using PyMuPDF."""
    try: doc = fitz.open(filepath); full_text = ""; num_pages = len(doc);
    except Exception as e: logging.error(f"Error opening PDF {filepath}: {e}"); logging.error(traceback.format_exc()); return None, 0 # Handle open error
    try:
        for page_num in range(num_pages): page = doc.load_page(page_num); full_text += page.get_text("text")
        doc.close(); logging.info(f"Extracted text (len: {len(full_text)}, pages: {num_pages})"); return full_text, num_pages
    except Exception as e: logging.error(f"Error extracting PDF text {filepath}: {e}"); logging.error(traceback.format_exc()); doc.close(); return None, 0 # Close doc even on error


# --- HTTP Routes ---
@app.route('/')
def landing_page(): return render_template('landing.html', now=datetime.utcnow(), google_login_enabled=google_enabled)

@app.route('/register', methods=['GET', 'POST'])
def register(): # (Keep logic)
    if is_logged_in(): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if db is None: flash("DB error.", "danger"); return render_template('register.html', now=datetime.utcnow(), google_login_enabled=google_enabled)
        username=request.form.get('username','').strip(); password=request.form.get('password',''); confirm=request.form.get('confirm_password','')
        error=None; # Basic validation...
        if not username or not password or password!=confirm or len(password)<6: flash("Invalid input.", "warning"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        hash_val=generate_password_hash(password)
        try: user_doc={"username":username, "password_hash":hash_val, "created_at":datetime.utcnow(), "login_method":"password"}; registrations_collection.insert_one(user_doc); flash("Registered!", "success"); return redirect(url_for('login'))
        except DuplicateKeyError: flash("Username exists.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        except Exception as e: logging.error(f"Reg error: {e}"); flash("Registration error.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
    return render_template('register.html', now=datetime.utcnow(), google_login_enabled=google_enabled)

@app.route('/login', methods=['GET', 'POST'])
def login(): # (Keep logic)
    if is_logged_in(): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if db is None: flash("DB error.", "danger"); return render_template('login.html', now=datetime.utcnow(), google_login_enabled=google_enabled)
        username=request.form.get('username','').strip(); password=request.form.get('password','')
        if not username or not password: flash("All fields required.", "warning"); return render_template('login.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        try:
            user=registrations_collection.find_one({"username": username})
            if user and user.get('password_hash') and check_password_hash(user['password_hash'], password): login_user(user); return redirect(url_for('dashboard'))
            else: flash("Invalid credentials.", "danger"); return render_template('login.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        except Exception as e: logging.error(f"Login error: {e}"); flash("Login error.", "danger"); return render_template('login.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
    return render_template('login.html', now=datetime.utcnow(), google_login_enabled=google_enabled)

@app.route("/google/authorized") # Keep Google Callback
def google_auth_callback(): # (Keep logic)
    if not google_enabled or not google.authorized: flash("Google login failed/disabled.", "danger"); return redirect(url_for("login"))
    try:
        resp = google.get("/oauth2/v3/userinfo");
        if not resp.ok: logging.error(f"Failed Google fetch:{resp.status_code}"); flash("Fetch Google info failed.", "danger"); return redirect(url_for("login"))
        user_info = resp.json(); google_id = user_info.get("sub"); email = user_info.get("email"); name = user_info.get("name")
        if not google_id: flash("No Google ID.", "danger"); return redirect(url_for("login"))
        user_doc = registrations_collection.find_one({"google_id": google_id})
        if not user_doc and email: user_doc = registrations_collection.find_one({"email": email})
        if user_doc: # Found user
            update_data={"$set":{"last_login_at":datetime.utcnow(),"login_method":"google"}};
            if not user_doc.get("google_id"):update_data["$set"]["google_id"]=google_id
            if not user_doc.get("name")and name:update_data["$set"]["name"]=name
            registrations_collection.update_one({"_id":user_doc["_id"]},update_data);user_doc=registrations_collection.find_one({"_id":user_doc["_id"]})
        else: # Create user
            user_data={"google_id":google_id,"email":email,"name":name,"created_at":datetime.utcnow(),"last_login_at":datetime.utcnow(),"login_method":"google"}
            user_doc=registrations_collection.find_one({"_id":registrations_collection.insert_one(user_data).inserted_id})
        if user_doc: login_user(user_doc); return redirect(url_for("dashboard"))
        else: flash("Login failed post-Google.", "danger"); return redirect(url_for("login"))
    except Exception as e: logging.error(f"Google callback error:{e}"); flash("Google login error.", "danger"); return redirect(url_for("login"))

@app.route('/logout') # Keep logout
def logout(): username = session.get('username', 'Unknown'); session.clear(); flash("Logged out.", "success"); logging.info(f"User '{username}' logged out."); return redirect(url_for('login'))

@app.route('/dashboard') # Keep dashboard
def dashboard():
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    username=session.get('username','User');
    available_models=["G 1.5 Flash", "G Pro"]; usable_models=["G 1.5 Flash"]; sectors=["Healthcare", "Finance", "Tech", "Edu", "Retail", "General"]; apps=[{"id":"tts","name":"TTS"},{"id":"ttv","name":"TTV"}]; services=["Analysis", "Viz", "Chat", "PDF"]
    dashboard_data={"username":username, "services":services, "available_models":available_models, "usable_models":usable_models, "sectors":sectors, "apps":apps}
    return render_template('dashboard.html', data=dashboard_data, now=datetime.utcnow())

@app.route('/index') # Keep report page route
def report_page(): return render_template('index.html', now=datetime.utcnow())

# --- Corrected `generate_report_route` Snippet ---
@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    logging.info("Req /generate_report")
    if not model: return jsonify({"error":"AI unavailable."}),503
    if db is None: return jsonify({"error":"DB unavailable."}),503
    if not request.is_json: return jsonify({"error":"Need JSON"}),400
    data=request.get_json(); input_text=data.get('text')
    if not input_text: return jsonify({"error":"No text"}),400

    prompt_doc_id=None; user_id=None
    if is_logged_in():
        try:user_id=ObjectId(session['user_id'])
        except:logging.warning("No valid user_id ObjectId")

    # Save Input Prompt
    try:
        prompt_doc={"original_text":input_text,"timestamp":datetime.utcnow(),"user_id":user_id}
        prompt_doc_id=input_prompts_collection.insert_one(prompt_doc).inserted_id
        logging.info(f"Saved input prompt ID: {prompt_doc_id}")
    except Exception as e:
        logging.error(f"Err save prompt:{e}")
        prompt_doc_id = None # Ensure it's None if save fails

    prompt=f"Analyze...\n{input_text}\nReport:\n---" # Use your full prompt

    generated_text = ""; response = None; report_content = None; chart_data = {}; doc_id = None
    try:
        # --- Call Gemini ---
        response=model.generate_content(prompt);
        if not response.candidates:raise ValueError("AI response empty/blocked.")
        gen_text=response.text;report_content=gen_text # Default content
        logging.info(f"/generate_report: Gemini success.")

        # --- Parse JSON ---
        try:
             json_start_marker="```json_chart_data"; json_end_marker="```"; start_index=gen_text.rfind(json_start_marker)
             if start_index!=-1:
                 end_index=gen_text.find(json_end_marker, start_index+len(json_start_marker))
                 if end_index!=-1:
                     json_string=gen_text[start_index+len(json_start_marker):end_index].strip()
                     try: chart_data=json.loads(json_string); report_content=gen_text[:start_index].strip()
                     except Exception as json_e: logging.error(f"JSON Parse Err: {json_e}")
        except Exception as parse_e: logging.error(f"Parse Err: {parse_e}")

        # --- Save Documentation ---
        try:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else 'UNKNOWN'
            doc_save={"input_prompt_id":prompt_doc_id,"user_id":user_id,"report_html":report_content,"chart_data":chart_data,"timestamp":datetime.utcnow(),"model_used":model_name,"finish_reason":finish_reason}
            doc_id=documentation_collection.insert_one(doc_save).inserted_id
            logging.info(f"Saved documentation ID: {doc_id}")
            if prompt_doc_id:
                input_prompts_collection.update_one({"_id":prompt_doc_id},{"$set":{"related_documentation_id":doc_id}})
                logging.info(f"Linked prompt {prompt_doc_id} to doc {doc_id}")

            # *** SUCCESS RETURN (only if save is successful) ***
            return jsonify({"report_html":report_content,"chart_data":chart_data,"report_context_for_chat":report_content[:3000],"documentation_id":str(doc_id) if doc_id else None})

        except Exception as e: # --- Handle DB Save Error ---
            logging.error(f"Err save doc:{e}")
            # Return the generated content anyway, but flag the DB error
            return jsonify({
                "error": "Report generated but failed to save to database.", # Add error message
                "report_html": report_content,
                "chart_data": chart_data,
                "report_context_for_chat": report_content[:3000],
                "documentation_id": None # No valid ID if save failed
                }), 200 # Return 200 OK because report was generated, but include error field

    except Exception as e: # --- Handle Gemini or Parsing Errors ---
        logging.error(f"ERROR gen report/parse: {e}");
        logging.error(traceback.format_exc()) # Log full traceback
        return jsonify({"error":"Server error during AI processing or report parsing."}), 500


# --- Agent Routes ---
@app.route('/education_agent')
def education_agent_page(): # Keep Education Page
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    return render_template('education_agent.html', now=datetime.utcnow())

@app.route('/education_agent_query', methods=['POST'])
def education_agent_query(): # Keep Education Query
    if not is_logged_in(): return jsonify({"error": "Auth required."}), 401
    if not model: return jsonify({"error": "AI unavailable."}), 503
    if db is None or education_chats_collection is None: return jsonify({"error": "DB unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Need JSON"}), 400
    data=request.get_json(); user_query=data.get('query','').strip(); username=session.get('username','User'); user_id_str=session.get('user_id')
    if not user_query or not user_id_str: return jsonify({"error":"Missing query/session."}), 400
    try: user_id=ObjectId(user_id_str)
    except: return jsonify({"error": "Session error."}), 500
    interaction_id = None
    try: doc={"user_id":user_id,"username":username,"query":user_query,"timestamp":datetime.utcnow(),"ai_answer":None}; interaction_id=education_chats_collection.insert_one(doc).inserted_id
    except Exception as e: logging.error(f"Err save edu query: {e}")
    prompt = f"Edu Assistant... Query: {user_query}\n Answer:"; ai_resp = "[AI Err]"
    try:
        response=model.generate_content(prompt)
        if response.candidates: ai_resp=response.text if response.text else "[AI empty]"
        else: ai_resp="[AI blocked/empty]"
        if interaction_id and not ai_resp.startswith("["):
            try: education_chats_collection.update_one({"_id":interaction_id},{"$set":{"ai_answer":ai_resp,"answered_at":datetime.utcnow()}})
            except Exception as e: logging.error(f"Err update edu answer: {e}")
        return jsonify({"answer": ai_resp })
    except Exception as e: logging.error(f"Err proc edu query: {e}"); return jsonify({"error": "Server error."}), 500

@app.route('/healthcare_agent')
def healthcare_agent_page(): # Keep Healthcare Page
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    return render_template('healthcare_agent.html', now=datetime.utcnow())

@app.route('/healthcare_agent_query', methods=['POST'])
def healthcare_agent_query(): # Keep Healthcare Query
    if not is_logged_in(): return jsonify({"error": "Auth required."}), 401
    if not model: return jsonify({"error": "AI unavailable."}), 503
    if db is None or healthcare_chats_collection is None: return jsonify({"error": "DB unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Need JSON"}), 400
    data=request.get_json(); user_query=data.get('query','').strip(); username=session.get('username','User'); user_id_str=session.get('user_id')
    if not user_query or not user_id_str: return jsonify({"error":"Missing query/session."}), 400
    try: user_id = ObjectId(user_id_str)
    except Exception as e: logging.error(f"Invalid user_id format: {e}"); return jsonify({"error": 'Session error.'}), 500
    interaction_id = None
    try: doc={"user_id":user_id,"username":username,"query":user_query,"timestamp":datetime.utcnow(),"ai_answer":None}; interaction_id=healthcare_chats_collection.insert_one(doc).inserted_id
    except Exception as e: logging.error(f"Err save health query: {e}")
    prompt = f"""Healthcare Info Assistant Disclaimer... Query: {user_query}\n Answer:""" # Use full prompt
    ai_resp = "[AI Err]"
    try:
        response=model.generate_content(prompt, safety_settings=safety_settings)
        if response.candidates: ai_resp=response.text if response.text else "[AI empty]"
        else: ai_resp="[AI blocked/empty]"; logging.error(f"Health AI blocked. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        if interaction_id and not ai_resp.startswith("["):
            try: healthcare_chats_collection.update_one({"_id":interaction_id},{"$set":{"ai_answer":ai_resp,"answered_at":datetime.utcnow()}})
            except Exception as e: logging.error(f"Err update health answer: {e}")
        return jsonify({"answer": ai_resp })
    except Exception as e: logging.error(f"Err proc health query: {e}"); return jsonify({"error": "Server error."}), 500

@app.route('/construction_agent')
def construction_agent_page(): # Keep Construction Page
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    return render_template('construction_agent.html', now=datetime.utcnow())

@app.route('/construction_agent_query', methods=['POST'])
def construction_agent_query(): # Keep Construction Query
    if not is_logged_in(): return jsonify({"error": "Auth required."}), 401
    if not model: return jsonify({"error": "AI unavailable."}), 503
    if db is None or construction_agent_interactions_collection is None: return jsonify({"error": "DB unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Need JSON"}), 400
    data = request.get_json(); user_query = data.get('query', '').strip(); data_context = data.get('context', '').strip()
    username = session.get('username', 'User'); user_id_str = session.get('user_id')
    if not user_query or not user_id_str: return jsonify({"error":"Missing query/session."}), 400
    try: user_id = ObjectId(user_id_str)
    except Exception as e: logging.error(f"Invalid user_id format: {e}"); return jsonify({"error": 'Session error.'}), 500
    interaction_id = None
    try:
        doc = { "user_id": user_id, "username": username, "query": user_query, "data_context": data_context, "timestamp": datetime.utcnow(), "ai_answer": None, "chart_data": None }
        interaction_id = construction_agent_interactions_collection.insert_one(doc).inserted_id
    except Exception as db_err: logging.error(f"Failed save construction query: {db_err}")
    prompt = f"""Constructive AI Agent... Context:\n{data_context if data_context else "N/A"}\n---\nQuery:\n{user_query}\n---\nIMPORTANT - Chart Data Request...\n```json_construction_chart_data...```\nAI Response:\n---""" # Keep full prompt
    ai_resp = "[AI Err]"; chart_data = {}
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        if response.candidates:
             raw_text = response.text; ai_resp = raw_text
             try: # Parse JSON
                 json_start_marker="```json_construction_chart_data"; json_end_marker="```"; start_index=raw_text.rfind(json_start_marker)
                 if start_index!=-1:
                     end_index=raw_text.find(json_end_marker, start_index+len(json_start_marker))
                     if end_index!=-1:
                         json_string=raw_text[start_index+len(json_start_marker):end_index].strip()
                         try: chart_data=json.loads(json_string); ai_resp=raw_text[:start_index].strip()
                         except Exception as json_e: logging.error(f"JSON Parse Err: {json_e}")
             except Exception as parse_e: logging.error(f"Parsing Err: {parse_e}")
        else: ai_resp="[AI blocked/empty]"; logging.error(f"Construction AI blocked. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        if interaction_id: # Update DB
            update_payload = {"$set": {"answered_at": datetime.utcnow()}}
            if not ai_resp.startswith("[AI Err"): update_payload["$set"]["ai_answer"] = ai_resp
            update_payload["$set"]["chart_data"] = chart_data
            try: construction_agent_interactions_collection.update_one({"_id":interaction_id}, update_payload)
            except Exception as e: logging.error(f"Err update construction answer: {e}")
        return jsonify({"answer": ai_resp, "chart_data": chart_data })
    except Exception as e: logging.error(f"Err proc construction query: {e}"); return jsonify({"error": "Server error."}), 500

@app.route('/pdf_analyzer') # Keep PDF Analyzer Page
def pdf_analyzer_page():
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    user_pdfs = [];
    if db is not None: # Safely check collections
        try: user_id_obj = ObjectId(session['user_id']); cursor = pdf_analysis_collection.find({"user_id": user_id_obj}, {"original_filename": 1, "upload_timestamp": 1, "_id": 1}).sort("upload_timestamp", -1).limit(10); user_pdfs = list(cursor);
        except Exception as e: logging.error(f"Error fetching user PDFs: {e}")
    for pdf in user_pdfs: pdf['_id'] = str(pdf['_id']) # Convert ID for template
    return render_template('pdf_analyzer.html', now=datetime.utcnow(), user_pdfs=user_pdfs)

@app.route('/upload_pdf', methods=['POST']) # Keep PDF Upload
def upload_pdf(): # (Keep logic)
    if not is_logged_in(): return jsonify({"error": "Auth required."}), 401
    if db is None or pdf_analysis_collection is None: return jsonify({"error": "DB unavailable."}), 503
    if 'pdfFile' not in request.files: return jsonify({"error": "No file part."}), 400
    file = request.files['pdfFile'];
    if file.filename == '': return jsonify({"error": "No file selected."}), 400
    if file and allowed_file(file.filename):
        try: user_id = ObjectId(session['user_id']); username = session.get('username', 'Unknown')
        except Exception: return jsonify({"error": "Invalid session."}), 401
        safe_name = secure_filename(file.filename); ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        filename = f"{user_id}_{ts}_{safe_name}"; filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath); logging.info(f"PDF saved: {filepath}")
            extracted_text, page_count = extract_text_from_pdf(filepath) # Capture page_count
            if extracted_text is None:
                if os.path.exists(filepath): os.remove(filepath); return jsonify({"error": "Failed text extraction."}), 500
            doc = {"user_id": user_id, "username": username, "original_filename": safe_name,"stored_filename": filename, "filepath": filepath, "page_count": page_count, "upload_timestamp": datetime.utcnow(), "extracted_text_preview": extracted_text[:1000], "full_text_extracted": True, "analysis_status": "extracted"}
            analysis_id = pdf_analysis_collection.insert_one(doc).inserted_id; logging.info(f"PDF record created ID: {analysis_id}")
            return jsonify({"message": "Success.", "analysis_id": str(analysis_id), "filename": safe_name, "text_preview": extracted_text[:3000] }), 200
        except Exception as e: logging.error(f"Error PDF upload: {e}");
        if 'filepath' in locals() and os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError as rm_err: logging.error(f"Failed to cleanup file: {filepath}. Error: {rm_err}")
        return jsonify({"error": "Server error processing file."}), 500
    else: return jsonify({"error": "Invalid file type."}), 400


# --- SocketIO Event Handlers ---
# (Keep all existing handlers: default namespace, /dashboard_chat, /pdf_chat)
# == Default Namespace (Report Chat) ==
@socketio.on('connect')
def handle_connect(): logging.info(f"(Report Chat) Connect: {request.sid}")
@socketio.on('disconnect')
def handle_disconnect(): logging.info(f"(Report Chat) Disconnect: {request.sid}")
@socketio.on('send_message')
def handle_send_message(data): # (Keep logic)
    sid=request.sid;logging.info(f"--- Report Chat Msg START (SID:{sid}) ---")
    if db is None:emit('error',{'message':'DB unavailable.'},room=sid);return
    user_msg=data.get('text');doc_id_str=data.get('documentation_id');
    if not user_msg or not doc_id_str:emit('error',{'message':'Missing data.'},room=sid);return
    try:doc_id=ObjectId(doc_id_str)
    except:emit('error',{'message':'Invalid ID.'},room=sid);return
    try:chats_collection.update_one({"documentation_id":doc_id},{"$push":{"messages":{"role":"user","text":user_msg,"timestamp":datetime.utcnow()}}},upsert=True)
    except Exception as e:logging.error(f"Err save user msg:{e}")
    ai_resp="[AI Err]";
    try:
        emit('typing_indicator',{'isTyping':True},room=sid)
        history=[];chat_doc=chats_collection.find_one({"documentation_id":doc_id})
        if chat_doc and"messages"in chat_doc:
            for msg in chat_doc["messages"]:history.append({'role':('model'if msg['role']=='AI'else msg['role']),'parts':[msg['text']]})
        else:doc_data=documentation_collection.find_one({"_id":doc_id});
        if doc_data and"report_html"in doc_data:history.extend([{'role':'user','parts':[f"Report:\n{doc_data['report_html'][:3000]}"]},{'role':'model','parts':["OK."]}])
        if model:chat=model.start_chat(history=history);response=chat.send_message(user_msg);
        if response.candidates:ai_resp=response.text
        if not ai_resp.startswith("[AI Err"):
            try:chats_collection.update_one({"documentation_id":doc_id},{"$push":{"messages":{"role":"AI","text":ai_resp,"timestamp":datetime.utcnow()}}}) 
            except Exception as e:logging.error(f"Err save AI msg:{e}")
        emit('receive_message',{'user':'AI','text':ai_resp},room=sid)
    except Exception as e:logging.error(f"Err proc report chat:{e}");emit('error',{'message':'Server error.'},room=sid)
    finally:emit('typing_indicator',{'isTyping':False},room=sid);logging.info(f"--- Report Chat Msg END (SID:{sid}) ---")

# == Dashboard Namespace (/dashboard_chat) ==
@socketio.on('connect', namespace='/dashboard_chat')
def handle_dashboard_connect(): # (Keep logic)
    if not is_logged_in(): return False
    logging.info(f"User '{session.get('username')}' connected dash chat: {request.sid}")
@socketio.on('disconnect', namespace='/dashboard_chat')
def handle_dashboard_disconnect(): logging.info(f"User '{session.get('username', 'Unknown')}' disconnected dash chat: {request.sid}")
@socketio.on('send_dashboard_message', namespace='/dashboard_chat')
def handle_dashboard_chat(data): # (Keep logic)
    sid=request.sid;logging.debug(f"--- Dash Chat START (SID:{sid}) ---")
    if not is_logged_in():emit('error',{'message':'Auth required.'},room=sid,namespace='/dashboard_chat');return
    if db is None or general_chats_collection is None: emit('error',{'message':'Chat DB unavailable.'},room=sid,namespace='/dashboard_chat');return
    username=session.get('username');user_id_str=session.get('user_id')
    if not username or not user_id_str:emit('error',{'message':'Session error.'},room=sid,namespace='/dashboard_chat');return
    try:user_id=ObjectId(user_id_str)
    except Exception as e:logging.error(f"Invalid sess user_id:{e}");emit('error',{'message':'Session error.'},room=sid,namespace='/dashboard_chat');return
    if not isinstance(data,dict):return
    user_msg=data.get('text','').strip()
    if not user_msg:return
    logging.info(f"Dash Chat from {username}: '{user_msg[:50]}...'")
    try:# Save User Msg
        update_res=general_chats_collection.update_one({"user_id":user_id},{"$push":{"messages":{"role":"user","text":user_msg,"timestamp":datetime.utcnow()}},"$setOnInsert":{"user_id":user_id,"username":username,"start_timestamp":datetime.utcnow()}},upsert=True)
    except Exception as e:logging.error(f"Err save dash user msg:{e}")
    ai_resp="[AI Err]"
    try:# Get AI Resp
        emit('typing_indicator',{'isTyping':True},room=sid,namespace='/dashboard_chat')
        history=[];chat_doc=general_chats_collection.find_one({"user_id":user_id})
        if chat_doc and"messages"in chat_doc: 
            for msg in chat_doc["messages"]:history.append({'role':('model'if msg['role']=='AI'else msg['role']),'parts':[msg['text']]})
        if model: chat=model.start_chat(history=history);response=chat.send_message(user_msg);
        if response.candidates:ai_resp=response.text
        # Save AI Msg
        if not ai_resp.startswith("[AI Err") and general_chats_collection is not None:
             try:general_chats_collection.update_one({"user_id":user_id},{"$push":{"messages":{"role":"AI","text":ai_resp,"timestamp":datetime.utcnow()}}})
             except Exception as e:logging.error(f"Err save dash AI resp:{e}")
        emit('receive_dashboard_message',{'user':'AI','text':ai_resp},room=sid,namespace='/dashboard_chat')
    except Exception as e:logging.error(f"Err proc dash chat:{e}");emit('error',{'message':'Server error.'},room=sid,namespace='/dashboard_chat')
    finally:emit('typing_indicator',{'isTyping':False},room=sid,namespace='/dashboard_chat');logging.debug(f"--- Dash Chat END (SID:{sid}) ---")

# == PDF Chat Namespace (/pdf_chat) ==
@socketio.on('connect', namespace='/pdf_chat')
def handle_pdf_chat_connect(): # (Keep logic)
    if not is_logged_in(): return False
    logging.info(f"User '{session.get('username')}' connected PDF chat: {request.sid}")
@socketio.on('disconnect', namespace='/pdf_chat')
def handle_pdf_chat_disconnect(): logging.info(f"User '{session.get('username', 'Unknown')}' disconnected PDF chat: {request.sid}")
@socketio.on('send_pdf_chat_message', namespace='/pdf_chat')
def handle_pdf_chat_message(data): # (Keep detailed logging logic)
    sid = request.sid; logging.debug(f"--- handle_pdf_chat_message START (SID:{sid}) ---")
    if not is_logged_in(): emit('error', {'message': 'Auth required.'}, room=sid, namespace='/pdf_chat'); logging.debug(f"--- END (SID:{sid}) ---"); return
    if db is None or pdf_analysis_collection is None or pdf_chats_collection is None or model is None: emit('error', {'message': 'Service unavailable.'}, room=sid, namespace='/pdf_chat'); logging.debug(f"--- END (SID:{sid}) ---"); return
    username = session.get('username','Unknown'); user_id_str = session.get('user_id')
    logging.debug(f"(SID:{sid}) PDF Chat Received data: {data}")
    if not isinstance(data, dict): logging.warning(f"Invalid PDF chat data from {username}"); logging.debug(f"--- END (SID:{sid}) ---"); return
    user_message = data.get('text', '').strip(); analysis_id_str = data.get('analysis_id')
    if not user_message or not analysis_id_str: emit('error', {'message': 'Missing text or analysis ID.'}, room=sid, namespace='/pdf_chat'); logging.debug(f"--- END (SID:{sid}) ---"); return
    try: analysis_id = ObjectId(analysis_id_str); user_id = ObjectId(user_id_str) if user_id_str else None; assert user_id is not None
    except Exception as e: logging.error(f"Invalid ID format: {e}"); emit('error', {'message': 'Invalid context ID.'}, room=sid, namespace='/pdf_chat'); logging.debug(f"--- END (SID:{sid}) ---"); return
    logging.info(f"PDF Chat msg for analysis {analysis_id} from '{username}' (SID:{sid}): '{user_message[:50]}...'")
    try: # Save User Msg
        logging.debug(f"(SID:{sid}) PDF Chat Saving user message...")
        user_msg_doc = {"role": "user", "text": user_message, "timestamp": datetime.utcnow()}; update_result_user = pdf_chats_collection.update_one( {"pdf_analysis_id": analysis_id}, {"$push": {"messages": user_msg_doc}, "$setOnInsert": {"pdf_analysis_id": analysis_id, "user_id": user_id, "username": username, "start_timestamp": datetime.utcnow()}}, upsert=True ); logging.info(f"(SID:{sid}) Saved PDF chat user msg result: {update_result_user.raw_result}")
    except Exception as e: logging.error(f"(SID:{sid}) Err save PDF chat user msg: {e}"); logging.error(traceback.format_exc())
    ai_response_text = "[AI Error processing PDF query]"
    try: # Get AI Response
        emit('typing_indicator', {'isTyping': True}, room=sid, namespace='/pdf_chat')
        pdf_text_context = ""; history = []; logging.debug(f"(SID:{sid}) PDF Chat Retrieving context _id={analysis_id} user_id={user_id}")
        pdf_doc = pdf_analysis_collection.find_one({"_id": analysis_id, "user_id": user_id})
        if pdf_doc: pdf_text_context = pdf_doc.get("extracted_text_preview", "")
        else: logging.error(f"PDF doc {analysis_id} not found/owned by user {user_id}"); emit('error', {'message': 'PDF context error.'}, room=sid, namespace='/pdf_chat'); emit('typing_indicator', {'isTyping': False}, room=sid, namespace='/pdf_chat'); logging.debug(f"--- END (SID:{sid}) ---"); return
        logging.debug(f"(SID:{sid}) PDF Chat Fetching history pdf_analysis_id={analysis_id}")
        chat_doc = pdf_chats_collection.find_one({"pdf_analysis_id": analysis_id})
        if chat_doc and "messages" in chat_doc:
             for msg in chat_doc["messages"][-6:]: history.append({'role': ('model' if msg['role']=='AI' else msg['role']), 'parts': [msg['text']]})
             logging.info(f"Rebuilt PDF chat history ({len(history)} msgs)")
        # *** CORRECTED HISTORY STRING CONSTRUCTION ***
        history_string = ""
        if history: history_string = "".join([f"{m['role']}: {m['parts'][0]}\n" for m in history])
        else: history_string = "No previous messages.\n"
        # *********************************************
        pdf_chat_prompt = f"""Context from PDF:\n---\n{pdf_text_context if pdf_text_context else "No text."}\n---\n\nChat History:\n---\n{history_string}---\n\nUser Question: {user_message}\n\nAnswer based *only* on the PDF Context and History:"""
        logging.debug(f"(SID:{sid}) PDF Gemini prompt (start):\n{pdf_chat_prompt[:300]}...")
        logging.info(f"(SID:{sid}) Sending PDF Chat Query to Gemini...")
        response = model.generate_content(pdf_chat_prompt, safety_settings=safety_settings)
        logging.info(f"(SID:{sid}) Received PDF Chat Gemini response.")
        if response.candidates: ai_response_text = response.text if hasattr(response, 'text') else "[AI format error]"
        else: ai_response_text = "[AI blocked/empty]"; logging.error(f"PDF Chat AI blocked. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        # Save AI Msg
        if not ai_response_text.startswith("[AI Err") and pdf_chats_collection is not None:
            try: logging.debug(f"(SID:{sid}) PDF Chat Saving AI message..."); ai_msg_doc = {"role": "AI", "text": ai_response_text, "timestamp": datetime.utcnow()}; update_result_ai = pdf_chats_collection.update_one({"pdf_analysis_id": analysis_id}, {"$push": {"messages": ai_msg_doc}}); logging.info(f"(SID:{sid}) Saved PDF chat AI response. Result: {update_result_ai.raw_result}")
            except Exception as e: logging.error(f"(SID:{sid}) Err save PDF chat AI msg: {e}"); logging.error(traceback.format_exc())
        else: logging.warning(f"(SID:{sid}) Skipping DB save for PDF AI error: {ai_response_text}")
        # Emit to Client
        logging.info(f"(SID:{sid}) Emitting 'receive_pdf_chat_message': '{ai_response_text[:50]}...'")
        emit('receive_pdf_chat_message', {'user': 'AI', 'text': ai_response_text}, room=sid, namespace='/pdf_chat')
    except Exception as e: logging.error(f"(SID:{sid}) Unexpected error in PDF chat processing for analysis {analysis_id}: {e}"); logging.error(traceback.format_exc()); emit('error', {'message': 'Server error during PDF chat.'}, room=sid, namespace='/pdf_chat')
    finally: emit('typing_indicator', {'isTyping': False}, room=sid, namespace='/pdf_chat'); logging.info(f"--- handle_pdf_chat_message END (SID:{sid}) ---")


# --- Main Execution ---
if __name__ == '__main__':
    if db is None: logging.critical("MongoDB connection failed. Aborting."); exit(1)
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'dev-secret-key-only-not-for-production!': logging.warning("WARNING: Running with insecure default FLASK_SECRET_KEY!")
    logging.info("Starting Flask-SocketIO server...")
    try:
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        logging.critical(f"Failed to start server: {e}"); logging.critical(traceback.format_exc())