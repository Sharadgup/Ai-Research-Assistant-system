# --- START: Apply Eventlet Monkey Patching ---
import eventlet
eventlet.monkey_patch()
# --- END: Apply Eventlet Monkey Patching ---

import os
import google.generativeai as genai
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session, Blueprint)
# *** ADD ProxyFix Import ***
from werkzeug.middleware.proxy_fix import ProxyFix
# **************************
from flask_socketio import SocketIO, emit, send
from dotenv import load_dotenv
import logging
import json
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, DuplicateKeyError
from datetime import datetime
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

# --- Flask-Dance Imports (Only Google) ---
from flask_dance.contrib.google import make_google_blueprint, google

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# Load environment variables
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

# *** APPLY ProxyFix Middleware FIRST ***
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# ***************************************

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
if not app.config['SECRET_KEY']:
    logging.critical("CRITICAL: FLASK_SECRET_KEY not set.")
    app.config['SECRET_KEY'] = 'dev-secret-key-only-not-for-production!'

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
    logging.info("Attempting to create Google OAuth Blueprint...")
    try:
        forced_redirect_uri = "https://5000-idx-ai-note-system-1744087101492.cluster-a3grjzek65cxex762e4mwrzl46.cloudworkstations.dev/login/google/authorized"
        # forced_redirect_uri = "http://127.0.0.1:5000/login/google/authorized" # For local testing
        logging.info(f"Forcing Google redirect_uri to: {forced_redirect_uri}")

        google_bp = make_google_blueprint(
            scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
            redirect_to="google_auth_callback",
            offline=False,
            redirect_uri=forced_redirect_uri
        )
        app.register_blueprint(google_bp, url_prefix="/login")
        logging.info("Google OAuth Blueprint registered successfully with forced redirect_uri.")
    except Exception as bp_error:
        logging.error(f"Failed to create or register Google Blueprint: {bp_error}")
        google_enabled = False


# --- MongoDB Initialization ---
MONGO_URI = os.getenv("MONGODB_URI"); DB_NAME = os.getenv("MONGODB_DB_NAME")
db = None; registrations_collection = None; input_prompts_collection = None;
documentation_collection = None; chats_collection = None; general_chats_collection = None;
if not MONGO_URI or not DB_NAME: logging.critical("Missing MongoDB Config")
else:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000); mongo_client.admin.command('ismaster')
        db = mongo_client[DB_NAME]; logging.info(f"MongoDB connected. DB: '{DB_NAME}'")
        registrations_collection = db["registrations"]; input_prompts_collection = db["input_prompts"]
        documentation_collection = db["documentation"]; chats_collection = db["chats"]; general_chats_collection = db["general_chats"]
        try: registrations_collection.create_index("username", unique=True, sparse=True)
        except: logging.warning("Username index issue")
        try: registrations_collection.create_index("email", unique=True, sparse=True)
        except: logging.warning("Email index issue")
        try: registrations_collection.create_index("google_id", unique=True, sparse=True)
        except: logging.warning("Google ID index issue")
        try: general_chats_collection.create_index("user_id", unique=True)
        except: logging.warning("General chat index issue")
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

# --- HTTP Routes ---
@app.route('/')
def landing_page(): return render_template('landing.html', now=datetime.utcnow(), google_login_enabled=google_enabled)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if is_logged_in(): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if db is None: flash("DB error.", "danger"); return render_template('register.html', now=datetime.utcnow(), google_login_enabled=google_enabled)
        username=request.form.get('username','').strip(); password=request.form.get('password',''); confirm=request.form.get('confirm_password','')
        error=None;
        if not username: error="Username required."
        elif not password: error="Password required."
        elif password!=confirm: error="Passwords don't match."
        elif len(password)<6: error="Password min 6 chars."
        if error: flash(error, "warning"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        hash_val=generate_password_hash(password)
        try: user_doc={"username":username, "password_hash":hash_val, "created_at":datetime.utcnow(), "login_method":"password"}; registrations_collection.insert_one(user_doc); flash("Registered!", "success"); return redirect(url_for('login'))
        except DuplicateKeyError: flash("Username exists.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
        except Exception as e: logging.error(f"Reg error: {e}"); flash("Registration error.", "danger"); return render_template('register.html', username=username, now=datetime.utcnow(), google_login_enabled=google_enabled)
    return render_template('register.html', now=datetime.utcnow(), google_login_enabled=google_enabled)

@app.route('/login', methods=['GET', 'POST'])
def login():
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

# --- OAuth Callback Route (Google Only) ---
@app.route("/google/authorized")
def google_auth_callback():
    # (Keep existing callback logic)
    if not google_enabled or not google.authorized: flash("Google login failed/disabled.", "danger"); return redirect(url_for("login"))
    try:
        resp = google.get("/oauth2/v3/userinfo");
        if not resp.ok: logging.error(f"Failed Google fetch:{resp.status_code}"); flash("Fetch Google info failed.", "danger"); return redirect(url_for("login"))
        user_info = resp.json(); google_id = user_info.get("sub"); email = user_info.get("email"); name = user_info.get("name")
        if not google_id: flash("No Google ID.", "danger"); return redirect(url_for("login"))
        logging.info(f"Google auth OK. ID:{google_id}, Email:{email}")
        user_doc = registrations_collection.find_one({"google_id": google_id})
        if not user_doc and email: user_doc = registrations_collection.find_one({"email": email})
        if user_doc: # Found user
            update_data={"$set":{"last_login_at":datetime.utcnow(),"login_method":"google"}}
            if not user_doc.get("google_id"):update_data["$set"]["google_id"]=google_id
            if not user_doc.get("name")and name:update_data["$set"]["name"]=name
            registrations_collection.update_one({"_id":user_doc["_id"]},update_data)
            user_doc=registrations_collection.find_one({"_id":user_doc["_id"]})
        else: # Create user
            user_data={"google_id":google_id,"email":email,"name":name,"created_at":datetime.utcnow(),"last_login_at":datetime.utcnow(),"login_method":"google"}
            user_doc=registrations_collection.find_one({"_id":registrations_collection.insert_one(user_data).inserted_id})
        if user_doc: login_user(user_doc); return redirect(url_for("dashboard"))
        else: flash("Login failed post-Google.", "danger"); return redirect(url_for("login"))
    except Exception as e: logging.error(f"Google callback error:{e}"); logging.error(traceback.format_exc()); flash("Google login error.", "danger"); return redirect(url_for("login"))

# --- Other Routes ---
@app.route('/logout')
def logout(): username = session.get('username', 'Unknown'); session.clear(); flash("Logged out.", "success"); logging.info(f"User '{username}' logged out."); return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    username=session.get('username','User');
    available_models=["G 1.5 Flash", "G Pro"]; usable_models=["G 1.5 Flash"]; sectors=["Healthcare", "Finance", "Tech", "Edu", "Retail", "General"]; apps=[{"id":"tts","name":"TTS"},{"id":"ttv","name":"TTV"}]; services=["Analysis", "Viz", "Chat", "PDF"]
    dashboard_data={"username":username, "services":services, "available_models":available_models, "usable_models":usable_models, "sectors":sectors, "apps":apps}
    return render_template('dashboard.html', data=dashboard_data, now=datetime.utcnow())

@app.route('/index')
def report_page(): return render_template('index.html', now=datetime.utcnow())

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
    try:prompt_doc={"original_text":input_text,"timestamp":datetime.utcnow(),"user_id":user_id};prompt_doc_id=input_prompts_collection.insert_one(prompt_doc).inserted_id
    except Exception as e:logging.error(f"Err save prompt:{e}")
    prompt=f"Analyze...\n{input_text}\nReport:\n---" # Use full prompt
    try:
        response=model.generate_content(prompt);
        if not response.candidates:raise ValueError("AI err")
        gen_text=response.text;report_content=gen_text;chart_data={}
        try: # Parse JSON
             json_start_marker="```json_chart_data"; json_end_marker="```"; start_index=gen_text.rfind(json_start_marker)
             if start_index!=-1:
                 end_index=gen_text.find(json_end_marker, start_index+len(json_start_marker))
                 if end_index!=-1:
                     json_string=gen_text[start_index+len(json_start_marker):end_index].strip()
                     try: chart_data=json.loads(json_string); report_content=gen_text[:start_index].strip()
                     except Exception as json_e: logging.error(f"JSON Parse Err: {json_e}")
        except Exception as parse_e: logging.error(f"Parse Err: {parse_e}")
        doc_id=None
        try:
            doc_save={"input_prompt_id":prompt_doc_id,"user_id":user_id,"report_html":report_content,"chart_data":chart_data,"timestamp":datetime.utcnow(),"model_used":model_name,"finish_reason":response.candidates[0].finish_reason.name}
            doc_id=documentation_collection.insert_one(doc_save).inserted_id
            if prompt_doc_id:input_prompts_collection.update_one({"_id":prompt_doc_id},{"$set":{"related_documentation_id":doc_id}})
        except Exception as e:logging.error(f"Err save doc:{e}")
        return jsonify({"report_html":report_content,"chart_data":chart_data,"report_context_for_chat":report_content[:3000],"documentation_id":str(doc_id) if doc_id else None})
    except Exception as e:logging.error(f"ERROR gen report: {e}"); return jsonify({"error":"Server error"}),500

# *** Corrected placement for Education Agent Route Definitions ***
@app.route('/education_agent')
def education_agent_page():
    """Renders the dedicated page for the Education AI Assistant."""
    if not is_logged_in():
        flash("Please log in to use the AI agents.", "warning")
        return redirect(url_for('login'))
    return render_template('education_agent.html', now=datetime.utcnow())

@app.route('/education_agent_query', methods=['POST'])
def education_agent_query():
    """Handles queries specifically for the Education AI agent."""
    if not is_logged_in(): return jsonify({"error": "Authentication required."}), 401
    if not model: return jsonify({"error": "AI model unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Need JSON"}), 400
    data = request.get_json(); user_query = data.get('query','').strip()
    if not user_query: return jsonify({"error": "No query."}), 400
    username = session.get('username', 'User')
    logging.info(f"Edu Agent query from '{username}': '{user_query[:100]}...'")
    education_prompt = f"You are an expert AI Education Assistant... User Query: {user_query}\n Answer:" # Use full prompt
    ai_response_text = "[Error processing query]"
    try:
        response = model.generate_content(education_prompt)
        if response.candidates: ai_response_text = response.text
        else: ai_response_text = "[AI response empty/blocked]"
        return jsonify({"answer": ai_response_text })
    except Exception as e: logging.error(f"Error processing edu query: {e}"); return jsonify({"error": "Server error processing query."}), 500
# ******************************************************************


# --- SocketIO Event Handlers ---
# (Keep all existing handlers for both namespaces)
# == Default Namespace (Report Chat) ==
@socketio.on('connect')
def handle_connect(): logging.info(f"(Report Chat) Connect: {request.sid}")
@socketio.on('disconnect')
def handle_disconnect(): logging.info(f"(Report Chat) Disconnect: {request.sid}")
@socketio.on('send_message')
def handle_send_message(data):
    sid=request.sid;logging.info(f"--- Report Chat Msg START (SID:{sid}) ---")
    if db is None:emit('error',{'message':'DB unavailable.'},room=sid);return
    user_msg=data.get('text');doc_id_str=data.get('documentation_id')
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
def handle_dashboard_connect():
    if not is_logged_in(): return False
    logging.info(f"User '{session.get('username')}' connected dash chat: {request.sid}")
@socketio.on('disconnect', namespace='/dashboard_chat')
def handle_dashboard_disconnect(): logging.info(f"User '{session.get('username', 'Unknown')}' disconnected dash chat: {request.sid}")
@socketio.on('send_dashboard_message', namespace='/dashboard_chat')
def handle_dashboard_chat(data):
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
        logging.info(f"User msg save result: {update_res.raw_result}")
    except Exception as e:logging.error(f"Err save dash user msg:{e}")
    ai_resp="[AI Err]"
    try:# Get AI Resp
        emit('typing_indicator',{'isTyping':True},room=sid,namespace='/dashboard_chat')
        history=[];chat_doc=general_chats_collection.find_one({"user_id":user_id})
        if chat_doc and"messages"in chat_doc: # History
             for msg in chat_doc["messages"]:history.append({'role':('model'if msg['role']=='AI'else msg['role']),'parts':[msg['text']]})
        if model:# Gemini
            chat=model.start_chat(history=history);response=chat.send_message(user_msg)
            if response.candidates:ai_resp=response.text
        # Save AI Msg (Corrected check: is not None)
        if not ai_resp.startswith("[AI Err") and general_chats_collection is not None: # Corrected Check
             try:general_chats_collection.update_one({"user_id":user_id},{"$push":{"messages":{"role":"AI","text":ai_resp,"timestamp":datetime.utcnow()}}})
             except Exception as e:logging.error(f"Err save dash AI resp:{e}")
        elif general_chats_collection is None: logging.error(f"Cannot save AI rsp, collection is None!")
        else: logging.warning(f"Skipping save for AI error: '{ai_resp}'")
        # Emit to client
        emit('receive_dashboard_message',{'user':'AI','text':ai_resp},room=sid,namespace='/dashboard_chat')
    except Exception as e:logging.error(f"Err proc dash chat:{e}");emit('error',{'message':'Server error.'},room=sid,namespace='/dashboard_chat')
    finally:emit('typing_indicator',{'isTyping':False},room=sid,namespace='/dashboard_chat');logging.debug(f"--- Dash Chat END (SID:{sid}) ---")


# --- Main Execution ---
if __name__ == '__main__':
    if db is None: logging.critical("MongoDB connection failed. Aborting."); exit(1)
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'dev-secret-key-only-not-for-production!': logging.warning("WARNING: Running with insecure default FLASK_SECRET_KEY!")
    logging.info("Starting Flask-SocketIO server...")
    try:
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        logging.critical(f"Failed to start server: {e}"); logging.critical(traceback.format_exc())