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
from datetime import datetime, timedelta
from bson import ObjectId, json_util
from bson.errors import InvalidId
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename # For file uploads
import fitz # PyMuPDF library
import pandas as pd # Added pandas
import io # For sending dataframe file data without saving temp file
from fpdf import FPDF # For PDF generation example
import plotly.express as px # Example using Plotly for visualizations
import plotly.io as pio # To convert Plotly figs to JSON
import requests
from urllib.parse import urlparse # To check if content is a URL


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


# --- Data Analyzer Constants ---
ANALYSIS_UPLOAD_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_data')
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
def allowed_analysis_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(ANALYSIS_UPLOAD_FOLDER):
    os.makedirs(ANALYSIS_UPLOAD_FOLDER)
    logging.info(f"Created analysis upload directory: {ANALYSIS_UPLOAD_FOLDER}")


# --- Get API Key (remains the same) ---

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
voice_conversations_collection = None # <-- NEW Voice Conversation Collection
analysis_uploads_collection = None # <-- NEW Analysis Collection
news_articles_collection  = None # <-- New option functionality add it

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
        pdf_analysis_collection = db["pdf_analysis"]; pdf_chats_collection = db["pdf_chats"]
        voice_conversations_collection = db["voice_conversations"] # <-- Assign voice collection
        logging.info("MongoDB Collections assigned.")
        analysis_uploads_collection = db["analysis_uploads"] # <-- Assign analysis collection
        logging.info("MongoDB Collections assigned.")
        news_articles_collection = db["news_articles"]
        logging.info("Assigned news_articles collection.")

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

        try: voice_conversations_collection.create_index("user_id") # <-- Add index for voice convos
        except Exception as idx_err: logging.warning(f"Voice Convo index warn: {idx_err}")

        try: analysis_uploads_collection.create_index("user_id") # <-- Add index for analysis uploads
        except Exception as idx_err: logging.warning(f"Analysis Uploads index warn: {idx_err}")

        try:
            # Index URL for faster lookups and ensuring uniqueness if storing articles
            logging.debug("Attempting to create unique index on 'url' for news_articles.") # Optional debug log
            news_articles_collection.create_index("url", unique=True)
            logging.info("Successfully created/verified unique index on 'url' for news_articles.") # Log success
        except Exception as idx_err_url:
            # Log warning specifically for the URL index failure
            logging.warning(f"News Articles 'url' unique index creation/verification warning: {idx_err_url}")

        # Try creating index on 'fetched_at'
        try:
            # Index for sorting or querying by fetch time if storing articles
            logging.debug("Attempting to create index on 'fetched_at' for news_articles.") # Optional debug log
            news_articles_collection.create_index("fetched_at")
            logging.info("Successfully created/verified index on 'fetched_at' for news_articles.") # Log success
        except Exception as idx_err_fetch:
            # Log warning specifically for the fetched_at index failure
            logging.warning(f"News Articles 'fetched_at' index creation/verification warning: {idx_err_fetch}")

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

# Option 2: Keep fallback but log clearly if it's used
fallback_key = "MTDrUuB40hsh8vr68q7KDqV9PysQ4czz" # For testing ONLY
WORLD_NEWS_API_KEY = os.getenv("WORLD_NEWS_API_KEY")
if not WORLD_NEWS_API_KEY:
    logging.warning("!!! WORLD_NEWS_API_KEY not found in environment. Using fallback key for testing. !!!")
    WORLD_NEWS_API_KEY = fallback_key
# --- Use the CORRECT endpoint for searching ---
api_endpoint = "https://api.worldnewsapi.com/search-news"


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

# --- Data Analysis Helper Functions ---

def get_dataframe(filepath):
    """Safely reads CSV or Excel into Pandas DataFrame."""
    try:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.lower().endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            logging.warning(f"Unsupported file type for get_dataframe: {filepath}")
            return None
        # Basic type optimization (optional, can be refined)
        df = df.convert_dtypes()
        logging.info(f"Successfully read dataframe from {os.path.basename(filepath)}, Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}", exc_info=True)
        return None

def get_column_info(df):
    """Generates summary info for DataFrame columns."""
    info = []
    for col in df.columns:
        # Ensure JSON serializability for counts (convert numpy types)
        null_count = int(df[col].isnull().sum())
        info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count
        })
    return info

def generate_data_profile(df):
    """Creates a basic profile of the DataFrame."""
    if df is None: return {}
    profile = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "column_info": get_column_info(df),
        "memory_usage": int(df.memory_usage(deep=True).sum()) # Ensure int for JSON
        # Add more profiling: duplicate rows, skewness, kurtosis, etc.
    }
    return profile

def generate_cleaning_recommendations(df):
    """Basic recommendation engine (can be significantly improved)."""
    if df is None: return []
    recommendations = []
    col_info = get_column_info(df)
    total_rows = len(df)
    if total_rows == 0: return ["Dataframe is empty."]

    for col in col_info:
        # Null Value Recommendations
        if col['null_count'] > 0:
            null_percent = (col['null_count'] / total_rows) * 100
            rec = f"Column **'{col['name']}'** has {col['null_count']} ({null_percent:.2f}%) null values. Consider handling (e.g., fill with mean/median/mode, custom value, or drop rows/column)."
            if null_percent > 50:
                 rec += " *High percentage suggests dropping the column might be viable.*"
            recommendations.append(rec)

        # Data Type Recommendations (Object/String)
        if 'object' in col['dtype'] or 'string' in col['dtype']:
             try:
                 unique_vals = df[col['name']].nunique()
                 if unique_vals < 20 and total_rows > 50: # Arbitrary thresholds
                     recommendations.append(f"Column **'{col['name']}'** ({col['dtype']} type) has low cardinality ({unique_vals} unique values). Consider converting to 'category' type for potential memory efficiency.")
                 # Check for potential numeric strings
                 if unique_vals > 0:
                     sample = df[col['name']].dropna().sample(min(5, len(df[col['name']].dropna()))) # Sample non-nulls
                     if all(s.replace('.', '', 1).replace('-', '', 1).isdigit() for s in sample if isinstance(s, str)):
                         recommendations.append(f"Column **'{col['name']}'** ({col['dtype']}) contains values that look numeric ('{sample.iloc[0]}'...). Consider converting to a numeric type (e.g., float, integer) if appropriate.")
                 # Check for very long strings
                 max_len = df[col['name']].astype(str).str.len().max()
                 if max_len > 200: # Arbitrary length
                    recommendations.append(f"Column **'{col['name']}'** ({col['dtype']}) has long text entries (max length: {max_len}). This might impact performance or analysis. Check if truncation/feature extraction is needed.")

             except Exception as e:
                 logging.warning(f"Error analyzing column '{col['name']}' for recommendations: {e}")

        # Numeric Type Recommendations
        if pd.api.types.is_numeric_dtype(df[col['name']]) and col['null_count'] < total_rows:
            try:
                skewness = df[col['name']].skew()
                if abs(skewness) > 1: # Arbitrary threshold for significant skew
                    recommendations.append(f"Numeric column **'{col['name']}'** appears skewed (skewness: {skewness:.2f}). Consider transformation (e.g., log, sqrt) if using models sensitive to distribution.")
                # Basic outlier check (crude method)
                q1 = df[col['name']].quantile(0.25)
                q3 = df[col['name']].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col['name']] < lower_bound) | (df[col['name']] > upper_bound)]
                if not outliers.empty:
                    recommendations.append(f"Numeric column **'{col['name']}'** may have potential outliers (found {len(outliers)} outside 1.5*IQR range). Investigate further.")
            except Exception as e:
                logging.warning(f"Error calculating numeric stats for column '{col['name']}': {e}")

    # Duplicate Row Check
    if df.duplicated().any():
         recommendations.append(f"Dataset contains duplicate rows ({df.duplicated().sum()}). Consider removing them using the 'Remove Duplicates' action.")

    if not recommendations:
        recommendations.append("No immediate cleaning recommendations based on basic checks. Data looks relatively clean.")

    return recommendations

def generate_gemini_insight_prompt(profile, cleaning_steps):
    """Generates a prompt for Gemini based on data profile."""
    prompt = f"""Analyze the following data profile and applied cleaning steps. Provide key insights, potential issues, and recommendations for further analysis.

**Data Profile Summary:**
- Rows: {profile.get('row_count', 'N/A')}
- Columns: {profile.get('col_count', 'N/A')}
- Memory Usage: {profile.get('memory_usage', 'N/A')} bytes
- Column Details:
"""
    if 'column_info' in profile:
        for col in profile['column_info']:
            prompt += f"  - Name: {col['name']}, Type: {col['dtype']}, Nulls: {col['null_count']} ({ (col['null_count'] / profile['row_count'] * 100) if profile.get('row_count',0) > 0 else 0 :.1f}%)\n"
    else:
        prompt += "  (Column details not available)\n"

    if cleaning_steps:
        prompt += "\n**Cleaning Steps Applied:**\n"
        for i, step in enumerate(cleaning_steps):
             action = step.get('action', 'N/A')
             column = step.get('column', 'N/A')
             method = step.get('method', step.get('new_type', 'N/A')) # Get method or new_type
             params = step.get('params', {})
             details = f"Method: {method}" if method != 'N/A' else ""
             if params and not method: # Add params if method wasn't captured but params exist
                 details += f" Params: {json.dumps(params)}"
             prompt += f"- Step {i+1}: Action='{action}', Column='{column}'"
             if details: prompt += f", Details='{details}'"
             prompt += "\n"
    else:
        prompt += "\n**Cleaning Steps Applied:** None\n"

    prompt += """
**Analysis Request:**

Based ONLY on the summary and cleaning steps provided above, provide the following in Markdown format:

1.  **Key Observations:**
    *   (e.g., Identify potential relationships between columns based on names/types. Highlight columns with significant null percentages. Note potential categorical features or identifiers. Comment on data size/memory.)
2.  **Potential Data Issues & Considerations:**
    *   (e.g., Mention implications of high null counts. Suggest potential data type mismatches or columns needing normalization/scaling. Discuss impact of cleaning steps, if any. Identify possible target variables based on names/types.)
3.  **Recommendations for Next Steps:**
    *   (e.g., Suggest specific visualizations like histograms for numeric columns, bar charts for categoricals, scatter plots for potential relationships, heatmaps if correlation was run. Recommend statistical tests like correlation analysis, t-tests/ANOVA if groups exist. Suggest feature engineering possibilities like creating date features or binning numeric data.)

**Important:** Focus on actionable insights derived *strictly* from the provided profile and cleaning information. Do not invent data points or assume external knowledge about the dataset's domain. Keep the response concise and focused.
"""
    return prompt

# --- PDF Report Generation Class ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Data Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255) # Light blue background
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(5)

    def chapter_body(self, body_text):
        self.set_font('Arial', '', 10)
        # Handle potential encoding issues - encode then decode with latin-1 replace
        safe_text = body_text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, safe_text)
        self.ln()

    def add_table(self, header, data, col_widths=None):
         self.set_font('Arial', 'B', 9)
         self.set_fill_color(224, 235, 255) # Slightly darker blue for header
         page_width = self.w - 2 * self.l_margin # Available width

         if col_widths is None: # Default equal widths
             num_cols = len(header)
             default_col_width = page_width / num_cols if num_cols > 0 else page_width
             col_widths = [default_col_width] * num_cols
         elif sum(col_widths) > page_width:
              logging.warning("PDF table column widths exceed page width. Adjusting.")
              # Simple scaling adjustment
              scale_factor = page_width / sum(col_widths)
              col_widths = [w * scale_factor for w in col_widths]

         # Header
         for i, h in enumerate(header):
             self.cell(col_widths[i], 7, str(h), 1, 0, 'C', fill=True)
         self.ln()

         # Data
         self.set_font('Arial', '', 8)
         self.set_fill_color(255, 255, 255) # White background for data rows
         fill = False
         for row in data:
             # Ensure row has the correct number of items, padding if necessary
             row_items = list(row) + [''] * (len(header) - len(row))
             for i, item in enumerate(row_items):
                 # Truncate long items gently, encode safely
                 item_str = str(item).encode('latin-1', 'replace').decode('latin-1')
                 max_chars = int(col_widths[i] / 1.8) # Heuristic for max chars based on width
                 display_item = (item_str[:max_chars-3] + '...') if len(item_str) > max_chars else item_str
                 self.cell(col_widths[i], 6, display_item, 1, 0, 'L', fill=fill)
             self.ln()
             fill = not fill # Alternate row shading
         self.ln()

    def add_json_block(self, title, json_data):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Courier', '', 8) # Use monospace for JSON
        try:
            # Pretty print JSON, handle potential encoding errors
            json_str = json.dumps(json_data, indent=2)
            safe_json_str = json_str.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_json_str)
        except Exception as e:
            logging.error(f"Error formatting JSON for PDF: {e}")
            self.set_font('Arial', 'I', 8)
            self.multi_cell(0, 5, "[Error displaying JSON data]")
        self.ln()




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

# --- Data Analyzer Routes ---

@app.route('/data_analyzer')
def data_analyzer_page():
    # Ensure necessary imports like flash, redirect, url_for, render_template, datetime, is_logged_in are available
    if not is_logged_in():
        flash("Please log in to use the Data Analyzer.", "warning")
        return redirect(url_for('login'))
    return render_template('data_analyzer.html', now=datetime.utcnow())


@app.route('/upload_analysis_data', methods=['POST'])
def upload_analysis_data():
    logging.info("--- Enter /upload_analysis_data ---")

    if not is_logged_in():
        return jsonify({"error": "Authentication required."}), 401

    if analysis_uploads_collection is None:
        return jsonify({"error": "Database service unavailable."}), 503

    if 'analysisFile' not in request.files:
        return jsonify({"error": "No file part named 'analysisFile' found in the request."}), 400

    file = request.files['analysisFile']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_analysis_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_ANALYSIS_EXTENSIONS)}"}), 400

    try:
        user_id = ObjectId(session['user_id'])
        username = session.get('username', 'Unknown')
    except Exception as session_err:
        logging.error(f"Session error: {session_err}", exc_info=True)
        return jsonify({"error": "Invalid session."}), 401

    original_filename = secure_filename(file.filename)
    _, f_ext = os.path.splitext(original_filename)
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    stored_filename = f"{user_id}_{ts}{f_ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)

    try:
        file.save(filepath)
        df = get_dataframe(filepath)
        if df is None:
            os.remove(filepath)
            return jsonify({"error": "Failed to read or unsupported file format."}), 400

        profile = generate_data_profile(df)

        doc = {
            "user_id": user_id, "username": username,
            "original_filename": original_filename, "stored_filename": stored_filename,
            "filepath": filepath, "upload_timestamp": datetime.utcnow(),
            "row_count": profile['row_count'],
            "col_count": profile['col_count'],
            "column_info": profile['column_info'],
            "cleaning_steps": [], "analysis_results": {},
            "generated_insights": [], "status": "uploaded",
            "last_modified": datetime.utcnow()
        }

        insert_result = analysis_uploads_collection.insert_one(doc)
        upload_id = insert_result.inserted_id

        return jsonify({
            "message": "File uploaded successfully.",
            "upload_id": str(upload_id),
            "filename": original_filename,
            "rows": profile['row_count'],
            "columns": profile['col_count'],
            "column_info": profile['column_info']
        }), 200

    except Exception as e:
        logging.error(f"Upload failed: {e}", exc_info=True)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as cleanup_err:
                logging.warning(f"Cleanup failed: {cleanup_err}")
        return jsonify({"error": "Server error during upload."}), 500


@app.route('/data_cleaner/<upload_id>')
def data_cleaner_page(upload_id):
    # Ensure necessary imports: logging, flash, redirect, url_for, session, ObjectId, analysis_uploads_collection,
    # render_template, datetime, get_dataframe, generate_data_profile, generate_cleaning_recommendations,
    # json, json_util, InvalidId, os
    logging.info(f"--- ENTER data_cleaner_page for upload_id: {upload_id} ---")

    if not is_logged_in():
        flash("Please log in.", "warning"); logging.warning("data_cleaner_page: Not logged in."); return redirect(url_for('login'))
    if analysis_uploads_collection is None:
        flash("Database service unavailable.", "danger"); logging.error("data_cleaner_page: DB collection unavailable."); return redirect(url_for('dashboard'))

    try:
        logging.info("Step 1: Validate ObjectId format...")
        oid = ObjectId(upload_id) # Can raise InvalidId
        logging.info(f"Step 1 SUCCESS: ObjectId seems valid: {oid}")

        logging.info("Step 2: Get User ID from session...")
        user_id_str = session.get('user_id')
        if not user_id_str: raise ValueError("User ID not found in session.")
        user_id = ObjectId(user_id_str) # Can raise InvalidId
        logging.info(f"Step 2 SUCCESS: User ObjectId: {user_id}")

        logging.info(f"Step 3: Find document _id={oid}, user_id={user_id}...")
        upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
        if not upload_doc:
            logging.warning(f"Step 3 FAILED: Record not found or access denied."); flash("Analysis record not found or access denied.", "danger"); return redirect(url_for('analysis_history'))
        logging.info("Step 3 SUCCESS: Document found.")

        logging.info("Step 4: Load DataFrame...")
        filepath = upload_doc.get('filepath')
        if not filepath or not os.path.exists(filepath):
             logging.error(f"Step 4 FAILED: Filepath missing or file does not exist. Path: {filepath}"); flash("Data file associated with this record is missing.", "danger"); return redirect(url_for('analysis_history'))
        df = get_dataframe(filepath)
        if df is None:
             logging.error(f"Step 4 FAILED: get_dataframe returned None for path: {filepath}"); flash("Error loading the data file.", "danger"); return redirect(url_for('analysis_history'))
        logging.info(f"Step 4 SUCCESS: DataFrame loaded. Shape: {df.shape}")

        logging.info("Step 5: Generate profile and recommendations...")
        # Use current data from DB first, then calculate recommendations from loaded DF
        profile = {
            "row_count": upload_doc.get('row_count'),
            "col_count": upload_doc.get('col_count'),
            "column_info": upload_doc.get('column_info', []) # Use info stored in DB
        }
        recommendations = generate_cleaning_recommendations(df) # Generate based on current DF state
        logging.info("Step 5 SUCCESS: Profile info retrieved, recommendations generated.")

        logging.info("Step 6: Prepare data for template (serializing BSON)...")
        try:
             template_upload_data = json.loads(json_util.dumps(upload_doc)) # Serialize the *whole* doc for the template
             preview_data = df.head(100).to_dict(orient='records') # Generate preview from loaded DF
        except Exception as serial_err:
             logging.error(f"Step 6 FAILED: Error during BSON serialization: {serial_err}", exc_info=True); raise
        logging.info("Step 6 SUCCESS: Data prepared for template.")

        logging.info(f"Step 7: Rendering data_cleaner.html template for upload_id: {upload_id}")
        return render_template('data_cleaner.html',
                               upload_data=template_upload_data, # Full serialized doc
                               preview_data=preview_data,       # Preview from current DF
                               column_info=profile['column_info'], # Column info from DB doc
                               recommendations=recommendations, # Recs from current DF
                               now=datetime.utcnow())

    except InvalidId:
        logging.error(f"Invalid ObjectId format received in URL ('{upload_id}') or session ('{session.get('user_id')}')")
        flash("Invalid analysis record identifier provided.", "danger")
        return redirect(url_for('analysis_history'))
    except ValueError as ve: # Catch specific errors like session key missing
         logging.error(f"Value error loading data cleaner page for {upload_id}: {ve}", exc_info=True)
         flash(f"Error loading cleaner: {ve}", "danger")
         return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Unexpected error loading data cleaner page for {upload_id}: {e}", exc_info=True)
        flash("An unexpected error occurred while loading the data cleaner.", "danger")
        return redirect(url_for('dashboard'))
    finally:
        logging.info(f"--- EXIT data_cleaner_page for upload_id: {upload_id} ---")


@app.route('/apply_cleaning_action/<upload_id>', methods=['POST'])
def apply_cleaning_action(upload_id):
    # Ensure necessary imports: logging, jsonify, request, session, ObjectId, analysis_uploads_collection,
    # pd, get_dataframe, generate_data_profile, generate_cleaning_recommendations, json, datetime, log_db_update_result
    if not is_logged_in(): return jsonify({"error": "Authentication required."}), 401
    if analysis_uploads_collection is None: return jsonify({"error": "Database service unavailable."}), 503

    try:
        oid = ObjectId(upload_id)
        user_id = ObjectId(session['user_id'])
    except Exception as e: return jsonify({"error": f"Invalid ID format: {e}"}), 400

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: return jsonify({"error": "Analysis record not found or unauthorized."}), 404

    data = request.get_json()
    if not data: return jsonify({"error": "Invalid request: No JSON data received."}), 400

    action = data.get('action')
    column = data.get('column')
    params = data.get('params', {})

    if not action: return jsonify({"error": "Cleaning 'action' parameter is required."}), 400
    if action in ['handle_nulls', 'convert_type', 'rename_column', 'drop_column'] and not column:
         return jsonify({"error": f"Action '{action}' requires a 'column' parameter."}), 400

    # --- Load Current DataFrame ---
    filepath = upload_doc.get('filepath')
    if not filepath or not os.path.exists(filepath):
         logging.error(f"Filepath missing or file does not exist for cleaning: {filepath}")
         return jsonify({"error": "Data file associated with this record is missing on server."}), 500
    df = get_dataframe(filepath)
    if df is None: return jsonify({"error": f"Failed to load the data file ({upload_doc.get('original_filename', 'N/A')})."}), 500

    original_shape = df.shape
    df_modified = df.copy() # IMPORTANT: Work on a copy
    cleaning_step_log = {"action": action, "column": column, "params": params, "timestamp": datetime.utcnow()}
    status_message = f"Action '{action}' applied." # Default success message

    # --- Apply Cleaning Logic ---
    logging.info(f"Applying cleaning '{action}' on '{column}' for {upload_id} by {session.get('username')}")
    try:
        # (Keep the extensive cleaning logic from the previous good example, including):
        # - handle_nulls (drop_row, drop_col, mean, median, mode, custom)
        # - convert_type (integer, float, string, category, datetime, boolean with safe coercion)
        # - remove_duplicates (with optional subset)
        # - rename_column
        # - drop_column
        # --- START Example Snippet ---
        if action == 'handle_nulls':
            method = params.get('method', 'drop_row')
            cleaning_step_log['method'] = method
            if column not in df_modified.columns: raise ValueError(f"Column '{column}' not found.")
            # ... (rest of handle_nulls logic: dropna, fillna with mean/median/mode[0]/custom) ...
            # ... (remember to update status_message specifically) ...
            # Example for mean fill:
            if method == 'mean':
                if pd.api.types.is_numeric_dtype(df_modified[column]):
                    fill_value = df_modified[column].mean()
                    df_modified[column].fillna(fill_value, inplace=True)
                    status_message = f"Filled nulls in '{column}' with mean ({fill_value:.3f})."
                    cleaning_step_log['fill_value'] = float(fill_value)
                else: raise ValueError("Mean requires numeric column")
            # ... (other methods) ...
        elif action == 'convert_type':
            new_type = params.get('new_type')
            if not new_type: raise ValueError("Parameter 'new_type' required.")
            if column not in df_modified.columns: raise ValueError(f"Column '{column}' not found.")
            cleaning_step_log['new_type'] = new_type
            original_dtype = str(df_modified[column].dtype)
            # ... (detailed type conversion logic as before using pd.to_numeric, astype, pd.to_datetime etc with error coercion) ...
            # Example:
            if new_type == 'integer': df_modified[column] = pd.to_numeric(df_modified[column], errors='coerce').astype('Int64')
            # ... (other types) ...
            final_dtype = str(df_modified[column].dtype)
            status_message = f"Converted '{column}' from '{original_dtype}' to '{final_dtype}'."
            # ... (check for new nulls) ...
        elif action == 'remove_duplicates':
             subset = params.get('subset')
             if isinstance(subset, list) and not subset: subset = None
             if subset and not all(c in df_modified.columns for c in subset): raise ValueError("Invalid subset column(s).")
             cleaning_step_log['subset'] = subset
             df_modified.drop_duplicates(subset=subset, inplace=True)
             rows_removed = original_shape[0] - df_modified.shape[0]
             cleaning_step_log['rows_removed'] = rows_removed
             status_message = f"Removed {rows_removed} duplicate rows" + (f" based on {subset}." if subset else ".")
        elif action == 'rename_column':
             new_name = params.get('new_name','').strip()
             if not new_name: raise ValueError("'new_name' required.")
             if column not in df_modified.columns: raise ValueError(f"Column '{column}' not found.")
             if new_name == column: raise ValueError("New name same as old.")
             if new_name in df_modified.columns: raise ValueError(f"Column '{new_name}' already exists.")
             df_modified.rename(columns={column: new_name}, inplace=True)
             cleaning_step_log['new_name'] = new_name
             status_message = f"Renamed '{column}' to '{new_name}'."
             column = new_name # Update for profiling if needed
        elif action == 'drop_column':
             if column not in df_modified.columns: raise ValueError(f"Column '{column}' not found.")
             df_modified.drop(columns=[column], inplace=True)
             status_message = f"Dropped column '{column}'."
             cleaning_step_log['column'] = column
             column = None # Column no longer exists
        else:
            raise ValueError(f"Unsupported cleaning action: '{action}'")
        # --- END Example Snippet ---


        # --- Save modified DataFrame back to the original file ---
        output_format = upload_doc['filepath'].lower().split('.')[-1]
        if output_format == 'csv':
            df_modified.to_csv(upload_doc['filepath'], index=False, encoding='utf-8') # Specify encoding
        elif output_format == 'xlsx':
            df_modified.to_excel(upload_doc['filepath'], index=False, engine='openpyxl')
        else:
            raise ValueError(f"Cannot save cleaned data: Unsupported file format '{output_format}'.")
        logging.info(f"Saved cleaned data ({df_modified.shape}) to {upload_doc['filepath']}")

        # --- Update DB Record ---
        new_profile = generate_data_profile(df_modified)
        # Ensure profile components are basic types for MongoDB
        new_profile['column_info'] = json.loads(json.dumps(new_profile.get('column_info', []), default=str))

        update_result = analysis_uploads_collection.update_one(
            {"_id": oid},
            {   "$push": {"cleaning_steps": cleaning_step_log},
                "$set": { "status": "cleaned", "row_count": new_profile.get('row_count', 0),
                          "col_count": new_profile.get('col_count', 0), "column_info": new_profile.get('column_info', []),
                          "last_modified": datetime.utcnow() } }
        )
        log_db_update_result(update_result, session.get('username'), f"cleaning_{upload_id}") # Use logging helper

        # --- Return updated data ---
        preview_data = df_modified.head(100).to_dict(orient='records')
        new_recommendations = generate_cleaning_recommendations(df_modified)

        return jsonify({
             "message": status_message, "preview_data": preview_data,
             "column_info": new_profile.get('column_info', []), "rows": new_profile.get('row_count', 0),
             "columns": new_profile.get('col_count', 0), "recommendations": new_recommendations
         }), 200

    except ValueError as ve: # Handle specific validation errors (e.g., bad params, column not found)
        logging.warning(f"Value error during cleaning action for {upload_id}: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as clean_err:
        logging.error(f"Cleaning action failed unexpectedly for {upload_id}: {clean_err}", exc_info=True)
        # Avoid saving potentially corrupted df_modified state
        return jsonify({"error": f"Failed to apply action '{action}'. An unexpected server error occurred."}), 500


@app.route('/run_analysis/<upload_id>/<analysis_type>', methods=['POST'])
def run_analysis(upload_id, analysis_type):
    # Ensure imports: logging, jsonify, request, session, ObjectId, analysis_uploads_collection,
    # pd, get_dataframe, json, log_db_update_result, datetime
    if not is_logged_in(): return jsonify({"error": "Authentication required."}), 401
    if analysis_uploads_collection is None: return jsonify({"error": "Database service unavailable."}), 503

    try: oid = ObjectId(upload_id); user_id = ObjectId(session['user_id'])
    except Exception as e: return jsonify({"error": f"Invalid ID format: {e}"}), 400

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: return jsonify({"error": "Analysis record not found or unauthorized."}), 404

    filepath = upload_doc.get('filepath')
    if not filepath or not os.path.exists(filepath): return jsonify({"error": "Data file missing for analysis."}), 500
    df = get_dataframe(filepath)
    if df is None: return jsonify({"error": "Failed to load data file for analysis."}), 500

    results = None
    update_key = f"analysis_results.{analysis_type}"
    status_message = f"Analysis '{analysis_type}' completed successfully."
    response_payload = {}

    logging.info(f"Running analysis '{analysis_type}' for {upload_id} by {session.get('username')}")
    try:
        # (Keep the analysis logic from the previous good example):
        # - descriptive_stats (using describe(include='all'), handling NaN for JSON)
        # - correlation (selecting numeric, handling empty/single column)
        # - value_counts (getting column param, using value_counts(dropna=False))
        # - Optional: group_by_agg, pivot_table etc.
        # --- START Example Snippet ---
        if analysis_type == 'descriptive_stats':
            stats_df = df.describe(include='all')
            # Convert NaN/NaT to None (or string 'NaN') for JSON compatibility
            stats_df_serializable = stats_df.astype(object).where(pd.notnull(stats_df), None)
            results = json.loads(stats_df_serializable.to_json(orient='index')) # Convert cleaned df to JSON dict
            # Prepare data suitable for simple HTML table display if needed
            response_payload["results_table"] = stats_df_serializable.reset_index().to_dict(orient='records')
            response_payload["results_raw"] = results
        elif analysis_type == 'correlation':
             numeric_df = df.select_dtypes(include='number')
             if numeric_df.empty: raise ValueError("No numeric columns found for correlation.")
             if len(numeric_df.columns) < 2: raise ValueError("At least two numeric columns required.")
             corr_matrix = numeric_df.corr()
             results = json.loads(corr_matrix.to_json(orient='index', default_handler=str)) # Use handler for safety
             response_payload["results_heatmap"] = {"z": corr_matrix.values.tolist(), "x": corr_matrix.columns.tolist(), "y": corr_matrix.index.tolist()}
             response_payload["results_raw"] = results
        elif analysis_type == 'value_counts':
             params = request.get_json() or {}
             column = params.get('column')
             if not column: raise ValueError("'column' required for value counts.")
             if column not in df.columns: raise ValueError(f"Column '{column}' not found.")
             counts = df[column].value_counts(dropna=False)
             results = json.loads(counts.to_json(orient='index', default_handler=str))
             response_payload["results_chart"] = {"labels": counts.index.astype(str).tolist(), "values": counts.values.tolist()}
             response_payload["results_raw"] = results
             status_message = f"Value counts for '{column}' calculated."
        # ... other analysis types ...
        else:
            return jsonify({"error": f"Unsupported analysis type: '{analysis_type}'."}), 400
        # --- END Example Snippet ---

        # --- Save results to DB ---
        if results is not None:
             update_result = analysis_uploads_collection.update_one(
                 {"_id": oid},
                 {"$set": {update_key: results, "status": "analyzed", "last_modified": datetime.utcnow()}}
             )
             log_db_update_result(update_result, session.get('username'), f"analysis_{upload_id}")
             response_payload["message"] = status_message
             return jsonify(response_payload), 200
        else:
             # Handle cases where analysis runs but produces no storable result (if any)
             logging.warning(f"Analysis '{analysis_type}' for {upload_id} produced null results.")
             return jsonify({"message": status_message, "results": None}), 200 # Still OK, just no data

    except ValueError as ve:
        logging.warning(f"Value error during analysis '{analysis_type}' for {upload_id}: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error during analysis '{analysis_type}' for {upload_id}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred during analysis."}), 500


@app.route('/generate_plot/<upload_id>', methods=['POST'])
def generate_plot(upload_id):
    # Ensure imports: logging, jsonify, request, session, ObjectId, analysis_uploads_collection,
    # pd, get_dataframe, json, pio (from plotly.io), px (from plotly.express)
    if not is_logged_in(): return jsonify({"error": "Authentication required."}), 401
    if analysis_uploads_collection is None: return jsonify({"error": "Database service unavailable."}), 503

    try: oid = ObjectId(upload_id); user_id = ObjectId(session['user_id'])
    except Exception as e: return jsonify({"error": f"Invalid ID format: {e}"}), 400

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: return jsonify({"error": "Analysis record not found or unauthorized."}), 404

    filepath = upload_doc.get('filepath')
    if not filepath or not os.path.exists(filepath): return jsonify({"error": "Data file missing for plotting."}), 500
    df = get_dataframe(filepath)
    if df is None: return jsonify({"error": "Failed to load data file for plotting."}), 500

    plot_config = request.get_json()
    if not plot_config: return jsonify({"error": "Invalid request: No plot configuration received."}), 400

    # --- Extract and Validate Config ---
    chart_type = plot_config.get('chart_type')
    x_col = plot_config.get('x')
    y_col = plot_config.get('y')
    color_col = plot_config.get('color')
    title = plot_config.get('title') # Optional custom title

    if not chart_type: return jsonify({"error": "Parameter 'chart_type' is required."}), 400
    supported_charts = ['histogram', 'scatter', 'bar', 'line', 'pie', 'box', 'heatmap']
    if chart_type not in supported_charts: return jsonify({"error": f"Unsupported chart type: '{chart_type}'."}), 400

    # Basic column existence checks (more specific checks happen in plotting logic)
    if x_col and x_col not in df.columns: return jsonify({"error": f"X-axis column '{x_col}' not found."}), 400
    if y_col and y_col not in df.columns: return jsonify({"error": f"Y-axis column '{y_col}' not found."}), 400
    if color_col and color_col not in df.columns: return jsonify({"error": f"Color column '{color_col}' not found."}), 400


    fig = None
    logging.info(f"Generating plot '{chart_type}' for {upload_id} by {session.get('username')}. Config: {plot_config}")

    # --- Generate Plot using Plotly ---
    try:
        # (Keep the plotting logic from the previous good example using px):
        # - histogram, scatter, bar (with count/aggregation logic), line, pie, box, heatmap
        # - Include basic error handling (e.g., checking numeric types where needed)
        # - Set appropriate titles
        # --- START Example Snippet ---
        plot_title = title or f"Plot: {chart_type.capitalize()}"

        if chart_type == 'histogram':
            if not x_col: raise ValueError("'x' column required for histogram.")
            plot_title = title or f'Distribution of {x_col}'
            fig = px.histogram(df, x=x_col, title=plot_title, color=color_col, marginal="rug")
        elif chart_type == 'scatter':
             if not x_col or not y_col: raise ValueError("'x' and 'y' columns required for scatter.")
             plot_title = title or f'{y_col} vs {x_col}'
             fig = px.scatter(df, x=x_col, y=y_col, title=plot_title, color=color_col, hover_data=df.columns)
        # ... other chart types with their specific logic and error checks ...
        elif chart_type == 'heatmap':
             corr_results = upload_doc.get('analysis_results', {}).get('correlation')
             if not corr_results: return jsonify({"error": "Run Correlation analysis first.", "action_needed": "run_correlation"}), 400
             try:
                 corr_df = pd.DataFrame.from_dict(corr_results)
                 plot_title = title or "Correlation Heatmap"
                 fig = px.imshow(corr_df, text_auto=True, title=plot_title, aspect="auto", color_continuous_scale='RdBu_r')
             except Exception as heat_err: raise ValueError(f"Failed to create heatmap: {heat_err}")
        else:
             raise ValueError(f"Plotting logic not fully implemented for chart type: '{chart_type}'") # Should not be reached if validation is correct
        # --- END Example Snippet ---


        # --- Post-processing and Response ---
        if fig:
            # Update layout for better appearance (optional)
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            # Convert to JSON using plotly.io
            graph_json_str = pio.to_json(fig)
            graph_json_obj = json.loads(graph_json_str) # Parse back to object for consistency
            # Optionally save plot to DB here if needed
            return jsonify({"message": "Plot generated successfully.", "plot_json": graph_json_obj}), 200
        else:
            # Should only happen if no logic path matched or fig became None unexpectedly
            return jsonify({"error": "Failed to generate the plot object."}), 500

    except ValueError as ve: # Handle specific validation/logic errors
        logging.warning(f"Value error during plot generation for {upload_id}: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as plot_err:
        logging.error(f"Plot generation failed unexpectedly for {upload_id}: {plot_err}", exc_info=True)
        return jsonify({"error": f"Failed to generate plot due to an unexpected server error."}), 500


@app.route('/generate_insights/<upload_id>', methods=['POST'])
def generate_insights(upload_id):
    # Ensure imports: logging, jsonify, session, ObjectId, analysis_uploads_collection, model,
    # get_dataframe, generate_data_profile, generate_gemini_insight_prompt, log_db_update_result, datetime,
    # log_gemini_response_details
    if not is_logged_in(): return jsonify({"error": "Authentication required."}), 401
    if analysis_uploads_collection is None: return jsonify({"error": "Database service unavailable."}), 503
    if model is None: return jsonify({"error": "AI model service unavailable."}), 503

    try: oid = ObjectId(upload_id); user_id = ObjectId(session['user_id'])
    except Exception as e: return jsonify({"error": f"Invalid ID format: {e}"}), 400

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: return jsonify({"error": "Analysis record not found or unauthorized."}), 404

    filepath = upload_doc.get('filepath')
    if not filepath or not os.path.exists(filepath): return jsonify({"error": "Data file missing for insights."}), 500
    df = get_dataframe(filepath)
    if df is None: return jsonify({"error": "Failed to load data file for insight generation."}), 500

    # Use profile info stored in DB for consistency with what user might see
    current_profile = {
        "row_count": upload_doc.get('row_count'), "col_count": upload_doc.get('col_count'),
        "column_info": upload_doc.get('column_info'),
        "memory_usage": upload_doc.get('memory_usage') # Assuming it was stored
    }
    cleaning_steps = upload_doc.get('cleaning_steps', [])

    prompt = generate_gemini_insight_prompt(current_profile, cleaning_steps)
    logging.info(f"Generating AI insights for {upload_id} by {session.get('username')}...")

    # --- Call Gemini API ---
    insights_text = "[AI Error: Failed to generate insights]"
    insights_list = [insights_text]

    try:
        response = model.generate_content(prompt)
        log_gemini_response_details(response, f"insights_{upload_id}")

        # (Keep the robust Gemini response parsing from the previous good example)
        # - Check block_reason, check candidates, extract text, parse into list
        # --- START Example Snippet ---
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason.name
             insights_text = f"[AI response blocked: {block_reason}]"; insights_list = [insights_text]
             logging.warning(f"Gemini insights BLOCKED for {upload_id}. Reason: {block_reason}")
        elif response.candidates:
            try:
                 raw_text = response.text
                 if not raw_text: insights_text = "[AI returned empty text]"; insights_list = [insights_text]
                 else:
                     insights_list = [ line.strip().lstrip('-* 1234567890.').strip()
                                       for line in raw_text.split('\n') if line.strip() and not line.strip().startswith('**') ]
                     if not insights_list: insights_list = [raw_text] # Fallback if parsing fails
                     insights_text = raw_text # Keep raw text for potential logging/debug
            except Exception as parse_err:
                 logging.error(f"Error parsing Gemini insights text for {upload_id}: {parse_err}"); insights_list = [insights_text]
        else: insights_text = "[AI returned no candidates]"; insights_list = [insights_text]; logging.warning(f"No candidates from Gemini for {upload_id}.")
         # --- END Example Snippet ---

    except Exception as ai_err:
        logging.error(f"Error calling Gemini API for insights ({upload_id}): {ai_err}", exc_info=True)
        # insights_list already defaults to error message

    # --- Save insights to DB ---
    try:
        update_result = analysis_uploads_collection.update_one(
            {"_id": oid},
            {"$set": {"generated_insights": insights_list, "last_modified": datetime.utcnow()}}
        )
        log_db_update_result(update_result, session.get('username'), f"insights_{upload_id}")
        logging.info(f"Stored generated insights ({len(insights_list)} items) for upload {upload_id}")
    except Exception as db_err:
         logging.error(f"Failed to save generated insights to DB for {upload_id}: {db_err}", exc_info=True)

    return jsonify({"message": "Insights processed.", "insights": insights_list}), 200


@app.route('/download/<upload_id>/cleaned_data/<fileformat>')
def download_cleaned_data(upload_id, fileformat):
    # Ensure imports: logging, flash, redirect, url_for, session, ObjectId, analysis_uploads_collection,
    # os, get_dataframe, io, send_file, pd, ExcelWriter
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    if analysis_uploads_collection is None: flash("DB unavailable.", "danger"); return redirect(url_for('data_analyzer_page'))

    try: oid = ObjectId(upload_id); user_id = ObjectId(session['user_id'])
    except Exception as e: flash("Invalid record identifier.", "danger"); logging.error(f"Invalid ID for download: {e}"); return redirect(url_for('analysis_history'))

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: flash("Record not found or access denied.", "danger"); return redirect(url_for('analysis_history'))

    filepath = upload_doc.get('filepath')
    if not filepath or not os.path.exists(filepath):
         flash("Data file missing on server.", "danger"); logging.error(f"Missing file for download: {filepath}")
         analysis_uploads_collection.update_one({"_id": oid}, {"$set": {"status": "error_file_missing", "last_modified": datetime.utcnow()}})
         return redirect(url_for('data_cleaner_page', upload_id=upload_id))

    original_filename_base, _ = os.path.splitext(upload_doc.get('original_filename', f'analysis_{upload_id}'))
    fileformat_lower = fileformat.lower()
    download_filename = f"{original_filename_base}_cleaned.{fileformat_lower}"

    mimetype = None
    if fileformat_lower == 'csv': mimetype = 'text/csv'
    elif fileformat_lower == 'xlsx': mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        flash(f"Invalid download format: '{fileformat}'. Allowed: csv, xlsx.", "warning")
        return redirect(url_for('data_cleaner_page', upload_id=upload_id))

    try:
        # Load the current dataframe state from file
        df = get_dataframe(filepath)
        if df is None: flash("Failed to load data file for download.", "danger"); return redirect(url_for('data_cleaner_page', upload_id=upload_id))

        # Prepare buffer in memory
        buffer = io.BytesIO()
        if fileformat_lower == 'csv':
            df.to_csv(buffer, index=False, encoding='utf-8-sig') # Use utf-8-sig for better Excel compatibility
        elif fileformat_lower == 'xlsx':
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer: # Use openpyxl engine
                df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
            # Buffer is populated after 'with' block exits

        buffer.seek(0) # Rewind buffer

        logging.info(f"Initiating download '{download_filename}' for {session.get('username')}, upload {upload_id}")
        return send_file(buffer, mimetype=mimetype, download_name=download_filename, as_attachment=True)

    except Exception as e:
        logging.error(f"Error preparing/sending cleaned data download for {upload_id}: {e}", exc_info=True)
        flash("Error preparing file for download.", "danger")
        return redirect(url_for('data_cleaner_page', upload_id=upload_id))


@app.route('/download/<upload_id>/pdf_report')
def download_pdf_report(upload_id):
    # Ensure imports: logging, flash, redirect, url_for, session, ObjectId, analysis_uploads_collection,
    # FPDF class definition, json, io, send_file, os, datetime
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    if analysis_uploads_collection is None: flash("DB unavailable.", "danger"); return redirect(url_for('data_analyzer_page'))

    try: oid = ObjectId(upload_id); user_id = ObjectId(session['user_id'])
    except Exception as e: flash("Invalid record identifier.", "danger"); logging.error(f"Invalid ID for PDF report: {e}"); return redirect(url_for('analysis_history'))

    upload_doc = analysis_uploads_collection.find_one({"_id": oid, "user_id": user_id})
    if not upload_doc: flash("Record not found or access denied.", "danger"); return redirect(url_for('analysis_history'))

    try:
        pdf = PDFReport(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Populate PDF Content ---
        # (Keep the PDF population logic from the previous good example):
        # - Section 1: Summary (Filename, Times, Rows, Cols, Column Table)
        # - Section 2: Cleaning Steps (Formatted list)
        # - Section 3: Analysis Results (Using add_json_block helper)
        # - Section 4: Visualizations (Placeholder or description)
        # - Section 5: AI Insights (Formatted list)
        # --- START Example Snippet ---
        # Section 1: Summary
        pdf.chapter_title('1. Data Summary')
        upload_time = upload_doc.get('upload_timestamp', datetime.utcnow())
        last_mod = upload_doc.get('last_modified', upload_time) # Use upload time if not modified
        summary_text = (f"Original Filename: {upload_doc.get('original_filename', 'N/A')}\n"
                       f"Upload Time: {upload_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                       f"Last Modified: {last_mod.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                       f"Status: {upload_doc.get('status', 'N/A')}\n"
                       f"Rows: {upload_doc.get('row_count', 'N/A')}\n"
                       f"Columns: {upload_doc.get('col_count', 'N/A')}\n\n"
                       "Column Details (Initial State):")
        pdf.chapter_body(summary_text)
        col_info = upload_doc.get('column_info', [])
        if col_info:
             col_header = ["Name", "Data Type", "Null Count"]
             col_data = [[c.get('name',''), c.get('dtype',''), c.get('null_count','N/A')] for c in col_info]
             col_widths = [pdf.w * 0.5 - pdf.l_margin, pdf.w * 0.25 - pdf.l_margin, pdf.w * 0.15 - pdf.l_margin] # Adjusted for margins
             pdf.add_table(col_header, col_data, col_widths=col_widths)
        else: pdf.chapter_body("Column info not available.")

        # Section 2: Cleaning Steps
        pdf.chapter_title('2. Cleaning Steps Applied')
        steps = upload_doc.get('cleaning_steps', [])
        if steps: # Format steps nicely
            step_text = ""
            for i, step in enumerate(steps):
                 step_time = step.get('timestamp', datetime.utcnow()).strftime('%H:%M:%S')
                 action = step.get('action', '?')
                 column = step.get('column', '')
                 params_str = json.dumps(step.get('params', {}), default=str, indent=None) # Compact params
                 step_text += f"[{step_time}] Step {i+1}: {action} "
                 if column: step_text += f"on '{column}' "
                 step_text += f"with params {params_str}\n"
            pdf.chapter_body(step_text)
        else: pdf.chapter_body("No cleaning steps recorded.")

        # Section 3: Analysis Results
        pdf.chapter_title('3. Analysis Results')
        analysis = upload_doc.get('analysis_results', {})
        if analysis:
            for name, data in analysis.items(): pdf.add_json_block(name.replace('_',' ').title(), data)
        else: pdf.chapter_body("No analysis results found.")

        # Section 4: Visualizations Placeholder
        pdf.chapter_title('4. Visualizations')
        pdf.chapter_body("(Visualizations generated interactively. Embeddings require saving plots as images during generation.)")

        # Section 5: AI Insights
        pdf.chapter_title('5. AI Generated Insights')
        insights = upload_doc.get('generated_insights', [])
        if insights: pdf.chapter_body("\n".join([f"- {insight}" for insight in insights]))
        else: pdf.chapter_body("No AI insights found.")
         # --- END Example Snippet ---


        # --- Generate PDF Output ---
        # Use latin-1 encoding compatible with FPDF default, replace unknown chars
        pdf_output_bytes = pdf.output(dest='S').encode('latin-1', errors='replace')
        buffer = io.BytesIO(pdf_output_bytes)
        buffer.seek(0)

        original_filename_base, _ = os.path.splitext(upload_doc.get('original_filename', f'analysis_{upload_id}'))
        download_filename = f"{original_filename_base}_report.pdf"

        logging.info(f"Generating PDF report '{download_filename}' for {session.get('username')}")
        return send_file(buffer, mimetype='application/pdf', download_name=download_filename, as_attachment=True)

    except Exception as e:
        logging.error(f"Error generating PDF report for {upload_id}: {e}", exc_info=True)
        flash("An error occurred while generating the PDF report.", "danger")
        return redirect(url_for('data_cleaner_page', upload_id=upload_id))


@app.route('/analysis_history')
def analysis_history():
    # Ensure imports: logging, flash, redirect, url_for, session, ObjectId, analysis_uploads_collection,
    # render_template, datetime
    if not is_logged_in(): flash("Please log in.", "warning"); return redirect(url_for('login'))
    if analysis_uploads_collection is None: flash("DB unavailable.", "danger"); return redirect(url_for('dashboard'))

    history = []
    try:
        user_id = ObjectId(session['user_id'])
        # Fetch history, sort by last modified (or upload if not modified), limit results
        history_cursor = analysis_uploads_collection.find(
            {"user_id": user_id},
            {"original_filename": 1, "upload_timestamp": 1, "last_modified": 1, "row_count": 1, "col_count": 1, "status": 1, "_id": 1}
        ).sort([("last_modified", -1), ("upload_timestamp", -1)]).limit(50)

        history = list(history_cursor)
        # Prepare data for template
        for item in history:
            item['_id'] = str(item['_id'])
            item['upload_timestamp_str'] = item.get('upload_timestamp').strftime('%Y-%m-%d %H:%M') if item.get('upload_timestamp') else 'N/A'
            # Show last modified only if it's different from upload time (or if it exists)
            last_mod = item.get('last_modified')
            upload_ts = item.get('upload_timestamp')
            item['last_modified_str'] = last_mod.strftime('%Y-%m-%d %H:%M') if last_mod and last_mod != upload_ts else '-'

        logging.info(f"Fetched {len(history)} history items for {session.get('username')}")

    except Exception as e:
        logging.error(f"Error fetching analysis history for {session.get('username')}: {e}", exc_info=True)
        flash("Error retrieving analysis history.", "danger")
        # Render page anyway, history list will be empty

    return render_template('analysis_history.html', history=history, now=datetime.utcnow())

# --- News Agent Routes (Using World News API) ---

@app.route('/news_agent')
def news_agent_page():
    # --- Minimal check ---
    key_value_at_route = WORLD_NEWS_API_KEY # Access the global variable directly
    is_present = bool(key_value_at_route)

    # --- Critical Log ---
    print(f"--- MINIMAL Route Check '/news_agent': Key Value='{key_value_at_route}', Passing available = {is_present} ---")

    # Directly render, bypassing login checks etc. for this TEST ONLY
    return render_template('news_agent.html',
                           news_api_available=is_present,
                           now=datetime.utcnow()) # Keep now if base template needs it

@app.route('/fetch_live_news')
def fetch_live_news():
    """Fetches news using the World News API."""
    logging.info("--- Enter /fetch_live_news (World News API) ---")

    if not is_logged_in():
        logging.warning("Fetch failed: Not logged in.")
        return jsonify({"error": "Authentication required"}), 401

    if not WORLD_NEWS_API_KEY:
        logging.error("Fetch failed: World News API key not configured.")
        return jsonify({"error": "News API key not configured on server."}), 503

    # --- Parameters for World News API ---
    # Ref: https://worldnewsapi.com/docs/#Search-News
    # Use sensible defaults, allow overrides via query parameters
    api_endpoint = "https://api.worldnewsapi.com/search-news"

    # Example Parameters (customize as needed)
    params = {
        'text': request.args.get('text', 'latest'), # Search query (default to 'latest')
        'source-countries': request.args.get('source-countries', 'us,gb'), # Example: US, UK
        'language': request.args.get('language', 'en'),
        'number': request.args.get('number', 25, type=int), # Number of results
        'sort': request.args.get('sort', 'publish-time'), # Sort by newest first
        'sort-direction': request.args.get('sort-direction', 'DESC'),
        # Add other params like 'earliest-publish-date', 'latest-publish-date', 'news-sources' as needed
        # 'earliest-publish-date': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d') # Example: last 24h
    }

    headers = {
        # Correct header name as per World News API docs
        'x-api-key': WORLD_NEWS_API_KEY
    }

    logging.info(f"Fetching World News API. Endpoint: {api_endpoint}, Params: { {k:v for k,v in params.items() if k != 'api-key'} }") # Log params without key

    try:
        response = requests.get(api_endpoint, headers=headers, params=params, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        news_data = response.json() # World News API returns JSON directly

        # --- Adapt Response Structure ---
        # World News API has a different structure than NewsAPI.org
        # The main list is usually under a 'news' key.
        articles = news_data.get('news', [])
        total_results = news_data.get('number', len(articles)) # 'number' in response might be limit, not total found

        # --- Map fields to the structure expected by frontend (if needed) ---
        # This maintains compatibility with news_agent.js expecting 'title', 'url', 'source.name', etc.
        mapped_articles = []
        for article in articles:
            mapped_articles.append({
                "title": article.get('title'),
                "description": article.get('text'), # Use 'text' field for description/summary
                "content": article.get('text'),    # Use 'text' field for content as well
                "url": article.get('url'),
                "urlToImage": article.get('image'), # Use 'image' field
                "publishedAt": article.get('publish_date'), # Use 'publish_date'
                "source": {
                    "id": None, # World News API might not provide a simple source ID like NewsAPI.org
                    "name": article.get('source_country') or article.get('authors') # Use country or authors as source name? Adjust as needed.
                 }
                # Add any other fields your frontend might use
            })


        # --- Optional: Store fetched articles in MongoDB ---
        # Adapt the storing logic if you implement it, using the new fields ('id', 'publish_date', etc.)
        # ... (MongoDB storage logic would go here, adjusted for World News API fields) ...

        logging.info(f"Fetched {len(mapped_articles)} articles via World News API successfully.")
        # Return the *mapped* articles so the frontend structure is consistent
        return jsonify({"articles": mapped_articles, "status": "ok", "totalResults": total_results})

    except requests.exceptions.Timeout:
         logging.error("World News API request timed out.")
         return jsonify({"error": "Request to news source timed out."}), 504
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        error_detail = f"HTTP error {status_code} from news source."
        try: # Try to parse the JSON error response from World News API / APILayer
            error_data = http_err.response.json()
            error_detail = error_data.get("message", error_detail) # Use 'message' field
            logging.error(f"World News API HTTP error {status_code}. Response: {error_data}")
        except ValueError:
            logging.error(f"World News API HTTP error {status_code}. Response was not JSON: {http_err.response.text[:200]}")
            pass # Use the default error detail
        # Map common status codes to user-friendly messages
        if status_code == 401: error_detail = "Invalid News API Key provided."
        elif status_code == 429: error_detail = "News API rate limit exceeded."
        elif status_code == 400: error_detail = f"Invalid request sent to news source: {error_detail}"

        return jsonify({"error": error_detail}), status_code
    except requests.exceptions.RequestException as req_err:
        logging.error(f"World News API request error: {req_err}")
        return jsonify({"error": "Could not connect to news source."}), 503
    except Exception as e:
        logging.error(f"Unexpected error fetching World News API: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred while fetching news."}), 500
    finally:
        logging.info("--- Exiting /fetch_live_news (World News API) ---")


@app.route('/summarize_news', methods=['POST'])
def summarize_news():
    """Summarizes news content using Gemini."""
    # This route remains largely the same, as it just takes text content
    if not is_logged_in(): return jsonify({"error": "Authentication required"}), 401
    if model is None: return jsonify({"error": "AI Summarizer not available."}), 503

    data = request.get_json()
    if not data: return jsonify({"error": "Invalid request: No JSON data."}), 400

    content_to_summarize = data.get('content')
    title = data.get('title', 'this news article')

    if not content_to_summarize:
        return jsonify({"error": "No content provided for summarization."}), 400

    # Check content length, maybe remove URL check if 'text' field is reliable
    if len(content_to_summarize) < 50: # Adjusted minimum length
        logging.warning("Content possibly too short for meaningful summary.")
        # Decide whether to proceed or return warning
        # return jsonify({"error": "Content too short for summarization."}), 400

    prompt = f"""Please provide a concise summary (around 2-3 sentences) of the following news article titled '{title}':

    "{content_to_summarize}"

    Focus on the main points and key information. Avoid introductory phrases like "The article discusses...".
    """

    logging.info("Sending content to Gemini for summarization...")
    summary = "[AI Error: Failed to generate summary]" # Default

    try:
        response = model.generate_content(prompt)
        log_gemini_response_details(response, f"summarize_{session.get('user_id')}") # Log Gemini response

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason.name
             summary = f"[AI summary blocked: {block_reason}]"
             logging.warning(f"Gemini summary BLOCKED. Reason: {block_reason}")
        elif response.candidates:
             try:
                 summary = response.text.strip()
                 if not summary: summary = "[AI returned empty summary]"
             except Exception as parse_err:
                 logging.error(f"Error parsing Gemini summary response: {parse_err}"); summary = "[Error parsing AI summary]"
        else:
             summary = "[AI returned no candidates]"; logging.warning("No candidates from Gemini for summary.")

    except Exception as e:
        logging.error(f"Error during Gemini summarization call: {e}", exc_info=True)
        # Keep default error message

    logging.info("Summary processing complete.")
    # Return summary even if it's an error message from AI/parsing
    return jsonify({"summary": summary})


# * NEW Route for Voice Agent Page *
@app.route('/voice_agent')
def voice_agent_page():
    """Renders the dedicated page for the Voice AI Assistant."""
    if not is_logged_in():
        flash("Please log in to use the Voice AI agent.", "warning")
        return redirect(url_for('login'))
    return render_template('voice_agent.html', now=datetime.utcnow())


# *************


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

# --- SocketIO Handlers for Voice Chat (/voice_chat namespace) ---
@socketio.on('connect', namespace='/voice_chat')
def handle_voice_connect():
    """Handles new client connections SPECIFICALLY to the /voice_chat namespace."""
    if not is_logged_in():
        logging.warning(f"Unauth connect /voice_chat: {request.sid}")
        return False # Reject connection

    user_id_str = session.get('user_id')
    username = session.get('username', 'Unknown')
    logging.info(f"User '{username}' (ID: {user_id_str}) connected to '/voice_chat' namespace. SID: {request.sid}")
    # Optional: Emit a confirmation back to this specific client
    emit('connection_ack', {'message': 'Successfully connected to voice chat.'}, room=request.sid, namespace='/voice_chat')

@socketio.on('disconnect', namespace='/voice_chat')
def handle_voice_disconnect():
    """Handles client disconnections from the /voice_chat namespace."""
    username = session.get('username', 'Unknown')
    logging.info(f"User '{username}' disconnected from '/voice_chat' namespace. SID: {request.sid}")


@socketio.on('send_voice_text', namespace='/voice_chat')
def handle_send_voice_text(data):
    """
    Handles transcribed text in MULTIPLE LANGUAGES (e.g., Hindi, English, German...).
    Instructs Gemini based on detected language and requests response in the SAME language.
    Saves language codes and interaction to MongoDB.
    """
    sid = request.sid
    # Add language to log message
    user_lang_from_payload = data.get('lang', 'en-US') # Get lang early for logging
    logging.info(f"--- Received 'send_voice_text' on /voice_chat (SID:{sid}, Lang:{user_lang_from_payload}) ---")

    # 1. Validation and Setup
    if not is_logged_in(): log_and_emit_error('Auth required.', sid); return
    if db is None or voice_conversations_collection is None: log_and_emit_error('DB service unavailable.', sid); return
    if model is None: log_and_emit_error('AI service unavailable.', sid); return

    username = session.get('username','Unknown_User'); user_id_str = session.get('user_id')
    if not isinstance(data, dict): logging.warning(f"Invalid data format from {username} (SID:{sid})"); return
    user_transcript = data.get('text', '').strip()
    # *** Use the language code sent from frontend ***
    user_lang = user_lang_from_payload # Assign to user_lang variable

    if not user_transcript: logging.debug(f"Empty transcript from {username} (SID:{sid}). Skipping."); return
    try: user_id = ObjectId(user_id_str)
    except Exception as e: log_and_emit_error(f"Invalid session identifier format: {e}", sid); return

    # Log received language clearly
    logging.info(f"Processing transcript from '{username}' (SID:{sid}, Detected Lang:{user_lang}): '{user_transcript[:60]}...'")

    # --- Prepare message data - include language ---
    user_msg_doc = {"role": "user", "text": user_transcript, "lang": user_lang}
    # Default error message for AI doc
    ai_response_text = "[AI Error: Processing failed]"
    # *** Assume AI response language matches user initially ***
    # This code tells the frontend TTS which language to *try* to use
    ai_lang = user_lang
    ai_msg_doc = {"role": "AI", "text": ai_response_text, "lang": ai_lang}

    # 2. Call Gemini API & Process Response (with MULTILINGUAL instruction)
    try:
        logging.debug(f"Attempting Gemini call for SID {sid} (Target Lang: {user_lang})...")

        # --- Build History (Optional, keep it concise) ---
        history = []
        try:
            convo_doc = voice_conversations_collection.find_one({"user_id": user_id}, {"messages": {"$slice": -4}}) # Limit history
            if convo_doc and "messages" in convo_doc:
                 for msg in convo_doc["messages"]:
                     if isinstance(msg, dict) and "role" in msg and "text" in msg:
                          # Use language from history if available for context, otherwise default
                          msg_lang = msg.get('lang', 'en-US')
                          # Prepend lang to historical part for clarity? (Optional)
                          # history_text = f"({msg_lang}): {msg['text']}"
                          history_text = msg['text']
                          history.append({'role': ('model' if msg['role']=='AI' else msg['role']), 'parts': [history_text]})
                 logging.info(f"Built history ({len(history)} messages) for {username} (SID:{sid})")
        except Exception as hist_err:
            logging.error(f"Error building history for SID {sid}: {hist_err}", exc_info=True)
            history = [] # Start fresh if history fails

        # --- Construct Multilingual Prompt ---
        # Explicitly tell Gemini the input language AND the desired output language.
        # Use clear language names if codes are ambiguous, or stick to codes if Gemini handles them well.
        # Example mapping (expand as needed):
        language_map = {
            'en-US': 'English', 'en-GB': 'English',
            'hi-IN': 'Hindi',
            'de-DE': 'German',
            'fr-FR': 'French',
            'es-ES': 'Spanish', 'es-MX': 'Spanish',
            'zh-CN': 'Mandarin Chinese', 'zh-TW': 'Mandarin Chinese',
            'ja-JP': 'Japanese',
            'ko-KR': 'Korean',
            # Add more mappings
        }
        # Get a human-readable language name for the prompt, default to code if not mapped
        language_name = language_map.get(user_lang, user_lang)

        multilingual_prompt_text = f"""
        **Role:** You are a multilingual voice assistant.
        **Context:** The user is speaking in '{language_name}' (language code: {user_lang}).
        **Task:** Respond DIRECTLY and conversationally IN THE SAME LANGUAGE ('{language_name}') to the user's input below.
        **Input ({language_name}):** "{user_transcript}"
        **Constraints:** Do NOT start with acknowledgements like "Okay, I understand..." or "I received...". Do NOT add filler like "How can I help?". Directly provide the answer/response in {language_name}.
        **Your Direct Response (in {language_name}):**
        """ # Note: Using language_name in the prompt might be more robust than just the code.

        # Prepare context for generate_content (history + final prompt)
        context_list = history
        context_list.append({'role': 'user', 'parts': [multilingual_prompt_text]})

        logging.debug(f"Sending context (length {len(context_list)}) to Gemini for SID {sid} requesting language {language_name}.")
        response = model.generate_content(context_list) # Use generate_content

        # --- Process Response ---
        logging.info(f"Received Gemini response for SID {sid}.")
        log_gemini_response_details(response, sid) # Log details

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             ai_response_text = f"[AI response blocked: {block_reason}]"
             logging.warning(f"Gemini response BLOCKED (Lang: {user_lang}) for SID {sid}. Reason: {block_reason}")
        elif response.candidates:
            try:
                ai_response_text = response.text
                if not ai_response_text:
                     finish_reason = response.candidates[0].finish_reason.name if hasattr(response.candidates[0], 'finish_reason') else "UNKNOWN"
                     ai_response_text = f"[AI returned empty text (Lang: {user_lang}). Finish: {finish_reason}]"
                     logging.warning(f"Gemini returned empty text (Lang: {user_lang}) for SID {sid}. Finish: {finish_reason}")
                else:
                     # Success! Assume response language is correct per instruction.
                     ai_lang = user_lang # Keep the language we requested
            except (AttributeError, ValueError, IndexError) as e:
                 ai_response_text = "[AI response format error]"
                 logging.error(f"Error extracting text (Lang: {user_lang}) from Gemini candidates for SID {sid}: {e}")
        else:
             ai_response_text = "[AI returned no candidates]"
             logging.warning(f"Gemini returned no candidates (Lang: {user_lang}) for SID {sid}.")

    except Exception as e:
        logging.error(f"CRITICAL ERROR during Gemini API call/processing (Lang: {user_lang}) for SID {sid}: {e}", exc_info=True)
        ai_response_text = "[Server error during AI processing]"
        # Keep ai_lang as user_lang even on error

    # Update AI message doc with final text and determined language
    ai_msg_doc["text"] = ai_response_text
    ai_msg_doc["lang"] = ai_lang # Use the determined language

    # 3. Save Conversation Turn to MongoDB (including language fields)
    if voice_conversations_collection is not None:
        try:
            # ... (DB saving logic using update_one with $push as before) ...
            # Ensure user_msg_doc and ai_msg_doc (with 'lang') are pushed
            current_time = datetime.utcnow()
            user_msg_doc["timestamp"] = current_time
            ai_msg_doc["timestamp"] = current_time
            logging.debug(f"Saving turn for user {user_id} (Lang: {user_lang} -> {ai_lang})")
            update_result = voice_conversations_collection.update_one(
                 {"user_id": user_id},
                 {"$push": {"messages": {"$each": [user_msg_doc, ai_msg_doc]}},
                 "$setOnInsert": {"user_id": user_id, "username": username, "start_timestamp": current_time}},
                 upsert=True
            )
            log_db_update_result(update_result, username, sid)
        except Exception as e:
            logging.error(f"Error saving multilingual conversation to MongoDB (SID:{sid}): {e}", exc_info=True)
    else:
        logging.warning(f"MongoDB collection unavailable. Skipping conversation save for SID:{sid}.")

    # 4. Emit AI Response back to Client (including language code)
    response_payload = {
        'user': 'AI',
        'text': ai_response_text,
        'lang': ai_lang # *** Send the correct language code back ***
    }
    try:
        logging.info(f"Emitting 'receive_ai_voice_text' to SID {sid} (Payload Lang: {ai_lang})")
        emit( 'receive_ai_voice_text', response_payload, room=sid, namespace='/voice_chat' )
        logging.info(f"Successfully called emit for SID {sid}.")
    except Exception as e:
         logging.error(f"CRITICAL ERROR emitting response (Lang: {ai_lang}) to SID {sid}: {e}", exc_info=True)

    logging.info(f"--- Finished handling 'send_voice_text' (Lang: {user_lang}) for SID {sid} ---")


# --- Helper function for logging Gemini response ---
def log_gemini_response_details(response, sid):
    """Logs details of the Gemini response object."""
    logging.debug(f"--- Gemini Response Details (SID:{sid}) ---")
    try:
        logging.debug(f"Response Object Type: {type(response)}")
        if hasattr(response, 'candidates') and response.candidates:
             logging.debug(f"Candidates Count: {len(response.candidates)}")
             for i, candidate in enumerate(response.candidates):
                 logging.debug(f"  Candidate[{i}]:")
                 if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                      logging.debug(f"    Content Parts: {str(candidate.content.parts)[:200]}...") # Log snippet
                 else: logging.debug(f"    Content: No parts or empty content")
                 if hasattr(candidate, 'finish_reason'): logging.debug(f"    Finish Reason: {candidate.finish_reason.name}")
                 else: logging.debug(f"    Finish Reason: N/A")
                 if hasattr(candidate, 'safety_ratings'): logging.debug(f"    Safety Ratings: {candidate.safety_ratings}")
        else: logging.debug("Candidates: None or empty list.")

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
             logging.debug(f"Prompt Feedback: {response.prompt_feedback}")
             if response.prompt_feedback.block_reason: logging.warning(f"*** PROMPT BLOCKED (in details)! Reason: {response.prompt_feedback.block_reason} ***")
        else: logging.debug("Prompt Feedback: Not available or empty.")

        if hasattr(response, 'text'): logging.debug(f"Text Attribute: '{str(response.text)[:200]}...'")
        else: logging.debug("Text Attribute: Does not exist.")
    except Exception as log_err: logging.error(f"Error occurred while logging response details for SID {sid}: {log_err}")
    logging.debug(f"--- End Gemini Response Details (SID:{sid}) ---")

# --- Helper function for logging DB result ---
def log_db_update_result(update_result, username, sid):
    """Logs details of the MongoDB update_one result."""
    try:
        logging.debug(f"DB Update Result for {username} (SID:{sid}): Matched={update_result.matched_count}, Modified={update_result.modified_count}, UpsertedId={update_result.upserted_id}, Ack={update_result.acknowledged}")
        if update_result.acknowledged:
            if update_result.upserted_id: logging.info(f"DB: INSERTED new doc for {username}. ID: {update_result.upserted_id}")
            elif update_result.modified_count > 0: logging.info(f"DB: UPDATED existing doc for {username}.")
            elif update_result.matched_count == 1 and update_result.modified_count == 0: logging.warning(f"DB: MATCHED doc for {username} but MODIFIED 0. Check 'messages' field type!")
            elif update_result.matched_count == 0 and not update_result.upserted_id: logging.error(f"DB ERROR for {username}: Matched 0 and no upsert ID. Upsert failed?")
        else: logging.error(f"DB ERROR for {username}: Update command was NOT acknowledged by server.")
    except Exception as log_db_err: logging.error(f"Error logging DB update result for SID {sid}: {log_db_err}")

# --- Helper function for emitting errors ---
def log_and_emit_error(message, sid):
    """Logs an error and emits it back to the client."""
    logging.error(f"Error for SID {sid}: {message}")
    try:
        emit('error', {'message': message}, room=sid, namespace='/voice_chat')
    except Exception as e:
         logging.error(f"Failed to emit error '{message}' to SID {sid}: {e}", exc_info=True)

# --- END of SocketIO Handlers for Voice Chat ---

# -------------------------------------------------------------------


# --- Main Execution ---
if __name__ == '__main__':
    if db is None: logging.critical("MongoDB connection failed. Aborting."); exit(1)
    if not app.config['SECRET_KEY'] or app.config['SECRET_KEY'] == 'dev-secret-key-only-not-for-production!': logging.warning("WARNING: Running with insecure default FLASK_SECRET_KEY!")
    logging.info("Starting Flask-SocketIO server...")
    try:
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except Exception as e:
        logging.critical(f"Failed to start server: {e}"); logging.critical(traceback.format_exc())