import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session # session might not be used directly here but good practice
from flask_socketio import SocketIO, emit, join_room, leave_room, send
from dotenv import load_dotenv
import logging
import json
from threading import Lock
import traceback # Import traceback for detailed error logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Flask App and SocketIO Initialization ---
app = Flask(__name__,
            template_folder='src/templates',
            static_folder='src/static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_dev_only!') # Use env var in prod
# Ensure eventlet is installed: pip install eventlet
# Use eventlet for async mode, recommended for SocketIO performance
socketio = SocketIO(app, async_mode='eventlet')

# --- Gemini API Configuration ---
api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-1.5-flash" # Or gemini-pro for potentially better results
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        # Define default safety settings (adjust thresholds as needed)
        # Blocking thresholds can be: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
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
        logging.error(traceback.format_exc()) # Log full traceback
else:
    # Log a critical error if the API key is missing, as the app won't work
    logging.critical("CRITICAL: GEMINI_API_KEY not found in environment variables. AI features will not work.")

# --- In-memory storage for active chats ---
# WARNING: Simple in-memory storage. Lost on server restart. Use Redis/DB for production.
active_chats = {}
chats_lock = Lock() # To prevent race conditions when accessing active_chats dictionary

# --- Helper function to get or start chat ---
def get_or_start_chat(sid, report_context=""):
    """Gets the existing chat session for a SID or starts a new one."""
    with chats_lock:
        if sid not in active_chats:
            if not model:
                logging.error(f"Cannot start chat for {sid}: Model not initialized.")
                return None, None
            try:
                logging.info(f"Starting new chat for SID {sid} with context length: {len(report_context)}")
                initial_history = []
                # Use system_instruction if your model version supports it well, otherwise prepend context
                # system_instruction = "You are a helpful assistant..."

                if report_context:
                     # Prepending context as user/model messages to guide the AI
                     initial_history.append({'role': 'user', 'parts': [f"This is the report we are discussing:\n\n{report_context}"]})
                     initial_history.append({'role': 'model', 'parts': ["Understood. I have the report context. How can I assist you with it?"]})

                # Check Gemini docs for the best way to pass system instructions if needed
                chat = model.start_chat(history=initial_history) # Pass history here

                active_chats[sid] = {
                    'chat_session': chat,
                    'report': report_context # Store the report context for reference if needed later
                }
                logging.info(f"Successfully started and stored new chat session for SID: {sid}")

            except Exception as e:
                logging.error(f"Failed to start chat session for {sid}: {e}")
                logging.error(traceback.format_exc())
                return None, None

        # Return the chat session object and the stored report context
        chat_info = active_chats.get(sid)
        if chat_info:
             return chat_info.get('chat_session'), chat_info.get('report')
        else:
             # Should not happen if lock is working, but defensive check
             logging.error(f"Chat info unexpectedly missing for SID {sid} after check.")
             return None, None

# --- HTTP Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # Using SID for tracking, Flask session not strictly required for this logic
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """Handles the report generation request."""
    logging.info("Received request for /generate_report")
    # 1. Check Model Availability
    if not model:
        logging.error("/generate_report: AI model not available.")
        # Return 503 Service Unavailable if the core component isn't ready
        return jsonify({"error": "AI model not available. Please try again later or check server status."}), 503

    # 2. Validate Request
    if not request.is_json:
        logging.warning("/generate_report: Request is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    input_text = data.get('text')

    if not input_text or not isinstance(input_text, str) or len(input_text.strip()) == 0:
        logging.warning("/generate_report: No valid text provided in request.")
        return jsonify({"error": "No valid text provided"}), 400

    logging.info(f"/generate_report: Processing text (length: {len(input_text)}).")

    # 3. Define the Prompt (Ensure this is your complete, detailed prompt)
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
    """

    # 4. Call Gemini API and Handle Response/Errors
    generated_text = ""
    response = None

    try:
        logging.info("/generate_report: Sending request to Gemini...")
        # Generation config can be set here if needed per-request
        # generation_config = genai.types.GenerationConfig(temperature=0.7)
        response = model.generate_content(prompt) # Safety settings applied during model init
        logging.info("/generate_report: Received response from Gemini.")

        # --- Validate Gemini Response ---
        if not response.candidates:
             logging.error("/generate_report: Gemini response has no candidates.")
             if hasattr(response, 'prompt_feedback'): logging.error(f"Prompt Feedback: {response.prompt_feedback}")
             # Return specific error about blocked/empty response
             return jsonify({"error": "AI response was empty or blocked. This might be due to safety settings or the prompt content."}), 500

        first_candidate = response.candidates[0]
        finish_reason_name = first_candidate.finish_reason.name
        logging.info(f"/generate_report: Gemini finish reason: {finish_reason_name}")

        # Check finish reason - other than STOP or MAX_TOKENS might indicate issues
        if finish_reason_name not in ["STOP", "MAX_TOKENS"]:
            logging.warning(f"/generate_report: Generation finished unexpectedly: {finish_reason_name}")
            if hasattr(response, 'prompt_feedback'): logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
            # Decide whether to return an error or try using partial content
            # Returning an error might be safer:
            return jsonify({"error": f"AI generation stopped unexpectedly ({finish_reason_name}). Check safety settings or prompt complexity."}), 500

        # Check for content parts
        if not hasattr(first_candidate, 'content') or not hasattr(first_candidate.content, 'parts') or not first_candidate.content.parts:
             logging.error("/generate_report: Gemini response candidate has no content parts.")
             return jsonify({"error": "AI response content is missing."}), 500

        # Extract text safely - usually in the first part for simple text models
        generated_text = first_candidate.content.parts[0].text
        logging.info(f"/generate_report: Extracted text (length: {len(generated_text)}).")

        # 5. Parse Response Content (Report + JSON)
        report_content = generated_text # Default to full text
        chart_data_json = {}
        json_string = None
        try:
            json_start_marker = "```json_chart_data"
            json_end_marker = "```"
            # Use rfind to find the *last* occurrence, assuming JSON is at the end
            start_index = generated_text.rfind(json_start_marker)

            if start_index != -1:
                logging.info("/generate_report: Found json_chart_data start marker.")
                start_index_content = start_index + len(json_start_marker)
                end_index = generated_text.find(json_end_marker, start_index_content)

                if end_index != -1:
                    logging.info("/generate_report: Found json_chart_data end marker.")
                    json_string = generated_text[start_index_content:end_index].strip()
                    try:
                        chart_data_json = json.loads(json_string)
                        report_content = generated_text[:start_index].strip() # Get text before the marker
                        logging.info("/generate_report: Successfully parsed chart data JSON.")
                    except json.JSONDecodeError as json_err:
                        logging.error(f"/generate_report: Failed to decode JSON: {json_err}")
                        logging.warning(f"/generate_report: Raw string that failed JSON parsing: '{json_string}'")
                        # Fallback: Keep full generated_text as report_content
                        report_content = generated_text
                else:
                     logging.warning("/generate_report: Found JSON start marker but no end marker after it.")
            else:
                logging.warning("/generate_report: JSON chart data block marker ```json_chart_data not found in AI response.")

        except Exception as parse_err:
             logging.error(f"/generate_report: Unexpected error during parsing AI response: {parse_err}")
             logging.error(traceback.format_exc())
             report_content = generated_text # Fallback to full text

        # 6. Prepare and Return Successful JSON Response
        # Limit context size for chat initiation (e.g., first 3000 chars)
        report_context_for_chat = report_content[:3000] if report_content else ""

        logging.info("/generate_report: Successfully processed request. Sending response to client.")
        return jsonify({
            "report_html": report_content,
            "chart_data": chart_data_json,
            "report_context_for_chat": report_context_for_chat
        })

    # 7. Catch-All Exception Handler for the Route
    except Exception as e:
        logging.error(f"CRITICAL ERROR processing /generate_report: {e}")
        logging.error(traceback.format_exc()) # Log the full stack trace
        # Try to log feedback if response object exists (might not if error was early)
        if response and hasattr(response, 'prompt_feedback'):
             logging.warning(f"Prompt Feedback on Error: {response.prompt_feedback}")
        # Return a generic server error message to the client
        return jsonify({"error": f"An unexpected server error occurred while generating the report."}), 500


# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new WebSocket connections."""
    # *** ADDED SERVER LOG ***
    logging.info(f"SERVER LOG: Client connected successfully: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections and cleans up chat history."""
    # *** ADDED SERVER LOG ***
    logging.info(f"SERVER LOG: Client disconnected event received for: {request.sid}")
    with chats_lock:
        if request.sid in active_chats:
            del active_chats[request.sid]
            logging.info(f"SERVER LOG: Cleaned up chat session for SID: {request.sid}")
        else:
            # This is normal if the user disconnected before starting a chat
            logging.info(f"SERVER LOG: No active chat session found to clean up for SID: {request.sid}")

@socketio.on('send_message')
def handle_send_message(data):
    """Handles incoming chat messages from a client."""
    sid = request.sid
    user_message = data.get('text')
    report_context = data.get('report_context') # Used only for *starting* the chat

    if not user_message or not isinstance(user_message, str) or len(user_message.strip()) == 0:
        logging.warning(f"Received empty or invalid message from {sid}.")
        # Optionally emit an error back, or just ignore
        # emit('error', {'message': 'Cannot send empty message.'}, room=sid)
        return

    logging.info(f"Received message from {sid}: '{user_message[:50]}...'")

    # Get the existing chat session or start a new one
    # Pass context only if starting a new chat
    chat_session, _ = get_or_start_chat(sid, report_context if sid not in active_chats else "")

    if not chat_session:
        logging.error(f"Chat session not found or failed to initialize for {sid}")
        emit('error', {'message': 'Could not initialize chat session. Please try refreshing.'}, room=sid)
        return

    # Send message to Gemini via the chat session
    try:
        emit('typing_indicator', {'isTyping': True}, room=sid)
        logging.info(f"Sending message to Gemini for SID {sid}...")
        response = chat_session.send_message(user_message)
        logging.info(f"Received chat response from Gemini for SID {sid}.")

        # --- Validate Chat Response ---
        ai_response_text = "[Error: AI response could not be processed]" # Default error text
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # Check finish reason (optional but good practice)
            finish_reason_name = response.candidates[0].finish_reason.name
            if finish_reason_name not in ["STOP", "MAX_TOKENS"]:
                logging.warning(f"Chat generation for {sid} finished with reason: {finish_reason_name}")
                if hasattr(response, 'prompt_feedback'): logging.warning(f"Chat Prompt Feedback: {response.prompt_feedback}")
                ai_response_text = f"[AI response may be incomplete, stopped due to: {finish_reason_name}]"
            else:
                # Safely access text using the .text accessor
                ai_response_text = response.text
        else:
             logging.error(f"Chat response for {sid} is empty or blocked.")
             if hasattr(response, 'prompt_feedback'): logging.error(f"Chat Prompt Feedback: {response.prompt_feedback}")
             ai_response_text = "[Error: AI response was empty or blocked by safety settings]"

        logging.info(f"Sending AI response to {sid}: '{ai_response_text[:50]}...'")
        emit('receive_message', {'user': 'AI', 'text': ai_response_text}, room=sid)

    except Exception as e:
        logging.error(f"Error processing message for {sid}: {e}")
        logging.error(traceback.format_exc())
        emit('error', {'message': f'An error occurred while communicating with the AI.'}, room=sid)
    finally:
        # Ensure typing indicator is turned off
        emit('typing_indicator', {'isTyping': False}, room=sid)


# --- Main Execution ---
if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server...")
    try:
        # *** Run with debug=True (for development feedback) but use_reloader=False (for WebSocket stability) ***
        # Use host='127.0.0.1' (localhost) unless you need access from other devices on your network (then use '0.0.0.0')
        socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except ValueError as ve:
        # Catch the specific async_mode error if eventlet still not installed
        if 'Invalid async_mode specified' in str(ve):
             logging.critical("ASYNC MODE ERROR: 'eventlet' is required but likely not installed.")
             logging.critical("Please install it: pip install eventlet")
             logging.critical("Alternatively, change async_mode in app.py to 'threading' (less performant).")
        else:
             logging.error(f"Error starting server: {ve}")
             logging.error(traceback.format_exc())
    except Exception as e:
        # Catch other potential startup errors (e.g., port in use)
        logging.critical(f"Failed to start server: {e}")
        logging.critical(traceback.format_exc())