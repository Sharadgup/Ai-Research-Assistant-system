// script.js (Combined & Corrected for Voice UI with Socket.IO)

// Ensure the Socket.IO client library is included in your HTML:
// <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>

document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const micButton = document.getElementById('mic-button');
    const micIcon = document.getElementById('mic-icon');
    const statusMessage = document.getElementById('status-message');
    const userChatArea = document.getElementById('user-chat-area');
    const agentChatArea = document.getElementById('agent-chat-area');
    const exitButton = document.getElementById('exit-button');
    const dashboardButton = document.getElementById('dashboard-button');

    // *** VERIFY ALL ELEMENTS EXIST ***
    if (!micButton || !micIcon || !statusMessage || !userChatArea || !agentChatArea || !exitButton || !dashboardButton) {
        console.error("FATAL ERROR: One or more required DOM elements (mic-button, mic-icon, status-message, user-chat-area, agent-chat-area, exit-button, dashboard-button) were not found. Please check your HTML IDs.");
        // Display error to user if statusMessage exists
        if (statusMessage) statusMessage.textContent = "UI Error: Elements missing.";
        // Stop script execution if critical elements are missing
        return;
    }
    console.log("All required DOM elements found.");

    // --- Web Speech API Setup (with prefix check) ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const SpeechSynthesis = window.speechSynthesis;

    // --- Check for browser support ---
    if (!SpeechRecognition) {
        console.error("Web Speech API (SpeechRecognition/webkitSpeechRecognition) not supported.");
        statusMessage.textContent = "Voice input not supported by browser.";
        micButton.disabled = true;
        micButton.style.cursor = 'not-allowed';
        micButton.title = "Voice input not supported";
        micButton.style.backgroundColor = '#ccc';
        return; // Stop script
    } else {
         console.log("SpeechRecognition API supported.");
    }
    if (!SpeechSynthesis) {
        console.warn("Speech Synthesis API not supported. Responses will not be spoken.");
    } else {
         console.log("SpeechSynthesis API supported.");
    }

    // --- Socket.IO Connection ---
    console.log("Attempting to connect to /voice_chat namespace...");
    // const socket = io('http://127.0.0.1:5001/voice_chat'); // Use full URL if needed
    const socket = io('/voice_chat'); // Assumes same host/port

    // --- Socket.IO Event Listeners ---
    socket.on('connect', () => {
        console.log('Socket connected to /voice_chat! SID:', socket.id);
        statusMessage.textContent = "Connected. Click mic.";
        micButton.disabled = false; // Enable mic button on connect
        micButton.style.cursor = 'pointer';
    });

    socket.on('connect_error', (err) => {
        console.error('Socket Connection Error to /voice_chat:', err.message);
        statusMessage.textContent = "Connection Error.";
        micButton.disabled = true; // Disable on error
        micButton.style.cursor = 'not-allowed';
    });

    socket.on('disconnect', (reason) => {
        console.log('Socket disconnected from /voice_chat:', reason);
        statusMessage.textContent = "Disconnected.";
        micButton.disabled = true; // Disable on disconnect
        micButton.style.cursor = 'not-allowed';
    });

    socket.on('connection_ack', (data) => { // Optional listener
         console.log('Backend ACK:', data.message);
    });

    // Listener for receiving AI responses
    socket.on('receive_ai_voice_text', (data) => {
        console.log("Received 'receive_ai_voice_text':", data);
        if (data && data.text) {
             displayMessage(data.text, 'agent');
             speakText(data.text, data.lang); // Pass language
        } else {
             console.warn("Received voice response event, but data is missing text:", data);
             displayMessage("[Received empty/invalid AI response]", 'agent');
        }
        // Re-enable mic after processing if interaction still active
        if (interactionActive) {
            micButton.classList.remove('processing');
            micButton.disabled = false;
            statusMessage.textContent = "Click mic to speak";
        }
    });

    // Listener for backend errors
    socket.on('error', (data) => {
        console.error('Received error event from backend:', data.message);
        statusMessage.textContent = `Error: ${data.message}`;
        // Reset button state if needed
        if (interactionActive) {
             micButton.classList.remove('processing');
             micButton.disabled = false;
        }
    });

    // --- Web Speech API Variables ---
    let recognition;
    let isRecording = false;
    let finalTranscript = '';
    let interactionActive = true;

    // --- Initialize Recognition Function ---
    const initializeRecognition = () => {
        console.log("Initializing SpeechRecognition...");
        if (recognition) { try { recognition.stop(); } catch (e) {console.warn("Error stopping previous recog:", e);} }

        try {
            recognition = new SpeechRecognition(); // Use the checked variable
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US'; // Default language

            // --- Assign Event Handlers ---
            recognition.onstart = () => {
                console.log("Recording started");
                isRecording = true;
                finalTranscript = '';
                micButton.classList.add('recording');
                micIcon.classList.remove('fa-microphone');
                micIcon.classList.add('fa-stop');
                statusMessage.textContent = "Listening...";
                clearPlaceholders();
            };

            recognition.onresult = (event) => {
                if (!interactionActive) return;
                let interimTranscript = '';
                finalTranscript = ''; // Reset on each result event, build up final
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const transcriptPart = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcriptPart + ' ';
                    } else {
                        interimTranscript += transcriptPart;
                    }
                }
                 // Optional: displayInterimMessage(interimTranscript || finalTranscript || "...");
            };

            recognition.onend = () => {
                console.log("Recognition ended.");
                isRecording = false; // Update state
                micButton.classList.remove('recording'); // Always remove class
                micIcon.classList.remove('fa-stop');
                micIcon.classList.add('fa-microphone');

                // Decide what to do based on whether interaction is active
                if (!interactionActive) {
                    statusMessage.textContent = "Interaction stopped. Click mic.";
                    micButton.disabled = false; // Make sure it's enabled
                    return;
                }

                finalTranscript = finalTranscript.trim(); // Clean final transcript

                if (finalTranscript) {
                    statusMessage.textContent = "Processing...";
                    micButton.classList.add('processing');
                    micButton.disabled = true; // Disable WHILE processing
                    console.log("Mic button DISABLED for processing in onend.");
                    displayMessage(finalTranscript, 'user');
                    // *** SEND TRANSCRIPT VIA SOCKET.IO ***
                    sendTranscriptToBackendViaSocket(finalTranscript, recognition.lang);
                } else {
                    statusMessage.textContent = "No speech detected. Click mic.";
                    micButton.disabled = false; // Enable if nothing detected
                    console.log("Mic button RE-ENABLED in onend (no speech detected).");
                }
            };

            recognition.onerror = (event) => {
                 console.error("Speech recognition error:", event.error, event.message);
                 let errorMsg = "Error: " + event.error;
                 if (event.error === 'no-speech') errorMsg = "No speech detected.";
                 if (event.error === 'audio-capture') errorMsg = "Microphone error.";
                 if (event.error === 'not-allowed') errorMsg = "Permission denied.";
                 if (event.error === 'aborted') errorMsg = "Listening stopped.";

                 if (interactionActive) statusMessage.textContent = errorMsg;

                 // Reset state completely on error
                 isRecording = false;
                 micButton.classList.remove('recording', 'processing');
                 micIcon.classList.remove('fa-stop');
                 micIcon.classList.add('fa-microphone');
                 if (interactionActive) {
                      micButton.disabled = false; // Re-enable button
                      console.log("Mic button RE-ENABLED on error.");
                 }
            };
            console.log("SpeechRecognition initialized successfully.");
        } catch (initError) {
             console.error("FATAL: Error initializing SpeechRecognition instance:", initError);
             statusMessage.textContent = "Error initializing voice input.";
             if(micButton) micButton.disabled = true; // Disable if init fails
        }
    }

    // --- Initialize on Load ---
    initializeRecognition();

    // --- UI Event Listeners ---
    micButton.addEventListener('click', () => {
        console.log("Mic button CLICKED. Current state: isRecording=", isRecording, "socket.connected=", socket.connected, "micButton.disabled=", micButton.disabled);
        if (micButton.disabled) { console.warn("Mic button is disabled, click ignored."); return; }
        if (!socket.connected) { statusMessage.textContent = "Connecting..."; console.warn("Mic clicked but socket not connected."); return; }
        if (!interactionActive) { interactionActive = true; hideExitMessage(); }

        if (isRecording) {
            try { console.log("Attempting to stop recognition..."); recognition.stop(); }
            catch(e) { console.error("Error stopping recognition:", e); /* Manual reset? */ }
        } else {
            if (SpeechSynthesis && SpeechSynthesis.speaking) { SpeechSynthesis.cancel(); }
            try { console.log("Attempting to start recognition..."); recognition.start(); }
            catch (error) {
                console.error("Error starting recognition:", error);
                if (error.name === 'InvalidStateError') { console.warn("Recognition in invalid state. Re-initializing..."); initializeRecognition(); statusMessage.textContent = "Voice input reset. Try again."; }
                else { statusMessage.textContent = "Could not start. Mic busy?"; }
                micButton.disabled = false; // Ensure button usable
            }
        }
    });

    exitButton.addEventListener('click', () => {
        console.log("Exit clicked");
        interactionActive = false;
        if (isRecording) { recognition.abort(); } // Abort immediately
        if (SpeechSynthesis && SpeechSynthesis.speaking) { SpeechSynthesis.cancel(); }
        statusMessage.textContent = "Interaction stopped. Click mic to restart.";
        // Reset button state visually and functionally
        micButton.classList.remove('recording', 'processing');
        micIcon.classList.remove('fa-stop');
        micIcon.classList.add('fa-microphone');
        micButton.disabled = false; // Allow restarting
        isRecording = false; // Ensure state flag is reset
        // clearChatAreas(); // Optional
    });

    dashboardButton.addEventListener('click', () => {
        console.log("Back to Dashboard clicked - Implement navigation");
        alert("Dashboard navigation not implemented in this example.");
        // window.location.href = '/dashboard'; // Example redirect
    });

    // --- Helper Functions ---
    function sendTranscriptToBackendViaSocket(transcript, language) {
        if (!socket.connected) {
             console.error("Socket not connected when trying to send.");
             statusMessage.textContent = "Not connected.";
             micButton.classList.remove('processing'); // Reset button state
             micButton.disabled = false;
             return;
        }
        const payload = { text: transcript, lang: language || 'en-US' };
        console.log("Emitting 'send_voice_text' via SocketIO:", payload);
        socket.emit('send_voice_text', payload);
    }

    function displayMessage(text, sender) {
        if (!text || !interactionActive) return;
        const chatArea = (sender === 'user') ? userChatArea : agentChatArea;
        const messageClass = (sender === 'user') ? 'user-message' : 'agent-message';
        const placeholder = chatArea.querySelector('.placeholder-text');
        if (placeholder && placeholder.style.display !== 'none') { placeholder.style.display = 'none'; }

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble', messageClass);
        messageBubble.textContent = text;
        chatArea.appendChild(messageBubble);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function speakText(text, lang = 'en-US') {
         if (!SpeechSynthesis || !text || !interactionActive) { console.log("Skipping speech synthesis."); return; }
         if (SpeechSynthesis.speaking || SpeechSynthesis.pending) { SpeechSynthesis.cancel(); }

         const utterance = new SpeechSynthesisUtterance(text);
         utterance.lang = lang; // Set language
         // utterance.voice = findVoiceForLang(lang); // Optional: Find specific voice
         utterance.onerror = (event) => console.error("Speech synthesis error:", event.error);
         utterance.onend = () => console.log("Finished speaking response.");

         setTimeout(() => { if (interactionActive) SpeechSynthesis.speak(utterance); }, 100);
    }

    function clearPlaceholders() {
        const placeholders = document.querySelectorAll('.placeholder-text');
        placeholders.forEach(p => p.style.display = 'none');
    }
    function hideExitMessage() {
         if (statusMessage.textContent.includes("Interaction stopped")) { statusMessage.textContent = "Click mic to start"; }
    }
    function clearChatAreas() {
         userChatArea.innerHTML = '<p class="placeholder-text" style="display: block;">User chat cleared.</p>';
         agentChatArea.innerHTML = '<p class="placeholder-text" style="display: block;">Agent chat cleared.</p>';
    }

}); // End DOMContentLoaded