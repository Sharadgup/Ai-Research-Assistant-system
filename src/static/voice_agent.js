/**
 * Frontend Script for Voice AI Assistant Page
 */
console.log("[Voice Agent JS] Script loaded.");

document.addEventListener('DOMContentLoaded', () => {
    console.log("[Voice Agent JS] DOM Content Loaded.");

    // --- Check for Browser Support ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const SpeechSynthesis = window.speechSynthesis;

    if (!SpeechRecognition) {
        showVoiceError("Speech Recognition API is not supported in this browser. Please try Chrome or Edge.");
        console.error("Speech Recognition not supported.");
        // Disable buttons if API is missing
        const startBtn = document.getElementById('startVoiceBtn');
        if(startBtn) startBtn.disabled = true;
        return; // Stop script execution
    }
     if (!SpeechSynthesis) {
        showVoiceError("Speech Synthesis API is not supported in this browser.");
        console.error("Speech Synthesis not supported.");
        // Allow recognition, but voice output won't work
    }

    // --- Elements ---
    const startBtn = document.getElementById('startVoiceBtn');
    const stopBtn = document.getElementById('stopVoiceBtn');
    const statusDiv = document.getElementById('voiceStatus');
    const errorDiv = document.getElementById('voiceError');
    const messagesDiv = document.getElementById('voiceChatMessages');
    // const typingIndicator = document.getElementById('voiceTypingIndicator'); // Optional

    // --- State ---
    let recognition;
    let isListening = false;
    let isSpeaking = false;
    let stopRequested = false;
    let finalTranscript = '';
    let currentUtterance = null; // To track ongoing speech
    let voiceSocket; // Socket for this namespace

    // --- Setup Recognition Instance ---
    function setupRecognition() {
        if (recognition) { // Stop previous instance if exists
            try { recognition.stop(); } catch(e){}
        }
        recognition = new SpeechRecognition();
        recognition.continuous = false; // Process after user stops speaking
        recognition.interimResults = true; // Show results as they come in (optional)
        recognition.lang = 'en-US'; // Default language - can be changed

        recognition.onstart = () => {
            console.log("Recognition started");
            isListening = true;
            stopRequested = false;
            updateStatus("LISTENING...", "listening");
            if(startBtn) startBtn.style.display = 'none';
            if(stopBtn) stopBtn.style.display = 'inline-block';
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            finalTranscript = ''; // Reset final transcript for this recognition cycle
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            // Display interim results if desired (optional)
            // console.log("Interim:", interimTranscript);
            // if (interimTranscript) updateStatus(`Listening... (Heard: ${interimTranscript})`, "listening");

            if (finalTranscript) {
                console.log("Final transcript received:", finalTranscript);
                // Don't stop listening here, let onend handle it
            }
        };

        recognition.onerror = (event) => {
            console.error("Recognition Error:", event.error);
            let errorMsg = `Speech recognition error: ${event.error}`;
            if (event.error === 'no-speech') {
                errorMsg = "No speech detected. Please try again.";
            } else if (event.error === 'audio-capture') {
                errorMsg = "Microphone error. Ensure it's connected and allowed.";
            } else if (event.error === 'not-allowed') {
                errorMsg = "Permission to use microphone denied. Please allow access.";
            } else if (event.error === 'network') {
                errorMsg = "Network error during speech recognition.";
            }
             else if (event.error === 'aborted') {
                 console.log("Recognition aborted, likely intentional."); // Often happens when stop() is called
                 errorMsg = null; // Don't show error for manual stop
             }

            if (errorMsg) showVoiceError(errorMsg);
            isListening = false;
            updateStatus("ERROR", "error");
            if(startBtn) startBtn.style.display = 'inline-block';
            if(stopBtn) stopBtn.style.display = 'none';
        };

        recognition.onend = () => {
            console.log("Recognition ended.");
            isListening = false;
            if(startBtn) startBtn.style.display = 'inline-block';
            if(stopBtn) stopBtn.style.display = 'none';

            if (finalTranscript && !stopRequested) {
                console.log("Processing final transcript:", finalTranscript);
                updateStatus("PROCESSING...", "processing");
                appendVoiceMessage('User', finalTranscript); // Display user message
                sendTranscriptToServer(finalTranscript, recognition.lang); // Send to backend
            } else if (stopRequested) {
                console.log("Stopped manually, not processing.");
                updateStatus("IDLE", "idle"); // Return to idle
            } else {
                 console.log("Recognition ended without final transcript (e.g., no speech).");
                 // No error shown here if 'no-speech' handled in onerror
                 if (statusDiv && !statusDiv.classList.contains('error')) {
                     updateStatus("IDLE", "idle");
                 }
            }
            finalTranscript = ''; // Clear for next time
        };
    }


    // --- Initialize Socket.IO (/voice_chat) ---
    try {
        if (typeof io === 'undefined') throw new Error("Socket.IO client library not found.");
        voiceSocket = io('/voice_chat'); // Connect to voice namespace
        setupVoiceSocketListeners();
        updateStatus("CONNECTING...", "idle");
    } catch (e) {
        console.error("Voice Socket.IO Init Error:", e);
        showVoiceError("Could not connect to voice chat service.");
        updateStatus("ERROR", "error");
        if(startBtn) startBtn.disabled = true;
    }

    // --- SocketIO Listeners (/voice_chat) ---
    function setupVoiceSocketListeners() {
        if (!voiceSocket) return;

        voiceSocket.on('connect', () => {
            console.log('VOICE CHAT: Connected.', voiceSocket.id);
            hideVoiceError();
            updateStatus("IDLE", "idle"); // Ready state
            if(startBtn) startBtn.disabled = false;
            appendVoiceMessage('System', 'Voice Assistant Ready.');
        });
        voiceSocket.on('disconnect', (reason) => {
            console.log(`VOICE CHAT: Disconnected: ${reason}`);
            showVoiceError("Voice chat disconnected.");
            updateStatus("DISCONNECTED", "error");
            if(startBtn) startBtn.disabled = true; // Disable on disconnect
             if(stopBtn) stopBtn.style.display = 'none';
             isListening = false; stopRecognition(); // Ensure recognition stops
        });
        voiceSocket.on('connect_error', (err) => {
            console.error(`VOICE CHAT: Conn error: ${err.message}`);
            showVoiceError(`Voice chat connection failed: ${err.message}`);
            updateStatus("ERROR", "error");
            if(startBtn) startBtn.disabled = true;
        });
        voiceSocket.on('receive_ai_voice_text', (data) => { // Listen for AI response
            console.log('VOICE CHAT: AI Response received:', data);
            if (data && data.user === 'AI' && typeof data.text === 'string') {
                appendVoiceMessage(data.user, data.text);
                speakText(data.text, data.lang); // Speak the response
            } else {
                appendVoiceMessage('System', '[Invalid AI response format]', true);
                restartListeningIfNeeded(); // Still try to restart listening
            }
        });
         voiceSocket.on('error', (data) => { // Errors from backend emit
             console.error('VOICE CHAT: Server Error:', data.message);
             showVoiceError(`Server error: ${data.message || 'Unknown'}`);
             updateStatus("ERROR", "error");
             restartListeningIfNeeded(); // Try to restart listening even after server error
         });
         // Optional: Handle typing indicator if server sends it
         // voiceSocket.on('typing_indicator', (data) => { ... });
    }

    // --- Speech Synthesis ---
    function speakText(text, lang = 'en-US') {
        if (!SpeechSynthesis) {
            console.warn("Speech Synthesis not supported, cannot speak.");
            restartListeningIfNeeded(); // Restart listening even if can't speak
            return;
        }
        if (isSpeaking && currentUtterance) {
             console.log("Cancelling previous speech.");
             SpeechSynthesis.cancel(); // Stop any ongoing speech
        }

        const utterance = new SpeechSynthesisUtterance(text);
        // Optionally find and set a specific voice
        // const voices = SpeechSynthesis.getVoices();
        // utterance.voice = voices.find(v => v.lang === lang) || voices.find(v => v.lang.startsWith(lang.split('-')[0])) || voices.find(v => v.default);
        utterance.lang = lang; // Set language for the utterance
        utterance.rate = 1.0; // Adjust rate as needed
        utterance.pitch = 1.0; // Adjust pitch as needed

        utterance.onstart = () => {
            console.log("SpeechSynthesis started.");
            isSpeaking = true;
            updateStatus("SPEAKING...", "speaking");
             stopRecognition(); // Make sure recognition is stopped while speaking
        };
        utterance.onend = () => {
            console.log("SpeechSynthesis finished.");
            isSpeaking = false;
            currentUtterance = null;
            // Restart listening ONLY if stop wasn't requested
            restartListeningIfNeeded();
        };
        utterance.onerror = (event) => {
            console.error("SpeechSynthesis Error:", event.error);
            showVoiceError(`Speech synthesis error: ${event.error}`);
            isSpeaking = false;
            currentUtterance = null;
             // Restart listening even if speech failed
            restartListeningIfNeeded();
        };

        currentUtterance = utterance; // Track current speech
        SpeechSynthesis.speak(utterance);
    }

    // --- Control Functions ---
    function startListening() {
        if (isListening) {
            console.log("Already listening.");
            return;
        }
         if (isSpeaking) {
            console.log("Cannot start listening while speaking.");
            // Maybe queue it up? For now, just ignore.
            return;
        }
        hideVoiceError();
        setupRecognition(); // Re-create instance to clear previous state
        try {
            console.log("Calling recognition.start()");
            recognition.start();
        } catch (e) {
            console.error("Error starting recognition:", e);
            showVoiceError("Could not start microphone. Check permissions.");
            updateStatus("ERROR", "error");
        }
    }

    function stopRecognition() {
        if (recognition && isListening) {
            console.log("Calling recognition.stop()");
            stopRequested = true; // Set flag to prevent processing transcript on end
            recognition.stop(); // This will trigger the 'onend' event
            isListening = false;
        } else {
            console.log("Recognition not active, no need to stop.");
        }
        // Ensure UI reflects stopped state immediately
        if (startBtn) startBtn.style.display = 'inline-block';
        if (stopBtn) stopBtn.style.display = 'none';
        if (!isSpeaking) updateStatus("IDLE", "idle"); // Go to idle if not speaking
    }

    function sendTranscriptToServer(transcript, lang) {
        if (voiceSocket && voiceSocket.connected) {
            console.log(`Sending transcript to server (lang: ${lang}):`, transcript);
            try {
                voiceSocket.emit('send_voice_text', { text: transcript, lang: lang });
            } catch(e) {
                console.error("Error emitting voice text:", e);
                showVoiceError("Failed to send voice data to server.");
                updateStatus("ERROR", "error");
                restartListeningIfNeeded(); // Still try to restart
            }
        } else {
            console.error("Cannot send transcript: Voice socket not connected.");
            showVoiceError("Not connected to voice chat server.");
             updateStatus("DISCONNECTED", "error");
        }
    }

    function restartListeningIfNeeded() {
        // Only restart if the user hasn't manually stopped
        if (!stopRequested && !isListening && !isSpeaking) {
             console.log("Speech ended, restarting recognition loop...");
             // Add a small delay before restarting
             setTimeout(() => {
                startListening();
             }, 250); // 250ms delay
        } else if (stopRequested) {
             console.log("Stop requested, not restarting listening.");
             updateStatus("IDLE", "idle");
        } else {
             console.log("Not restarting listening (isListening or isSpeaking is true, or stop requested).");
        }
    }

    // --- UI Helpers ---
    function updateStatus(text, statusClass) {
        if (!statusDiv) return;
        statusDiv.textContent = `Status: ${text}`;
        // Remove previous status classes
        statusDiv.classList.remove('idle', 'listening', 'processing', 'speaking', 'error', 'disconnected');
        // Add the new class
        if (statusClass) {
            statusDiv.classList.add(statusClass);
        }
    }

    function appendVoiceMessage(user, text, isSystem = false) {
        // Similar to dashboard/report chat append, uses 'voiceChatMessages' div
         if (!messagesDiv) return;
         const el = document.createElement('div'); el.classList.add('message');
         if(isSystem){el.classList.add('system'); el.innerHTML=text;}
         else {el.classList.add(user.toLowerCase()==='ai'?'ai':'user'); const sText=text.replace(/</g,"<").replace(/>/g,">"); el.textContent=sText;}
         messagesDiv.appendChild(el); scrollToBottom(messagesDiv);
     }

    function showVoiceError(message) { if(errorDiv) { errorDiv.textContent = message; errorDiv.style.display = 'block';} else { console.error("Voice Error Div not found:", message); } }
    function hideVoiceError() { if(errorDiv) { errorDiv.style.display = 'none'; errorDiv.textContent = ''; } }
    function scrollToBottom(element) { if(element) { element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' }); } } // Use smooth scroll


    // --- Initial Setup ---
    function initializeVoiceUI() {
        console.log("[Voice Agent JS] Initializing UI...");
        hideVoiceError();
        if(startBtn) startBtn.disabled = true; // Wait for socket connect
        if(stopBtn) stopBtn.style.display = 'none';
        if(messagesDiv) messagesDiv.innerHTML = '<div class="message system">Initializing connection...</div>'; // Clear log
        // Status set by socket connect listener
    }

    // --- Attach Button Listeners ---
    if (startBtn) {
        startBtn.addEventListener('click', startListening);
    } else { console.error("Start Listening button not found!"); }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRecognition);
    } else { console.error("Stop Listening button not found!"); }

    // Initial UI state setup
    initializeVoiceUI();

}); // End DOMContentLoaded