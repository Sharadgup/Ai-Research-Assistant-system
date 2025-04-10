/**
 * AI Note Taker - Frontend Script (with Documentation ID handling and PDF fix attempt)
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const generateBtn = document.getElementById('generateBtn');
    const inputText = document.getElementById('inputText');
    const reportContainer = document.getElementById('reportContainer');
    const reportOutput = document.getElementById('reportOutput');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const loadingText = document.getElementById('loadingText');
    const errorMessageDiv = document.getElementById('errorMessage');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const keywordChartContainer = document.getElementById('keywordChartContainer');
    const sentimentChartContainer = document.getElementById('sentimentChartContainer');
    const keywordChartCanvas = document.getElementById('keywordChart')?.getContext('2d');
    const sentimentChartCanvas = document.getElementById('sentimentChart')?.getContext('2d');
    const noChartsMessage = document.getElementById('noChartsMessage');
    const chatContainer = document.getElementById('chatContainer');
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');
    const typingIndicator = document.getElementById('typingIndicator');

    // --- State Variables ---
    let keywordChartInstance = null;
    let sentimentChartInstance = null;
    let currentDocumentationId = null; // Stores the ID for the current chat

    // --- Initialize Socket.IO ---
    let socket;
    try {
        console.log("Attempting to initialize Socket.IO...");
        socket = io({ /* options like reconnection settings can go here */ });
        console.log("Socket.IO object created, setting up listeners...");
        setupSocketListeners();
        disableChatInput(true, "Connecting to chat...");
    } catch (e) {
        console.error("CRITICAL: Failed to initialize Socket.IO:", e);
        showError("Could not initialize chat connection. Please refresh.", true);
        disableChatInput(true, "Chat unavailable.");
    }

    // --- SocketIO Event Listeners Setup ---
    function setupSocketListeners() {
        if (!socket) { console.error("setupSocketListeners called but socket is not initialized."); return; }
        console.log("Setting up Socket.IO event listeners...");

        socket.on('connect', () => {
            console.log('CLIENT LOG: Socket.IO Connected event received. SID:', socket.id);
            hideError();
            if (chatContainer && chatContainer.style.display !== 'none' && currentDocumentationId) {
                console.log('CLIENT LOG: Reconnected - Chat visible and has Doc ID, enabling input.');
                disableChatInput(false);
            } else if (chatContainer && chatContainer.style.display !== 'none' && !currentDocumentationId) {
                 console.warn('CLIENT LOG: Reconnected - Chat visible BUT Doc ID missing, keeping disabled.');
                 disableChatInput(true, "Generate report first or refresh.");
            } else {
                 console.log('CLIENT LOG: Connect event - Chat not visible yet, input remains disabled.');
            }
        });

        socket.on('disconnect', (reason) => {
            console.log(`CLIENT LOG: Socket.IO Disconnected event received. Reason: ${reason}`);
            if (reason !== 'io client disconnect') {
                 showError("Chat connection lost. Trying to reconnect...", false);
                 disableChatInput(true, "Chat disconnected. Reconnecting...");
            } else {
                 disableChatInput(true, "Chat disconnected.");
            }
        });

        socket.on('connect_error', (err) => {
            console.error(`CLIENT LOG: Socket.IO Connection error event received: ${err.message}`);
            showError(`Chat connection failed: ${err.message}. Server might be down.`, true);
            disableChatInput(true, "Chat connection failed.");
        });

        socket.on('receive_message', (data) => {
            console.log('Message received from server:', data);
            if (data && data.user && typeof data.text === 'string') { appendMessage(data.user, data.text); }
            else { console.warn("Received malformed message data:", data); appendMessage('System', '[Received incomplete message from server]', true); }
        });

        socket.on('error', (data) => {
            console.error('WebSocket Server Error Event:', data.message);
            appendMessage('System', `Server Error: ${data.message || 'Unknown error'}`, true);
        });

        socket.on('typing_indicator', (data) => {
            if (typingIndicator) { typingIndicator.style.display = data.isTyping ? 'flex' : 'none'; if (data.isTyping) { scrollToBottom(chatMessages); } }
        });
    } // --- End setupSocketListeners ---

    // --- Event Listeners ---
    if (generateBtn) { generateBtn.addEventListener('click', handleGenerateReport); }
    if (sendChatBtn) { sendChatBtn.addEventListener('click', sendMessage); }
    if (chatInput) { chatInput.addEventListener('keypress', (event) => { if (event.key === 'Enter' && !event.shiftKey && !chatInput.disabled) { event.preventDefault(); sendMessage(); } }); }
    if (downloadPdfBtn) { downloadPdfBtn.addEventListener('click', handleDownloadPdf); } // Attach PDF handler

    // --- Core Functions ---

    async function handleGenerateReport() {
        const text = inputText.value.trim();
        if (!text) { showError("Please enter some text.", true); return; }
        setLoadingState(true, "Generating report...");
        hideError();
        if(reportContainer) reportContainer.style.display = 'none';
        if(chatContainer) chatContainer.style.display = 'none';
        clearChatMessages();
        destroyCharts();
        currentDocumentationId = null; // Reset Doc ID
        disableChatInput(true, "Generating report...");

        let response;
        try {
            console.log("Sending request to /generate_report");
            response = await fetch('/generate_report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) });
            console.log(`Received response status: ${response.status}`);
            if (!response.ok) { let errorMsg = `Server error! Status: ${response.status} ${response.statusText}`; try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; } catch (e) {} throw new Error(errorMsg); }
            const data = await response.json();
            console.log("Received report data:", data);
            if (!data || typeof data.report_html !== 'string') { throw new Error("Received invalid or incomplete report data."); }

            // Wrap UI updates
            try {
                if (reportOutput) { reportOutput.innerHTML = data.report_html; }
                processChartData(data.chart_data);
                if (reportContainer) reportContainer.style.display = 'block';
                currentDocumentationId = data.documentation_id; // Store the ID
                console.log(`CLIENT LOG: Report generated. Received documentation_id: ${currentDocumentationId}`);

                if (currentDocumentationId) {
                    if (chatContainer) chatContainer.style.display = 'flex';
                    appendMessage('System', 'Report generated. Ask me anything about it!', true);
                    if (socket && socket.connected) { disableChatInput(false); }
                    else { console.warn("Socket disconnected after report!"); showError("Chat connection is down.", false); disableChatInput(true, "Chat unavailable."); }
                } else {
                    console.error("documentation_id missing from server response!");
                    if (chatContainer) chatContainer.style.display = 'none';
                    showError("Report generated, but failed to initialize chat (missing ID).", true);
                    disableChatInput(true, "Chat unavailable (error).");
                }
             } catch (uiError) { console.error("Error updating UI:", uiError); showError(`Error displaying report: ${uiError.message}`, true); currentDocumentationId = null; }

        } catch (error) {
            console.error("Error during handleGenerateReport:", error);
            showError(`Failed to generate report: ${error.message}`, true);
            currentDocumentationId = null;
            if(reportContainer) reportContainer.style.display = 'none';
            if(chatContainer) chatContainer.style.display = 'none';
        } finally {
            setLoadingState(false);
        }
    } // --- End handleGenerateReport ---

    function sendMessage() {
        if (!socket || !socket.connected) { showError("Cannot send: Not connected.", false); disableChatInput(true, "Chat disconnected."); return; }
        if (!chatInput || chatInput.disabled) { console.warn("Input disabled."); return; }
        if (!currentDocumentationId) { console.error("sendMessage Error: currentDocumentationId missing!"); showError("Cannot send: No active report.", false); return; }
        const messageText = chatInput.value.trim();
        if (!messageText) return;
        appendMessage('User', messageText);
        chatInput.value = '';
        console.log(`Sending 'send_message' with doc ID: ${currentDocumentationId}`);
        socket.emit('send_message', { text: messageText, documentation_id: currentDocumentationId });
    }

    // --- Helper Functions ---

    function disableChatInput(disabled, placeholderText = "Ask a question about the report...") {
        console.log(`Setting chat input disabled: ${disabled}, placeholder: "${placeholderText}"`);
        if (chatInput) { chatInput.disabled = disabled; chatInput.placeholder = placeholderText; }
        if (sendChatBtn) { sendChatBtn.disabled = disabled; }
    }

    function setLoadingState(isLoading, text = "Loading...") {
        if (generateBtn) generateBtn.disabled = isLoading;
        if (loadingIndicator) { if(loadingText) loadingText.textContent = text; loadingIndicator.style.display = isLoading ? 'flex' : 'none'; }
    }

    function appendMessage(user, text, isSystem = false) {
        if (!chatMessages) return; const messageElement = document.createElement('div'); messageElement.classList.add('message'); if (isSystem) { messageElement.classList.add('system'); messageElement.innerHTML = text; messageElement.style.fontStyle = 'italic'; } else { messageElement.classList.add(user.toLowerCase() === 'ai' ? 'ai' : 'user'); const sanitizedText = text.replace(/</g, "<").replace(/>/g, ">"); messageElement.textContent = sanitizedText; } chatMessages.appendChild(messageElement); scrollToBottom(chatMessages);
    }

    function clearChatMessages() { if(chatMessages) chatMessages.innerHTML = ''; }

    function scrollToBottom(element) { if(element) { element.scrollTop = element.scrollHeight; } }

    function showError(message, isFatal = false) {
         console.error("Displaying Error:", message); if(errorMessageDiv) { errorMessageDiv.textContent = message; errorMessageDiv.style.display = 'block'; errorMessageDiv.style.backgroundColor = isFatal ? '#dc3545' : '#ffc107'; errorMessageDiv.style.color = isFatal ? 'white' : 'black'; } else { alert(`Error: ${message}`); }
    }

    function hideError() { if(errorMessageDiv) { errorMessageDiv.style.display = 'none'; errorMessageDiv.textContent = ''; } }

    function destroyCharts() {
        if (keywordChartInstance) { keywordChartInstance.destroy(); keywordChartInstance = null; }
        if (sentimentChartInstance) { sentimentChartInstance.destroy(); sentimentChartInstance = null; }
        if(keywordChartContainer) keywordChartContainer.style.display = 'none';
        if(sentimentChartContainer) sentimentChartContainer.style.display = 'none';
        if(noChartsMessage) noChartsMessage.style.display = 'none';
     }

    function processChartData(chartData) {
        destroyCharts(); let chartsGenerated = false;
        try { if (keywordChartContainer && keywordChartCanvas && chartData?.keyword_frequencies && Object.keys(chartData.keyword_frequencies).length > 0) { const keywords = Object.keys(chartData.keyword_frequencies); const counts = Object.values(chartData.keyword_frequencies); if (keywords.length > 0) { keywordChartContainer.style.display = 'block'; keywordChartInstance = new Chart(keywordChartCanvas, { type: 'bar', data: { labels: keywords, datasets: [{ label: 'Keyword Frequency', data: counts, backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1 }] }, options: { indexAxis: 'y', scales: { x: { beginAtZero: true } }, responsive: true, maintainAspectRatio: false, plugins:{legend:{display:false}} } }); chartsGenerated = true; } } } catch (chartError) { console.error("Error creating keyword chart:", chartError); showError("Failed to display keyword chart.", false); if (keywordChartContainer) keywordChartContainer.style.display = 'none'; }
        try { if (sentimentChartContainer && sentimentChartCanvas && chartData?.sentiment_score && Object.keys(chartData.sentiment_score).length > 0) { const sentimentLabels = Object.keys(chartData.sentiment_score); const sentimentValues = Object.values(chartData.sentiment_score); const filteredLabels = [], filteredValues = [], backgroundColors = []; const colorMap = { positive: 'rgba(75, 192, 192, 0.6)', negative: 'rgba(255, 99, 132, 0.6)', neutral: 'rgba(201, 203, 207, 0.6)' }; let totalScore = 0; sentimentLabels.forEach((label, index) => { const value = sentimentValues[index] || 0; if (value > 0) { filteredLabels.push(label.charAt(0).toUpperCase() + label.slice(1)); filteredValues.push(value); backgroundColors.push(colorMap[label.toLowerCase()] || 'rgba(153, 102, 255, 0.6)'); totalScore += value; } }); if (filteredLabels.length > 0 && totalScore > 0) { sentimentChartContainer.style.display = 'block'; sentimentChartInstance = new Chart(sentimentChartCanvas, { type: 'doughnut', data: { labels: filteredLabels, datasets: [{ label: 'Sentiment Analysis', data: filteredValues, backgroundColor: backgroundColors, hoverOffset: 4 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: function(c){ let l=c.label||''; let v=c.raw||0; let s=c.chart.data.datasets[0].data.reduce((a,b)=>a+b,0); let p=s>0?((v/s)*100).toFixed(1)+'%':'0%'; return `${l}: ${p} (${v.toFixed(2)})`; } } } } } }); chartsGenerated = true; } } } catch (chartError) { console.error("Error creating sentiment chart:", chartError); showError("Failed to display sentiment chart.", false); if (sentimentChartContainer) sentimentChartContainer.style.display = 'none'; }
        if(noChartsMessage) { noChartsMessage.style.display = chartsGenerated ? 'none' : 'block'; }
    }

    // --- MODIFIED PDF Download Function ---
    function handleDownloadPdf() {
        const reportElement = document.getElementById('reportContainer');
        if (!reportElement || reportElement.style.display === 'none') {
            showError("Cannot download PDF: No report is currently displayed.", false); return;
        }
        const reportTitle = inputText.value.substring(0, 30).replace(/[^a-z0-9]/gi, '_').trim() || "ai_generated_report";

        // Temporarily hide button to avoid it appearing in PDF
        if(downloadPdfBtn) downloadPdfBtn.style.visibility = 'hidden';
        setLoadingState(true, "Generating PDF...");

        const options = {
            margin: [0.5, 0.5, 0.5, 0.5], // inches
            filename: `${reportTitle}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: {
                scale: 2,
                logging: false, // Set to true for more detailed html2canvas logs if needed
                useCORS: true,
                // scrollY: -window.scrollY, // Often needed, keep commented unless testing scroll issues
                // --- Added explicit width/height ---
                windowWidth: reportElement.scrollWidth,
                windowHeight: reportElement.scrollHeight
                // -----------------------------------
             },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };


        // Use html2pdf library
        console.log("Calling html2pdf with options:", options); // Log options before call
        html2pdf().from(reportElement).set(options).save()
            .then(() => {
                console.log("PDF generated successfully.");
            }).catch(err => {
                console.error("Error generating PDF:", err);
                showError(`Failed to generate PDF: ${err.message}`, false);
            }).finally(() => {
                // Always make button visible again and hide loading state
                 if(downloadPdfBtn) downloadPdfBtn.style.visibility = 'visible';
                 setLoadingState(false);
            });
    }
    // --- End MODIFIED PDF Download Function ---

    function initializeUI() {
        console.log("Initializing UI state...");
        destroyCharts();
        if (reportContainer) reportContainer.style.display = 'none';
        if (chatContainer) chatContainer.style.display = 'none';
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        hideError();
        disableChatInput(true, "Initializing chat connection...");
        if (!keywordChartCanvas || !sentimentChartCanvas) { console.warn("One or both chart canvas elements not found on initialization."); }
     }

    initializeUI(); // Call initialization function

}); // End DOMContentLoaded