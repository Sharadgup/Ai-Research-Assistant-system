/**
 * AI Note Taker - Frontend Script (DEBUGGING VERSION)
 * Temporarily comments out chat UI activation after report generation
 * to isolate the cause of disconnects.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Element References (Report, Charts, PDF, Chat) ---
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
    let currentReportContext = ""; // Still store context, just don't use it immediately

    // --- Initialize Socket.IO ---
    let socket;
    try {
        console.log("Attempting to initialize Socket.IO...");
        socket = io({ /* options */ });
        console.log("Socket.IO object created, setting up listeners...");
        setupSocketListeners();
        disableChatInput(true, "Connecting to chat...");
    } catch (e) {
        console.error("CRITICAL: Failed to initialize Socket.IO:", e);
        showError("Could not connect to chat server. Please refresh.", true);
        disableChatInput(true, "Chat unavailable.");
    }

    // --- SocketIO Event Listeners Setup ---
    function setupSocketListeners() {
        if (!socket) { console.error("setupSocketListeners called but socket is not initialized."); return; }
        console.log("Setting up Socket.IO event listeners...");

        socket.on('connect', () => {
            console.log('CLIENT LOG: Socket.IO Connected event received. SID:', socket.id);
            hideError();
            // Only enable if chat isn't supposed to be hidden (e.g., after report)
            // For this test, we'll let handleGenerateReport control enabling later
             // disableChatInput(false); // Don't enable here for this test
            console.log('CLIENT LOG: Connect event - Chat input state unchanged for now.');
        });

        socket.on('disconnect', (reason) => {
            console.log(`CLIENT LOG: Socket.IO Disconnected event received. Reason: ${reason}`);
            if (reason !== 'io client disconnect') {
                 showError("Chat connection lost. Trying to reconnect...", false);
                 if(chatContainer) chatContainer.style.display = 'none';
                 console.log('CLIENT LOG: Chat input disabled on disconnect.');
                 disableChatInput(true, "Chat disconnected.");
            } else {
                 console.log("CLIENT LOG: Disconnected by client action. Disabling chat input.");
                 disableChatInput(true, "Chat disconnected.");
            }
        });

        socket.on('connect_error', (err) => {
            console.error(`CLIENT LOG: Socket.IO Connection error event received: ${err.message}`);
            showError(`Chat connection failed: ${err.message}. Please check server and refresh.`, true);
            if(chatContainer) chatContainer.style.display = 'none';
            console.log('CLIENT LOG: Chat input disabled on connection error.');
            disableChatInput(true, "Chat connection failed.");
        });

        // --- Other existing listeners ---
        socket.on('receive_message', (data) => { /* ... Keep implementation ... */
            console.log('Message received:', data);
            if (data && data.user && typeof data.text === 'string') { appendMessage(data.user, data.text); }
            else { console.warn("Received malformed message data:", data); appendMessage('System', '[Received incomplete message from server]', true); }
        });
        socket.on('error', (data) => { /* ... Keep implementation ... */
            console.error('WebSocket Server Error Event:', data.message);
            appendMessage('System', `Server Error: ${data.message || 'Unknown error'}`, true);
        });
        socket.on('typing_indicator', (data) => { /* ... Keep implementation ... */
            if (typingIndicator) { typingIndicator.style.display = data.isTyping ? 'flex' : 'none'; if (data.isTyping) { scrollToBottom(chatMessages); } }
        });
    } // --- End setupSocketListeners ---


    // --- Event Listeners ---
    if (generateBtn) { generateBtn.addEventListener('click', handleGenerateReport); }
    else { console.warn("Generate button not found."); }

    async function handleGenerateReport() {
        const text = inputText.value.trim();
        if (!text) { showError("Please enter some text.", true); return; }
        setLoadingState(true, "Generating report...");
        hideError();
        if(reportContainer) reportContainer.style.display = 'none';
        if(chatContainer) chatContainer.style.display = 'none'; // Keep chat hidden initially
        clearChatMessages();
        destroyCharts();
        let response;
        try {
            console.log("Sending request to /generate_report");
            response = await fetch('/generate_report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) });
            console.log(`Received response status: ${response.status}`);
            if (!response.ok) { /* ... Keep error handling ... */
                let errorMsg = `Server error! Status: ${response.status} ${response.statusText}`;
                try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; console.error("Server returned error JSON:", errorData); }
                catch (jsonError) { try { const errorText = await response.text(); errorMsg += `\nResponse: ${errorText.substring(0, 500)}`; console.error("Server returned non-JSON error text:", errorText); } catch(textError) { console.error("Could not read error response body:", textError); } }
                throw new Error(errorMsg);
            }
            const data = await response.json();
            console.log("Received data:", data);
            if (!data || typeof data.report_html !== 'string') { console.error("Invalid data received from server:", data); throw new Error("Received invalid or incomplete report data from the server."); }

            // *** Wrap UI updates in try...catch ***
            try {
                 // Display report and charts as usual
                 if (reportOutput) { reportOutput.innerHTML = data.report_html; } else { console.warn("reportOutput element not found."); }
                 processChartData(data.chart_data);
                 if(reportContainer) reportContainer.style.display = 'block';

                // Store context but DON'T activate chat UI yet
                currentReportContext = data.report_context_for_chat || "";

                // --- Re-enable Chat UI Activation ---
                console.log("CLIENT LOG: Report generated successfully. Preparing chat UI.");
                if(chatContainer) chatContainer.style.display = 'flex'; // Show chat
                // Check connection status *again* before enabling
                if (socket && socket.connected) {
                    console.log("CLIENT LOG: Socket is connected after report generation. Enabling chat input.");
                    disableChatInput(false);
                } else {
                    console.warn("CLIENT LOG: Socket is NOT connected right after report generation! Keeping chat input disabled.");
                    showError("Chat disconnected after report loaded.", false);
                    disableChatInput(true, "Chat disconnected.");
                }
                appendMessage('System', 'Report generated. Ask me anything about it!', true);

                if (socket && socket.connected) {
                    console.log("CLIENT LOG : Socket IS connected immediately after report display.");
                 }


             } catch (uiError) { /* ... Keep UI error handling ... */
                 console.error("Error updating UI after receiving data:", uiError);
                 showError(`Error displaying report: ${uiError.message}`, true);
                 if(reportContainer) reportContainer.style.display = 'none';
                 if(chatContainer) chatContainer.style.display = 'none';
             }
        } catch (error) { /* ... Keep fetch/server error handling ... */
            console.error("Error during handleGenerateReport:", error);
            showError(`Failed to generate report: ${error.message}`, true);
             if(reportContainer) reportContainer.style.display = 'none';
             if(chatContainer) chatContainer.style.display = 'none';
        } finally {
             console.log("Finished handleGenerateReport processing.");
            setLoadingState(false);
        }
    } // --- End handleGenerateReport ---


    // Send Chat Message Button / Enter Key (Send will fail if input is disabled)
    if (sendChatBtn) { sendChatBtn.addEventListener('click', sendMessage); } else { console.warn("Send chat button not found."); }
    if (chatInput) { chatInput.addEventListener('keypress', (event) => { if (event.key === 'Enter' && !event.shiftKey && !chatInput.disabled) { event.preventDefault(); sendMessage(); } }); } else { console.warn("Chat input field not found."); }

    // PDF Download Button
    if (downloadPdfBtn) { downloadPdfBtn.addEventListener('click', handleDownloadPdf); } else { console.warn("Download PDF button not found."); }


    // --- Helper Functions ---

    function disableChatInput(disabled, placeholderText = "Ask a question about the report...") { /* ... Keep implementation ... */
        console.log(`Setting chat input disabled state: ${disabled}, placeholder: "${placeholderText}"`);
        if (chatInput) { chatInput.disabled = disabled; chatInput.placeholder = placeholderText; }
        else { console.warn("disableChatInput: chatInput element not found"); }
        if (sendChatBtn) { sendChatBtn.disabled = disabled; }
        else { console.warn("disableChatInput: sendChatBtn element not found"); }
    }
    function sendMessage() { /* ... Keep implementation ... */
        if (!socket || !socket.connected) { showError("Cannot send message: Not connected to chat server.", false); disableChatInput(true, "Chat disconnected."); return; }
        if (!chatInput || chatInput.disabled) { console.warn("sendMessage called but input is disabled or not found."); return; }
        const messageText = chatInput.value.trim(); if (!messageText) return; appendMessage('User', messageText); chatInput.value = ''; console.log("Sending message via WebSocket:", messageText); socket.emit('send_message', { text: messageText, report_context: currentReportContext });
    }
    function setLoadingState(isLoading, text = "Loading...") { /* ... Keep implementation ... */
        if (generateBtn) generateBtn.disabled = isLoading; if (loadingIndicator) { if(loadingText) loadingText.textContent = text; loadingIndicator.style.display = isLoading ? 'flex' : 'none'; }
    }
    function appendMessage(user, text, isSystem = false) { /* ... Keep implementation ... */
        if (!chatMessages) return; const messageElement = document.createElement('div'); messageElement.classList.add('message'); if (isSystem) { messageElement.classList.add('system'); messageElement.textContent = text; messageElement.style.fontStyle = 'italic'; } else { messageElement.classList.add(user.toLowerCase()); messageElement.textContent = text; } chatMessages.appendChild(messageElement); scrollToBottom(chatMessages);
    }
    function clearChatMessages() { if(chatMessages) chatMessages.innerHTML = ''; }
    function scrollToBottom(element) { if(element) { element.scrollTop = element.scrollHeight; } }
    function showError(message, isFatal = false) { /* ... Keep implementation ... */
        console.error("Displaying Error:", message); if(errorMessageDiv) { errorMessageDiv.textContent = message; errorMessageDiv.style.display = 'block'; errorMessageDiv.style.backgroundColor = isFatal ? '#dc3545' : '#ffc107'; errorMessageDiv.style.color = isFatal ? 'white' : 'black'; } else { alert(`Error: ${message}`); }
    }
    function hideError() { /* ... Keep implementation ... */
        if(errorMessageDiv) { errorMessageDiv.style.display = 'none'; errorMessageDiv.textContent = ''; }
    }
    function destroyCharts() { /* ... Keep implementation ... */
        if (keywordChartInstance) { keywordChartInstance.destroy(); keywordChartInstance = null; } if (sentimentChartInstance) { sentimentChartInstance.destroy(); sentimentChartInstance = null; } if(keywordChartContainer) keywordChartContainer.style.display = 'none'; if(sentimentChartContainer) sentimentChartContainer.style.display = 'none'; if(noChartsMessage) noChartsMessage.style.display = 'none';
    }
    function processChartData(chartData) { /* ... Keep implementation ... */
        destroyCharts(); let chartsGenerated = false;
        if (keywordChartContainer && keywordChartCanvas && chartData?.keyword_frequencies && Object.keys(chartData.keyword_frequencies).length > 0) { const keywords = Object.keys(chartData.keyword_frequencies); const counts = Object.values(chartData.keyword_frequencies); if (keywords.length > 0) { keywordChartContainer.style.display = 'block'; try { keywordChartInstance = new Chart(keywordChartCanvas, { type: 'bar', data: { labels: keywords, datasets: [{ label: 'Keyword Frequency', data: counts, backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1 }] }, options: { scales: { y: { beginAtZero: true } }, responsive: true, maintainAspectRatio: true } }); chartsGenerated = true; } catch (chartError) { console.error("Error creating keyword chart:", chartError); showError("Failed to display keyword chart.", false); keywordChartContainer.style.display = 'none'; } } }
        if (sentimentChartContainer && sentimentChartCanvas && chartData?.sentiment_score && Object.keys(chartData.sentiment_score).length > 0) { const sentimentLabels = Object.keys(chartData.sentiment_score); const sentimentValues = Object.values(chartData.sentiment_score); const filteredLabels = [], filteredValues = [], backgroundColors = []; const colorMap = { positive: 'rgba(75, 192, 192, 0.6)', negative: 'rgba(255, 99, 132, 0.6)', neutral: 'rgba(201, 203, 207, 0.6)' }; sentimentLabels.forEach((label, index) => { if (sentimentValues[index] > 0) { filteredLabels.push(label.charAt(0).toUpperCase() + label.slice(1)); filteredValues.push(sentimentValues[index]); backgroundColors.push(colorMap[label.toLowerCase()] || 'rgba(153, 102, 255, 0.6)'); } }); if (filteredLabels.length > 0) { sentimentChartContainer.style.display = 'block'; try { sentimentChartInstance = new Chart(sentimentChartCanvas, { type: 'doughnut', data: { labels: filteredLabels, datasets: [{ label: 'Sentiment Analysis', data: filteredValues, backgroundColor: backgroundColors, hoverOffset: 4 }] }, options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: function(c){ let l=c.label||''; let v=c.raw||0; let s=c.chart.data.datasets[0].data.reduce((a,b)=>a+b,0); let p=s>0?((v/s)*100).toFixed(1)+'%':'0%'; return `${l}: ${p} (${v})`; } } } } } }); chartsGenerated = true; } catch (chartError) { console.error("Error creating sentiment chart:", chartError); showError("Failed to display sentiment chart.", false); sentimentChartContainer.style.display = 'none'; } } }
        if(noChartsMessage) { noChartsMessage.style.display = chartsGenerated ? 'none' : 'block'; }
    }
    function handleDownloadPdf() { /* ... Keep implementation ... */
        const reportElement = document.getElementById('reportContainer'); if (!reportElement || reportElement.style.display === 'none') { showError("Cannot download PDF: No report is currently displayed.", false); return; } const reportTitle = inputText.value.substring(0, 30).replace(/[^a-z0-9]/gi, '_') || "ai_report"; const options = { margin: [0.5, 0.5, 0.5, 0.5], filename: `${reportTitle}_report.pdf`, image: { type: 'jpeg', quality: 0.98 }, html2canvas: { scale: 2, logging: false, useCORS: true,scrollY: -window.scrollY }, jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' } }; if(downloadPdfBtn) downloadPdfBtn.style.visibility = 'hidden'; setLoadingState(true, "Generating PDF..."); html2pdf().from(reportElement).set(options).save().then(() => { console.log("PDF generated successfully."); }).catch(err => { console.error("Error generating PDF:", err); showError(`Failed to generate PDF: ${err.message}`, false); }).finally(() => { if(downloadPdfBtn) downloadPdfBtn.style.visibility = 'visible'; setLoadingState(false); });
    }


    // --- Initial UI State ---
    function initializeUI() { /* ... Keep implementation ... */
        console.log("Initializing UI state..."); destroyCharts(); if(reportContainer) reportContainer.style.display = 'none'; if(chatContainer) chatContainer.style.display = 'none'; if(loadingIndicator) loadingIndicator.style.display = 'none'; hideError(); disableChatInput(true, "Initializing chat..."); if (!keywordChartCanvas || !sentimentChartCanvas) { console.warn("One or both chart canvas elements not found on initialization."); }
    }

    initializeUI(); // Call initialization function

}); // End DOMContentLoaded