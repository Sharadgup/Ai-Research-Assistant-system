/**
 * Handles interactions for the Construction AI Assistant page.
 */
console.log("[Construction Agent JS] Script loaded.");

document.addEventListener('DOMContentLoaded', () => {
    console.log("[Construction Agent JS] DOM Content Loaded.");

    // --- Construction Agent Elements ---
    const contextInput = document.getElementById('constructionDataContext');
    const queryInput = document.getElementById('constructionQueryInput');
    const submitBtn = document.getElementById('submitConstructionQueryBtn');
    const agentOutput = document.getElementById('constructionAgentOutput');
    const loadingIndicator = document.getElementById('constructionAgentLoading');
    const errorDiv = document.getElementById('constructionAgentError');

    // --- Initial Check ---
    if (!contextInput || !queryInput || !submitBtn || !agentOutput) {
        console.error("[Construction Agent JS] Core UI elements not found. Functionality disabled.");
        if(loadingIndicator) loadingIndicator.style.display = 'none';
        if(errorDiv) errorDiv.style.display = 'none';
        return;
    }
    console.log("[Construction Agent JS] UI elements found.");

    // --- Event Listeners ---
    submitBtn.addEventListener('click', handleConstructionQuery);
    // Optional: Allow sending via Enter in query input (if desired)
    // queryInput.addEventListener('keypress', (event) => {
    //     if (event.key === 'Enter' && !event.shiftKey && !submitBtn.disabled) {
    //         event.preventDefault();
    //         handleConstructionQuery();
    //     }
    // });
    console.log("[Construction Agent JS] Event listeners attached.");

    // --- Construction Agent Query Handler ---
    async function handleConstructionQuery() {
        const query = queryInput.value.trim();
        const context = contextInput.value.trim(); // Get context as well

        if (!query) {
            showAgentError("Please enter a query or task request.");
            return;
        }
        // Context is optional, so no check needed here unless required by backend

        console.log("[Construction Agent JS] Sending query:", query, "with context length:", context.length);
        setAgentLoading(true);
        hideAgentError();
        agentOutput.innerHTML = '<p><i>Analyzing data and generating insights...</i></p>';

        try {
            const response = await fetch('/construction_agent_query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, context: context }) // Send both
            });
            console.log(`[Construction Agent JS] Rcvd status: ${response.status}`);

            if (!response.ok) {
                let errorMsg = `Error: ${response.status}`;
                try { const errData = await response.json(); errorMsg = errData.error || errorMsg; } catch(e) {}
                if(response.status === 401) { errorMsg = "Authentication required."; }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            console.log("[Construction Agent JS] Rcvd data:", data);

            if (data.error) { throw new Error(data.error); }

            if (data.answer) {
                // Use innerHTML directly as the response might contain markdown/formatting
                agentOutput.innerHTML = data.answer; // Assume server sends formatted HTML or Markdown processed text
                // If server sends plain text:
                // const sanitizedAnswer = data.answer.replace(/</g, "<").replace(/>/g, ">");
                // agentOutput.innerHTML = `<p>${sanitizedAnswer.replace(/\n/g, '<br>')}</p>`;
            } else { throw new Error("Received an empty answer."); }

        } catch (error) {
            console.error("[Construction Agent JS] Fetch/Process Error:", error);
            showAgentError(`Failed: ${error.message}`);
            agentOutput.innerHTML = `<p><i>Sorry, an error occurred while processing the request.</i></p>`;
        } finally {
            setAgentLoading(false);
        }
    }
    // ------------------------------------------

    // --- Construction Agent Helper Functions ---
     function setAgentLoading(isLoading) {
         if (loadingIndicator) { loadingIndicator.style.display = isLoading ? 'flex' : 'none'; }
         if (submitBtn) { submitBtn.disabled = isLoading; }
         if (queryInput) { queryInput.disabled = isLoading; } // Optionally disable inputs
         if (contextInput) { contextInput.disabled = isLoading;}
     }
     function showAgentError(message) {
         console.error("[Construction Agent JS] Error:", message);
         if (errorDiv) { errorDiv.textContent = message; errorDiv.style.display = 'block'; }
         else { alert("Construction Agent Error: " + message); }
     }
     function hideAgentError() { if (errorDiv) { errorDiv.style.display = 'none'; errorDiv.textContent = ''; } }
     // ---------------------------------------------

     // --- Initial UI State for Construction Agent ---
     function initializeAgentUI() {
         console.log("[Construction Agent JS] Initializing UI state...");
         hideAgentError();
         setAgentLoading(false); // Ensure loading is off
         if (agentOutput) agentOutput.innerHTML = "<p><i>The AI's insights or suggested tasks will appear here.</i></p>";
         if (queryInput) queryInput.value = '';
         if (contextInput) contextInput.value = '';
     }
     initializeAgentUI();

     function scrollToBottom(element) {
        if(element) {
            // element.scrollTop = element.scrollHeight; // Instant scroll
            element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' }); // Smooth scroll
        }
    }

}); // End DOMContentLoaded