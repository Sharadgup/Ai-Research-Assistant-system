/**
 * Handles interactions for the Education AI Assistant panel.
 * Assumes the necessary HTML elements exist on the page where this is loaded.
 */
console.log("[Education Agent] Script loaded.");

document.addEventListener('DOMContentLoaded', () => {
    console.log("[Education Agent] DOM Content Loaded.");

    // --- Education Agent Elements ---
    const educationQueryInput = document.getElementById('educationQueryInput');
    const submitEducationQueryBtn = document.getElementById('submitEducationQueryBtn');
    const educationAgentOutput = document.getElementById('educationAgentOutput');
    const educationAgentLoading = document.getElementById('educationAgentLoading');
    const educationAgentError = document.getElementById('educationAgentError'); // Specific error div for this agent

    // --- Initial Check ---
    // Only proceed if the core elements for this agent are found
    if (!educationQueryInput || !submitEducationQueryBtn || !educationAgentOutput) {
        console.warn("[Education Agent] Core UI elements (input, button, or output) not found. Agent functionality disabled for this page.");
        // Attempt to hide related elements if they exist partially
        if(educationAgentLoading) educationAgentLoading.style.display = 'none';
        if(educationAgentError) educationAgentError.style.display = 'none';
        return; // Stop execution for this script if elements are missing
    }
    console.log("[Education Agent] UI elements found.");

    // --- Event Listeners ---
    submitEducationQueryBtn.addEventListener('click', handleEducationQuery);
    educationQueryInput.addEventListener('keypress', (event) => {
         // Allow Shift+Enter for newlines in textarea, send on Enter alone
        if (event.key === 'Enter' && !event.shiftKey && !submitEducationQueryBtn.disabled) {
            event.preventDefault(); // Prevent default newline in textarea on plain Enter
            handleEducationQuery();
        }
    });
    console.log("[Education Agent] Event listeners attached.");

    // --- Education Agent Query Handler ---
    async function handleEducationQuery() {
        const query = educationQueryInput.value.trim();
        if (!query) {
            showEducationError("Please enter an education-related question.");
            return;
        }

        console.log("[Education Agent] Sending query:", query);
        setEducationLoading(true); // Show loading specific to this agent
        hideEducationError();      // Hide previous errors for this agent
        educationAgentOutput.innerHTML = '<p><i>Fetching answer...</i></p>'; // Indicate processing

        try {
            // Fetch request to the dedicated Flask endpoint
            const response = await fetch('/education_agent_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Include credentials if your endpoint requires session cookies
                    // 'credentials': 'include'
                },
                body: JSON.stringify({ query: query })
            });
            console.log(`[Education Agent] Rcvd status: ${response.status}`);

            // Check for HTTP errors (like 401 Unauthorized, 500 Internal Server Error)
            if (!response.ok) {
                let errorMsg = `Error: ${response.status}`;
                try {
                    const errData = await response.json();
                    errorMsg = errData.error || errorMsg; // Prefer specific error from JSON
                } catch (e) {
                    // If response wasn't JSON, use status text
                    errorMsg = `Error ${response.status}: ${response.statusText}`;
                }
                 // Handle specific common errors
                 if(response.status === 401) { errorMsg = "Authentication required. Please log in."; }
                 else if(response.status === 503) { errorMsg = "AI model service is temporarily unavailable."; }

                throw new Error(errorMsg);
            }

            // Parse the successful JSON response
            const data = await response.json();
            console.log("[Education Agent] Rcvd data:", data);

            // Check for application-level errors within the JSON response
            if (data.error) {
                 throw new Error(data.error);
            }

            // Display the answer
            if (data.answer) {
                // Basic sanitization (replace < >) - Use a library for production!
                const sanitizedAnswer = data.answer
                                        .replace(/</g, "<")
                                        .replace(/>/g, ">");
                // Replace newline characters with <br> for HTML display
                educationAgentOutput.innerHTML = `<p>${sanitizedAnswer.replace(/\n/g, '<br>')}</p>`;
            } else {
                 // Handle case where server responded 200 OK but no answer was provided
                 throw new Error("Received an empty answer from the agent.");
            }

        } catch (error) {
            console.error("[Education Agent] Fetch/Process Error:", error);
            showEducationError(`Failed to get answer: ${error.message}`);
            educationAgentOutput.innerHTML = `<p><i>Sorry, an error occurred while fetching the answer.</i></p>`;
        } finally {
            setEducationLoading(false); // Hide loading indicator regardless of success/failure
        }
    }
    // ------------------------------------------

    // --- Education Agent Helper Functions ---
     function setEducationLoading(isLoading) {
         if (educationAgentLoading) {
             educationAgentLoading.style.display = isLoading ? 'flex' : 'none';
         } else { console.warn("Education loading indicator not found."); }
         if (submitEducationQueryBtn) {
             submitEducationQueryBtn.disabled = isLoading;
         } else { console.warn("Education submit button not found."); }
          if (educationQueryInput) { // Optionally disable input while loading
             educationQueryInput.disabled = isLoading;
         }
     }

     function showEducationError(message) {
         console.error("[Education Agent] Error:", message);
         if (educationAgentError) {
             educationAgentError.textContent = message;
             educationAgentError.style.display = 'block';
         } else {
             // Fallback if specific error div isn't found (shouldn't happen with initial check)
             alert("Education Agent Error: " + message);
         }
     }

     function hideEducationError() {
          if (educationAgentError) {
             educationAgentError.style.display = 'none';
             educationAgentError.textContent = '';
         }
     }
     // ---------------------------------------------

     // --- Initial UI State for Education Agent ---
     function initializeEducationAgentUI() {
         console.log("[Education Agent] Initializing UI state...");
         hideEducationError();
         setEducationLoading(false); // Ensure loading is off initially
         if (educationAgentOutput) educationAgentOutput.innerHTML = "<p><i>The AI's answer will appear here.</i></p>";
         if (educationQueryInput) educationQueryInput.value = ''; // Clear input
     }
     initializeEducationAgentUI(); // Run initial setup

}); // End DOMContentLoaded