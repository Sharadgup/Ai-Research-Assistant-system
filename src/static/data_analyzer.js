document.addEventListener('DOMContentLoaded', () => {
    console.log("DOMContentLoaded fired for:", window.location.pathname); // Debug: Check DOM ready

    // --- Element Selection ---
    const fileInput = document.getElementById('analysisFile');
    const uploadButton = document.getElementById('uploadButton');
    const feedbackDiv = document.getElementById('upload-feedback');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // --- Element Existence Check ---
    console.log('Element check - fileInput:', fileInput);
    console.log('Element check - uploadButton:', uploadButton);
    console.log('Element check - feedbackDiv:', feedbackDiv);
    console.log('Element check - loadingIndicator:', loadingIndicator);

    if (!fileInput || !uploadButton || !feedbackDiv || !loadingIndicator) {
        console.error("Data Analyzer Script Error: One or more required HTML elements are missing.");
        // ... (error handling) ...
        return; // Stop execution
    }
    console.log("Element check passed - All critical elements found.");

    // --- Initial State ---
    uploadButton.disabled = true;

    // --- Event Listeners ---
    console.log('Attaching listener to fileInput:', fileInput);
    fileInput.addEventListener('change', () => {
        // ... (file change logic as before) ...
         if (fileInput.files && fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const validationError = validateFile(file);
            if (validationError) {
                displayFeedback(validationError, true);
                uploadButton.disabled = true;
                fileInput.value = '';
            } else {
                uploadButton.disabled = false;
                clearFeedback();
            }
        } else {
            uploadButton.disabled = true;
            clearFeedback();
        }
    });

    console.log('Attaching listener to uploadButton:', uploadButton);
    uploadButton.addEventListener('click', handleUpload);

    // --- Core Upload Logic ---
    async function handleUpload() {
        console.log('handleUpload function EXECUTED!');
        // ... (checks for file selection, validation) ...
        if (!fileInput.files || fileInput.files.length === 0) {
             displayFeedback('Please select a file first.', true);
             uploadButton.disabled = true;
             return;
        }
        const file = fileInput.files[0];
        const validationError = validateFile(file);
        if (validationError) {
             displayFeedback(validationError, true);
             uploadButton.disabled = true;
             return;
        }


        const formData = new FormData();
        formData.append('analysisFile', file);

        showLoading(true);
        displayFeedback('Uploading and processing file...', false, false);
        console.log('Sending fetch request to /upload_analysis_data');

        try {
            const response = await fetch('/upload_analysis_data', {
                method: 'POST',
                body: formData,
            });
            console.log('Received fetch response. Status:', response.status);

            // Call the CORRECT processUploadResponse function
            await processUploadResponse(response, file.name);

        } catch (error) {
            console.error("Network or client-side upload error:", error);
            displayFeedback('A network error occurred. Please check your connection and try again.', true);
            showLoading(false);
        }
    }

    // --- Response Processing (Defined INSIDE DOMContentLoaded) ---
    async function processUploadResponse(response, originalFileName) {
        let result;
        const responseText = await response.text();
        console.log('Raw server response text:', responseText);

        try {
            result = JSON.parse(responseText);
            console.log('Parsed JSON response:', result);
        } catch (jsonError) {
            console.error("Failed to parse server response as JSON:", jsonError);
            const preview = responseText.substring(0, 200) + (responseText.length > 200 ? '...' : '');
            displayFeedback(`Server Error: Unexpected response format (Status: ${response.status}). Response started with: "${preview}". Check browser console.`, true);
            showLoading(false);
            return;
        }

        // --- THIS IS THE CORRECT LOCATION FOR THE REDIRECT LOGIC ---
        if (response.ok && result && result.upload_id) {
            console.log('Upload successful. Preparing redirect...'); // Log success path reached
            const redirectId = result.upload_id;

            // --- ADD THE DEBUG LOG *HERE* ---
            console.log(`Redirecting to: /data_cleaner/${redirectId} (Type: ${typeof redirectId})`);
            // --- END DEBUG LOG ---

            displayFeedback(`Success! File '${result.filename || originalFileName}' uploaded (${result.rows} rows, ${result.columns} columns). Redirecting...`, false, false);
            window.location.href = `/data_cleaner/${redirectId}`; // Perform redirect
        } else {
            // --- Error Handling ---
            const errorMessage = result?.error || `Upload failed (Status: ${response.status}).`;
            console.error("Upload failed:", errorMessage, result);
            displayFeedback(`Error: ${errorMessage}`, true);
            showLoading(false);
        }
        // --- END OF CORRECT LOCATION ---
    }

    // --- Helper Functions (Defined INSIDE DOMContentLoaded) ---
    function validateFile(file) { /* ... as before ... */ return null;}
    function showLoading(isLoading) { /* ... as before ... */ }
    function displayFeedback(message, isError = false, autoDismiss = true) { /* ... as before ... */ }
    function clearFeedback() { /* ... as before ... */ }
    function clearFeedbackAnimated() { /* ... as before ... */ }

}); // End DOMContentLoaded

// --- DO NOT PUT CODE OUTSIDE DOMContentLoaded like the previous attempt ---