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
        console.error("Check for IDs: 'analysisFile', 'uploadButton', 'upload-feedback', 'loadingIndicator'.");
        if (feedbackDiv) {
            displayFeedback("Page Initialization Error: Cannot find required elements. Upload functionality may be broken.", true, false);
        }
        return; // Stop execution
    }
    console.log("Element check passed - All critical elements found.");

    // --- Initial State ---
    uploadButton.disabled = true;

    // --- Event Listeners ---
    console.log('Attaching listener to fileInput:', fileInput);
    fileInput.addEventListener('change', () => {
        console.log('File input changed.'); // Debug Log 1
        if (fileInput.files && fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const validationError = validateFile(file);
            if (validationError) {
                console.log('File validation failed:', validationError); // Debug Log 2
                displayFeedback(validationError, true);
                uploadButton.disabled = true;
                fileInput.value = '';
            } else {
                console.log('File validation passed.'); // Debug Log 3
                uploadButton.disabled = false;
                clearFeedback();
            }
        } else {
            console.log('File input cleared.'); // Debug Log 4
            uploadButton.disabled = true;
            clearFeedback();
        }
    });

    console.log('Attaching listener to uploadButton:', uploadButton);
    uploadButton.addEventListener('click', handleUpload);

    // --- Core Upload Logic ---
    async function handleUpload() {
        console.log('handleUpload function EXECUTED!'); // Debug Log 5: Check if handler runs
        if (!fileInput.files || fileInput.files.length === 0) {
            console.log('Upload attempt with no file selected.'); // Debug Log 6
            displayFeedback('Please select a file first.', true);
            uploadButton.disabled = true;
            return;
        }
        const file = fileInput.files[0];
        const validationError = validateFile(file);
        if (validationError) {
            console.log('Upload blocked due to validation error:', validationError); // Debug Log 7
            displayFeedback(validationError, true);
            uploadButton.disabled = true;
            return;
        }

        const formData = new FormData();
        formData.append('analysisFile', file);

        showLoading(true);
        displayFeedback('Uploading and processing file...', false, false);
        console.log('Sending fetch request to /upload_analysis_data'); // Debug Log 8

        try {
            const response = await fetch('/upload_analysis_data', {
                method: 'POST',
                body: formData,
            });
            console.log('Received fetch response. Status:', response.status); // Debug Log 9

            await processUploadResponse(response, file.name);

        } catch (error) {
            console.error("Network or client-side upload error:", error); // Debug Log 10
            displayFeedback('A network error occurred. Please check your connection and try again.', true);
            showLoading(false);
        }
    }

    // --- Response Processing ---
    async function processUploadResponse(response, originalFileName) {
        let result;
        const responseText = await response.text(); // Get raw text first for debugging
        console.log('Raw server response text:', responseText); // Debug Log 11

        try {
            result = JSON.parse(responseText); // Now try to parse
            console.log('Parsed JSON response:', result); // Debug Log 12
        } catch (jsonError) {
            console.error("Failed to parse server response as JSON:", jsonError); // Debug Log 13
            const preview = responseText.substring(0, 200) + (responseText.length > 200 ? '...' : '');
            displayFeedback(`Server Error: Unexpected response format (Status: ${response.status}). Response started with: "${preview}". Check browser console.`, true);
            showLoading(false);
            return;
        }

        if (response.ok && result && result.upload_id) {
            console.log('Upload successful. Redirecting...'); // Debug Log 14
            displayFeedback(`Success! File '${result.filename || originalFileName}' uploaded (${result.rows} rows, ${result.columns} columns). Redirecting...`, false, false);
            window.location.href = `/data_cleaner/${result.upload_id}`;
        } else {
            const errorMessage = result?.error || `Upload failed (Status: ${response.status}).`;
            console.error("Upload failed:", errorMessage, result); // Debug Log 15
            displayFeedback(`Error: ${errorMessage}`, true);
            showLoading(false);
        }
    }

    // --- Helper Functions ---

    function validateFile(file) {
        // console.log('Validating file:', file.name, file.type, file.size); // Debug validate start
        const allowedExtensions = ['.csv', '.xlsx'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedExtensions.includes(fileExtension)) {
            // console.log('Validation failed: Invalid extension', fileExtension); // Debug validate fail extension
            return `Invalid file type (${fileExtension}). Please select a CSV or XLSX file.`;
        }

        const maxSizeMB = 50; // Example: 50 MB limit
        const maxSize = maxSizeMB * 1024 * 1024;
        if (file.size > maxSize) {
             // console.log('Validation failed: File too large', file.size); // Debug validate fail size
            return `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum size is ${maxSizeMB} MB.`;
        }

        // console.log('Validation passed for file:', file.name); // Debug validate pass
        return null; // No validation errors found
    }

    function showLoading(isLoading) {
         // console.log('showLoading called with isLoading:', isLoading); // Debug showLoading
         if (isLoading) {
             uploadButton.disabled = true;
             loadingIndicator.style.display = 'inline-block';
             loadingIndicator.setAttribute('aria-busy', 'true');
         } else {
             const currentFile = fileInput.files && fileInput.files.length > 0 ? fileInput.files[0] : null;
             // Only enable if a file exists AND it passes validation
             uploadButton.disabled = !currentFile || !!validateFile(currentFile);
             loadingIndicator.style.display = 'none';
             loadingIndicator.removeAttribute('aria-busy');
         }
    }

    function displayFeedback(message, isError = false, autoDismiss = true) {
        // console.log(`displayFeedback called: msg='${message}', isError=${isError}, autoDismiss=${autoDismiss}`); // Debug displayFeedback
        clearTimeout(feedbackDiv.timeoutId);
        feedbackDiv.timeoutId = null;

        feedbackDiv.textContent = message;
        feedbackDiv.className = `feedback-message alert ${isError ? 'alert-danger' : 'alert-success'}`;
        feedbackDiv.style.display = 'block';
        feedbackDiv.setAttribute('role', isError ? 'alert' : 'status');

        if (autoDismiss) {
            feedbackDiv.timeoutId = setTimeout(() => {
                 if (feedbackDiv.textContent === message) {
                     clearFeedbackAnimated();
                 }
            }, 5000);
        }
    }

    function clearFeedback() {
        // console.log('clearFeedback called.'); // Debug clearFeedback
        clearTimeout(feedbackDiv.timeoutId);
        feedbackDiv.timeoutId = null;
        feedbackDiv.textContent = '';
        feedbackDiv.style.display = 'none';
        feedbackDiv.className = 'feedback-message';
        feedbackDiv.removeAttribute('role');
    }

    function clearFeedbackAnimated() {
         // console.log('clearFeedbackAnimated called.'); // Debug clearFeedbackAnimated
         feedbackDiv.style.opacity = '0'; // Ensure CSS transition is set
         setTimeout(() => {
             clearFeedback();
             feedbackDiv.style.opacity = '1';
         }, 300); // Match CSS transition duration
    }
     // Add CSS transition for the fade effect (in your data_analyzer.css):
     /*
     #upload-feedback {
         transition: opacity 0.3s ease-in-out;
     }
     */

}); // End DOMContentLoaded