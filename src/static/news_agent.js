document.addEventListener('DOMContentLoaded', () => {
    console.log("DOMContentLoaded fired for:", window.location.pathname); // Debug: Check DOM ready

    // --- DOM Elements ---
    const mainNewsContentDiv = document.getElementById('main-news-content');
    const notificationPanelDiv = document.getElementById('notification-panel');
    const statusUpdateSpan = document.getElementById('status-update');
    const liveHeadlinesSpan = document.getElementById('live-headlines');
    const summarizeReadBtn = document.getElementById('summarize-read-btn');
    const stopReadingBtn = document.getElementById('stop-reading-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const notificationTemplate = document.getElementById('notification-template');
    const mainNewsControlsDiv = document.querySelector('.main-news-controls');

    // --- Element Existence Check (Initial & Critical) ---
    console.log("--- Running Initial Element Checks ---");
    const criticalElements = {
        mainNewsContentDiv, notificationPanelDiv, statusUpdateSpan,
        liveHeadlinesSpan, summarizeReadBtn, stopReadingBtn,
        themeToggleBtn, notificationTemplate, mainNewsControlsDiv
    };
    let allElementsFound = true;
    for (const key in criticalElements) {
        console.log(`Element check - ${key}:`, criticalElements[key]);
        if (!criticalElements[key]) {
            // Log error for missing elements, but allow script to continue if possible
            // Critical ones like statusUpdateSpan might break functionality later.
            console.error(`News Agent Script Warning: UI Element missing: ${key}. Some features might not work.`);
            if (key === 'statusUpdateSpan' || key === 'mainNewsContentDiv' || key === 'notificationPanelDiv') {
                allElementsFound = false; // Mark if truly critical ones are missing
            }
        }
    }
     if (!allElementsFound) {
          console.error("News Agent Script Error: Essential page elements (status, content, or notification panel) are missing. Aborting initial setup.");
          // Display a user-facing error if possible
          if (statusUpdateSpan) statusUpdateSpan.textContent = "Status: Page Error!";
          if (mainNewsContentDiv) mainNewsContentDiv.innerHTML = "<p class='text-danger'>Page failed to initialize correctly.</p>";
          return; // Stop if essential elements are gone
     }
     console.log("--- Initial Element Checks Passed (All essential elements found) ---");


    // --- State Variables ---
    let currentMainArticle = null; // Stores the full article object currently in the main view
    let lastFetchTime = 0;         // Timestamp of the last successfully processed fetch
    let knownNotificationUrls = new Set(); // Tracks URLs already shown in notifications
    let isReading = false;          // Flag for Text-to-Speech state
    let pollingIntervalId = null;   // To store the interval ID for potential clearing
    const POLLING_INTERVAL = 30000; // Fetch news every 30 seconds (30000 ms)

    // --- Helper Functions ---

    /** Shows/hides a loading indicator element and disables controls in its parent panel. */
    function showLoading(elementId, show = true) {
        // console.log(`showLoading called for ${elementId}, show=${show}`); // Optional log
        const indicator = document.getElementById(elementId);
        if (indicator) {
            indicator.style.display = show ? 'flex' : 'none';
            // Find the parent panel to disable controls within it
            const parentPanel = indicator.closest('.panel, .main-news-display-area'); // Adapt selectors if needed
            if (parentPanel) {
                const controls = parentPanel.querySelectorAll('button, input, select');
                controls.forEach(control => control.disabled = show);
            }
        } else {
            // console.warn(`Loading indicator element not found: ${elementId}`);
        }
    }

    /** Updates the status text in the bottom bar. */
    function updateStatus(message, isError = false) {
        console.log(`updateStatus called: msg='${message}', isError=${isError}`); // Debug status update
        if (statusUpdateSpan) {
            statusUpdateSpan.textContent = `Status: ${message}`;
            // Use CSS variables which handle theme automatically via the class on body
            statusUpdateSpan.style.color = isError ? 'var(--status-text-light)' : 'inherit';
            // Add a specific class for error styling if needed, remove if not error
            statusUpdateSpan.classList.toggle('status-error', isError);
        } else {
             console.error("statusUpdateSpan element not found! Cannot update status.");
        }
    }

    /** Updates the live headlines text in the bottom bar. */
    function updateLiveHeadlines(text = "") {
         console.log(`updateLiveHeadlines called: text='${text}'`); // Debug headline update
         if (liveHeadlinesSpan) {
             liveHeadlinesSpan.innerHTML = text ? `<strong>${text}</strong>` : '';
         } else {
              console.warn("liveHeadlinesSpan element not found! Cannot update headlines.");
         }
    }

    /** Formats a date string into a relative time string (e.g., "5m ago"). */
    function formatTimeAgo(dateString) {
        if (!dateString) return '';
        try {
            const date = new Date(dateString);
            const now = new Date();
            const secondsPast = (now.getTime() - date.getTime()) / 1000;

            if (secondsPast < 60) return `${Math.round(secondsPast)}s ago`;
            if (secondsPast < 3600) return `${Math.round(secondsPast / 60)}m ago`;
            if (secondsPast <= 86400 * 2) return `${Math.round(secondsPast / 3600)}h ago`; // Show hours up to 2 days

            // Format older dates simply
            const options = { year: 'numeric', month: 'short', day: 'numeric' };
            return date.toLocaleDateString(undefined, options);

        } catch (e) {
            console.error("Error parsing date:", dateString, e);
            return dateString; // Return original string if parsing fails
        }
    }

    // --- News Fetching & Display Logic ---

    /** Fetches news from the backend and initiates processing. */
    async function fetchNews() {
        console.log("--- fetchNews initiated ---"); // Debug fetch start
        console.log("Current NEWS_API_AVAILABLE flag:", window.NEWS_API_AVAILABLE);

        if (!window.NEWS_API_AVAILABLE) {
            updateStatus("News API Key missing on server", true);
            showLoading('mainNewsLoading', false);
            if (mainNewsContentDiv) mainNewsContentDiv.innerHTML = '<p class="placeholder-text text-danger">News Agent Disabled: API Key Missing.</p>';
            console.log("fetchNews aborted: API key not available.");
            // Stop polling if the key is missing
            if (pollingIntervalId) clearInterval(pollingIntervalId);
            pollingIntervalId = null;
            return;
        }

        updateStatus("Fetching live news...");
        showLoading('mainNewsLoading', true); // Show loading for main area

        try {
            console.log("Sending request to /fetch_live_news"); // Debug fetch URL
            const response = await fetch('/fetch_live_news');
            console.log("Fetch response received. Status:", response.status); // Debug fetch status

            // Try to parse JSON regardless of status to get potential error messages
            let data;
            const responseText = await response.text(); // Get raw text for better debugging
            console.log("Raw response text:", responseText.substring(0, 500) + (responseText.length > 500 ? '...' : '')); // Log response start

            try {
                data = JSON.parse(responseText);
                console.log("Parsed response data:", data); // Log parsed data
            } catch (jsonError) {
                console.error("Failed to parse server response as JSON:", jsonError);
                // Throw a specific error that includes the status and raw text hint
                throw new Error(`Server returned non-JSON response (Status: ${response.status}). Starts with: "${responseText.substring(0, 100)}..."`);
            }

            // Now check response.ok AND the status field within the JSON
            if (!response.ok || data.status !== 'ok') {
                const errorMessage = data.error || data.message || `Unknown error (Status: ${response.status})`;
                 console.error("Fetch error response from server:", data); // Log error details
                throw new Error(errorMessage);
            }

            // Success Path
            if (data.articles && Array.isArray(data.articles)) {
                const fetchTimestamp = Date.now();
                console.log(`Fetched ${data.articles.length} articles. Processing...`); // Log article count
                processNews(data.articles, fetchTimestamp); // Process the fetched articles
                updateStatus("News updated successfully.", false);
                updateLiveHeadlines("Live headlines showing.");
            } else {
                console.log("No articles array found or empty in the response."); // Log no articles
                updateStatus("No new articles found.", false);
                updateLiveHeadlines(""); // Clear if no articles
            }

        } catch (error) {
            console.error("Error during fetchNews:", error); // Log caught error
            updateStatus(`Error fetching news: ${error.message || error}`, true); // Display error message
            updateLiveHeadlines(""); // Clear headlines on error
        } finally {
            console.log("--- fetchNews finally block ---"); // Debug finally
            showLoading('mainNewsLoading', false); // Ensure loading is always hidden
        }
    }

    /** Processes fetched articles, updates main display and notifications. */
    function processNews(articles, fetchTimestamp) {
         console.log(`processNews called. Articles: ${articles.length}, FetchTime: ${fetchTimestamp}, LastTime: ${lastFetchTime}`);
        if (fetchTimestamp <= lastFetchTime) {
            console.log("Skipping processNews: fetched data is not newer.");
            return; // Avoid processing old data if fetches overlap
        }

        // 1. Update Main Display (Show the *latest* article if it changed)
        const latestArticle = articles.length > 0 ? articles[0] : null;
        if (latestArticle && latestArticle.url && latestArticle.url !== currentMainArticle?.url) {
            console.log("New latest article found, updating main display:", latestArticle.title);
            displayMainArticle(latestArticle);
        } else if (!currentMainArticle && latestArticle) {
            // If nothing is displayed yet, show the first one
            console.log("No current main article, displaying first fetched article:", latestArticle.title);
            displayMainArticle(latestArticle);
        } else {
             console.log("Main article content condition not met (no new article, or no articles fetched).");
        }


        // 2. Update Notifications (Show *new* articles since last time)
        let newNotifications = 0;
        // Clear "Waiting..." message only if we have articles and the message exists
        if (articles.length > 0 && notificationPanelDiv && notificationPanelDiv.querySelector('.text-muted.small.p-2')) {
             console.log("Clearing 'Waiting...' message from notifications.");
             notificationPanelDiv.innerHTML = '';
        }

        if (notificationPanelDiv) {
            // Iterate backwards through NEW articles to prepend newest first
            for (let i = articles.length - 1; i >= 0; i--) {
                 const article = articles[i];
                 // Ensure article has a URL and hasn't been seen before
                 if (article.url && !knownNotificationUrls.has(article.url)) {
                     addNotification(article);
                     knownNotificationUrls.add(article.url); // Mark as seen
                     newNotifications++;
                 }
             }
             // Optional: Limit notification queue size
             const maxNotifications = 50;
             while (notificationPanelDiv.children.length > maxNotifications) {
                 const oldestNotification = notificationPanelDiv.lastElementChild;
                 if (oldestNotification) {
                     const urlToRemove = oldestNotification.dataset.url;
                     if(urlToRemove) knownNotificationUrls.delete(urlToRemove);
                     notificationPanelDiv.removeChild(oldestNotification);
                     // console.log(`Removed oldest notification (${urlToRemove}) to maintain limit.`);
                 } else {
                     break; // Should not happen if children exist
                 }
             }
        } else {
             console.warn("Notification panel div not found, cannot add notifications.");
        }

         console.log(`Processed news. Added ${newNotifications} notifications. Total known URLs: ${knownNotificationUrls.size}`);
         lastFetchTime = fetchTimestamp; // Update timestamp ONLY after successful processing
    }

    /** Renders a single article in the main display area. */
    function displayMainArticle(article) {
        console.log("Displaying main article:", article?.title);
        if (!article || !article.url) {
            console.warn("Attempted to display invalid or incomplete article.");
            return;
        }
        currentMainArticle = article; // Store current article details

        if (!mainNewsContentDiv || !mainNewsControlsDiv) {
            console.error("Cannot display main article, essential DOM elements (mainNewsContentDiv or mainNewsControlsDiv) missing.");
            return;
        }

        let contentHtml = '';
        // Use loading="lazy" for images
        if (article.urlToImage) {
            contentHtml += `<img src="${article.urlToImage}" alt="Article image" class="img-fluid mb-3" loading="lazy" onerror="this.style.display='none'; console.warn('Image failed to load: ${article.urlToImage}');">`; // Add basic error handling
        }
        contentHtml += `<h3>${article.title || 'No Title'}</h3>`;
        if (article.source?.name || article.publishedAt) {
             contentHtml += `<p class="text-muted small mb-2">`;
             if(article.source?.name) contentHtml += `Source: ${article.source.name}`;
             if(article.source?.name && article.publishedAt) contentHtml += ` | `;
             if(article.publishedAt) contentHtml += `Published: ${formatTimeAgo(article.publishedAt)}`;
             contentHtml += `</p>`;
        }
        // Prioritize 'content', then 'description'. Handle null/undefined safely.
        const articleText = article.content || article.description || '';
        contentHtml += `<p>${articleText || '<em class="text-muted">No content available for this article.</em>'}</p>`;

        // Provide link to original source
        if (article.url) {
            contentHtml += `<a href="${article.url}" target="_blank" rel="noopener noreferrer" class="btn btn-outline-secondary btn-sm">Read Full Story <i class="fas fa-external-link-alt fa-xs"></i></a>`;
        }

        mainNewsContentDiv.innerHTML = contentHtml;
        mainNewsContentDiv.classList.remove('wavy-background'); // Remove placeholder style
        mainNewsControlsDiv.style.display = 'block'; // Show controls
        summarizeReadBtn.disabled = !articleText; // Enable only if actual text exists
        stopReading(); // Stop any previous reading when loading new article
    }

    /** Creates and prepends a notification item to the sidebar. */
    function addNotification(article) {
        // console.log("Adding notification for:", article?.title); // Optional log
         if (!notificationTemplate || !notificationPanelDiv) {
             console.warn("Cannot add notification, template or panel missing.");
             return;
         }
         if (!article || !article.url || !article.title) {
              console.warn("Skipping notification for incomplete article data:", article);
              return;
         }

         try {
             const notificationClone = notificationTemplate.content.cloneNode(true);
             const itemDiv = notificationClone.querySelector('.notification-item');
             if (!itemDiv) { console.error("Notification template missing .notification-item structure"); return; }
             itemDiv.dataset.url = article.url; // Store url for tracking removal if needed

             const sourceEl = itemDiv.querySelector('.notification-source');
             const titleEl = itemDiv.querySelector('.notification-title');
             const timeEl = itemDiv.querySelector('.notification-time');

             if (sourceEl) sourceEl.textContent = article.source?.name || 'Unknown Source';
             if (titleEl) titleEl.textContent = article.title; // Title is required now
             if (timeEl) timeEl.textContent = formatTimeAgo(article.publishedAt);

             itemDiv.addEventListener('click', (e) => {
                 e.preventDefault(); // Prevent potential default behaviors
                 console.log("Notification clicked:", article.title);
                 displayMainArticle(article);
                 if (mainNewsContentDiv) mainNewsContentDiv.scrollTop = 0; // Scroll main view to top
             });

             notificationPanelDiv.prepend(notificationClone); // Add to top of the list
         } catch(e) {
              console.error("Error creating notification element:", e);
         }
    }

    // --- Summarization and Text-to-Speech (TTS) ---

    /** Handles click on the Summarize & Read button. */
    async function handleSummarizeAndRead() {
        console.log("handleSummarizeAndRead called."); // Debug summarize click
        if (!currentMainArticle || isReading) {
             console.warn("Summarize aborted: No current article or already reading.");
             return;
        }

        // Use the stored full article object
        const content = currentMainArticle.content || currentMainArticle.description;
        const title = currentMainArticle.title;

        if (!content) {
            console.warn("No content available in the main article to summarize.");
            alert("No content available in the main article to summarize.");
            return;
        }

        summarizeReadBtn.disabled = true;
        stopReadingBtn.style.display = 'none'; // Hide stop initially
        showLoading('summaryLoading', true);
        updateStatus("Summarizing article...");
        console.log("Sending content for summarization:", title, content.substring(0, 80) + "...");

        try {
            const apiUrl = '/summarize_news';
            const payload = { content: content, title: title };
            console.log("Calling fetchApi for summarization:", apiUrl, payload); // Debug API call
            const result = await fetchApi(apiUrl, 'POST', payload);
            console.log("Summarize fetchApi result:", result); // Debug API result

            showLoading('summaryLoading', false);

            if (result.ok && result.data.summary && !result.data.summary.startsWith("[AI")) {
                console.log("Summary received, calling speakText:", result.data.summary);
                updateStatus("Summary complete. Reading aloud...", false);
                speakText(result.data.summary);
            } else {
                 // Throw error to be caught below
                 throw new Error(result.data.error || result.data.summary || "Failed to get valid summary.");
            }
        } catch (error) {
             console.error("Error in handleSummarizeAndRead:", error);
             updateStatus(`Summarization Error: ${error.message}`, true);
             showLoading('summaryLoading', false);
             // Re-enable button only if there's content to try again
             summarizeReadBtn.disabled = !content;
        }
    }

    /** Uses the browser's SpeechSynthesis API to read text aloud. */
    function speakText(text) {
         console.log("speakText called with text:", text.substring(0, 80) + "...");
        if (!('speechSynthesis' in window)) {
            console.error("TTS not supported by browser.");
            alert("Sorry, your browser doesn't support Text-to-Speech.");
            updateStatus("TTS not supported by browser.", true);
            summarizeReadBtn.disabled = !(currentMainArticle?.content || currentMainArticle?.description); // Re-enable if content exists
            return;
        }

        // Ensure any previous speech is stopped *before* creating new utterance
        stopReading();

        const utterance = new SpeechSynthesisUtterance(text);
        // Optional: Configure voice, rate, pitch here if desired
        // utterance.lang = 'en-US'; // Example: Set language

        utterance.onstart = () => {
             console.log("TTS playback started.");
             isReading = true;
             summarizeReadBtn.disabled = true; // Keep disabled while reading
             if(stopReadingBtn) stopReadingBtn.style.display = 'inline-block'; // Show stop button
             updateStatus("Reading summary aloud...", false);
        };

        utterance.onend = () => {
            console.log("TTS playback finished.");
            isReading = false;
            // Re-enable summarize button ONLY if there is content in the currently displayed article
            if(summarizeReadBtn) summarizeReadBtn.disabled = !(currentMainArticle?.content || currentMainArticle?.description);
            if(stopReadingBtn) stopReadingBtn.style.display = 'none';
            updateStatus("Finished reading summary.", false);
        };

        utterance.onerror = (event) => {
             console.error("TTS error:", event.error, event);
             isReading = false;
             if(summarizeReadBtn) summarizeReadBtn.disabled = !(currentMainArticle?.content || currentMainArticle?.description);
             if(stopReadingBtn) stopReadingBtn.style.display = 'none';
             updateStatus(`TTS Error: ${event.error}`, true);
        };

        // Small delay before speaking, sometimes helps avoid issues
        setTimeout(() => {
            console.log("Issuing speechSynthesis.speak command.");
            window.speechSynthesis.speak(utterance);
        }, 100);

    }

    /** Stops any ongoing TTS playback. */
    function stopReading() {
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            console.log("Stopping TTS playback.");
            window.speechSynthesis.cancel(); // Stop current and pending utterances
             // Reset state immediately, don't rely solely on onend event after cancel
             isReading = false;
             if(summarizeReadBtn) summarizeReadBtn.disabled = !(currentMainArticle?.content || currentMainArticle?.description);
             if(stopReadingBtn) stopReadingBtn.style.display = 'none';
             // Don't necessarily change status here, let onend handle natural finish status
        } else {
             // console.log("stopReading called but nothing was speaking."); // Optional log
        }
    }


    // --- Theme Toggle ---

    /** Applies the 'dark-theme' class to the body. */
    function applyTheme(theme) {
        console.log("Applying theme:", theme);
        if (theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else {
            document.body.classList.remove('dark-theme');
        }
    }

    /** Attaches listener to the theme toggle button. */
    if (themeToggleBtn) {
         console.log('Attaching listener to themeToggleBtn.');
        themeToggleBtn.addEventListener('click', () => {
            const isDark = document.body.classList.toggle('dark-theme');
            const newTheme = isDark ? 'dark' : 'light';
            try {
                 localStorage.setItem('newsAgentTheme', newTheme); // Save preference
                 console.log("Theme toggled to:", newTheme);
            } catch (e) {
                 console.warn("Could not save theme preference to localStorage:", e);
            }
        });
    } else {
         console.warn("Theme toggle button not found.");
    }


    // --- Initial Setup Execution ---
    console.log("--- Running Initial Page Setup ---");

    // 1. Load saved theme or default
    try {
        const savedTheme = localStorage.getItem('newsAgentTheme') || 'light'; // Default to light
        applyTheme(savedTheme);
    } catch (e) {
        console.warn("Could not load theme preference from localStorage:", e);
        applyTheme('light'); // Default to light on error
    }


     // 2. Attach button listeners only if buttons exist and are found
    if (summarizeReadBtn) {
         console.log('Attaching listener to summarizeReadBtn.');
         summarizeReadBtn.addEventListener('click', handleSummarizeAndRead);
    }
    if (stopReadingBtn) {
         console.log('Attaching listener to stopReadingBtn.');
         stopReadingBtn.addEventListener('click', stopReading);
    }


    // 3. Initial status update (after elements are confirmed)
    updateStatus("Ready.", false);


    // 4. Fetch initial news
    console.log("Calling initial fetchNews().");
    fetchNews(); // Call immediately on load


    // 5. Start polling (only if API key is available initially)
    if (window.NEWS_API_AVAILABLE) {
        console.log(`Starting polling interval (${POLLING_INTERVAL}ms).`);
        if(pollingIntervalId) clearInterval(pollingIntervalId); // Clear any previous just in case
        pollingIntervalId = setInterval(fetchNews, POLLING_INTERVAL);
    } else {
        console.warn("Polling not started because News API key is not available.");
    }

    console.log("--- Initial Page Setup Complete ---");


}); // End DOMContentLoaded