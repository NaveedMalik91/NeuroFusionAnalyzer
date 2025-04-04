/**
 * Script for model-specific functionality
 * This contains functions for data processing and visualization related to the model
 */

// Constants for the expected data format
const FEATURE_COLUMNS = [
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power", 
    "Spectral Entropy", "Mean Coherence", "BOLD Mean", "BOLD Variance", 
    "ALFF Mean", "ALFF Variance"
];

// Brain states/classes
const STATES = ["Rest", "Sleep"];

/**
 * Validates that the uploaded file matches the expected format
 * @param {File} file - The uploaded CSV file
 * @returns {Promise} - Resolves with true if valid, rejects with error message if not
 */
function validateCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const content = e.target.result;
                const lines = content.split('\n');
                
                // Check if file has headers
                if (lines.length < 2) {
                    reject('File appears to be empty or has no data rows');
                    return;
                }
                
                // Check header row
                const headers = lines[0].split(',').map(h => h.trim());
                
                // Check if the headers exactly match our expected columns in the correct order
                if (headers.length !== FEATURE_COLUMNS.length) {
                    reject(`Invalid number of columns. Expected ${FEATURE_COLUMNS.length} columns but found ${headers.length}.`);
                    return;
                }
                
                // Check if columns match exactly in the correct order
                for (let i = 0; i < FEATURE_COLUMNS.length; i++) {
                    if (headers[i] !== FEATURE_COLUMNS[i]) {
                        reject(`Column mismatch at position ${i+1}. Expected '${FEATURE_COLUMNS[i]}' but found '${headers[i]}'.`);
                        return;
                    }
                }
                
                resolve(true);
            } catch (error) {
                reject(`Error processing file: ${error.message}`);
            }
        };
        
        reader.onerror = function() {
            reject('Error reading file');
        };
        
        reader.readAsText(file);
    });
}

/**
 * Displays an error message in the UI
 * @param {string} message - The error message to display
 */
function showErrorMessage(message) {
    const predictionText = document.getElementById('predictionText');
    if (predictionText) {
        predictionText.innerHTML = `<div class="alert alert-danger">${message}</div>`;
    } else {
        alert(message);
    }
}

/**
 * Updates the UI to show loading state
 * @param {boolean} isLoading - Whether loading is in progress
 */
function setLoadingState(isLoading) {
    const submitBtn = document.querySelector('.submit-btn');
    if (submitBtn) {
        submitBtn.disabled = isLoading;
        submitBtn.textContent = isLoading ? 'Processing...' : 'Run Analysis';
    }
}

/**
 * Processes a successful response from the server
 * @param {Object} data - The response data from the server
 */
function processResponse(data) {
    const predictionText = document.getElementById('predictionText');
    const plotContainer = document.getElementById('plotContainer');
    
    // Check if the response has an error
    if (data.error) {
        showErrorMessage(data.error);
        return;
    }
    
    // Display predictions
    if (predictionText && data.predictions) {
        const predictionCounts = {};
        data.predictions.forEach(state => {
            // Convert numeric prediction to state name if needed
            const stateName = typeof state === 'number' ? STATES[state] : state;
            predictionCounts[stateName] = (predictionCounts[stateName] || 0) + 1;
        });
        
        // Find the dominant state
        let dominantState = '';
        let maxCount = 0;
        for (const [state, count] of Object.entries(predictionCounts)) {
            if (count > maxCount) {
                maxCount = count;
                dominantState = state;
            }
        }
        
        // Format and display the prediction results
        const totalSamples = data.predictions.length;
        let html = `<h4 class="result-title">Analysis Results</h4>`;
        html += `<p>Analyzed ${totalSamples} data points</p>`;
        html += `<p>Dominant Brain State: <strong class="${dominantState === 'Sleep' ? 'sleep-state' : 'rest-state'}">${dominantState}</strong></p>`;
        html += `<h5>State Distribution:</h5><ul class="state-distribution">`;
        
        for (const [state, count] of Object.entries(predictionCounts)) {
            const percentage = ((count / totalSamples) * 100).toFixed(1);
            const stateClass = state === 'Sleep' ? 'sleep-state' : 'rest-state';
            html += `<li class="${stateClass}"><span class="state-name">${state}:</span> <span class="state-count">${count}</span> <span class="state-percentage">(${percentage}%)</span></li>`;
        }
        
        html += `</ul>`;
        
        // Add pagination controls if we have more than 20 entries
        if (data.pagination && data.individualPredictions && data.individualPredictions.length > 20) {
            html += generatePaginationControls(data.pagination);
        }
        
        predictionText.innerHTML = html;
        
        // Now handle displaying individual prediction entries if available
        if (data.individualPredictions) {
            displayIndividualPredictions(data.individualPredictions);
        }
    }
    
    // Display visualizations
    if (plotContainer && data.visualizations) {
        plotContainer.innerHTML = '';
        
        for (const [key, imageBase64] of Object.entries(data.visualizations)) {
            const title = formatVisualizationTitle(key);
            const description = getVisualizationDescription(key);
            
            const plotItem = document.createElement('div');
            plotItem.className = 'visualization-item';
            plotItem.innerHTML = `
                <div class="visualization-header">
                    <h4>${title}</h4>
                </div>
                <div class="visualization-content">
                    <img src="data:image/png;base64,${imageBase64}" alt="${title}" class="visualization-image">
                    <div class="visualization-description">
                        <p>${description}</p>
                    </div>
                </div>
            `;
            
            plotContainer.appendChild(plotItem);
        }
    }
}

/**
 * Format visualization title from snake_case key to Title Case
 * @param {string} key - The visualization key
 * @returns {string} - Formatted title
 */
function formatVisualizationTitle(key) {
    return key.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

/**
 * Get description text for a specific visualization
 * @param {string} key - The visualization key
 * @returns {string} - Description text
 */
function getVisualizationDescription(key) {
    const descriptions = {
        'feature_distributions': 'Shows the distribution of EEG features (Delta, Theta, Alpha, Beta, Gamma power) and fMRI features (BOLD signal, ALFF) across all data points. This helps identify patterns in the neurophysiological data.',
        'prediction_distribution': 'Displays the overall percentage breakdown of predicted brain states (Rest vs. Sleep), giving you a quick overview of the dominant state in your dataset.',
        'probability_heatmap': 'Visualizes the model\'s confidence for each prediction as a heatmap. Brighter colors indicate higher confidence in the classification decision.',
        'feature_importance': 'Ranks the features (both EEG and fMRI) by their importance in making predictions, revealing which neural signals are most relevant for discriminating between Rest and Sleep states.',
        'time_series_trends': 'Tracks how key features change across sequential data points, potentially revealing temporal patterns in brain activity.',
        'confusion_matrix': 'For datasets with known labels, shows the accuracy of predictions by comparing true vs. predicted states, helping evaluate model performance.',
        'feature_correlations': 'Examines relationships between different EEG and fMRI features, revealing how different neurophysiological measures interact.',
        'individual_entries': 'Detailed visualization of individual data points with their predictions, allowing in-depth examination of specific cases.'
    };
    
    return descriptions[key] || 'This visualization provides insights into the neurophysiological data patterns.';
}

/**
 * Generate pagination controls for individual predictions
 * @param {Object} pagination - Pagination information
 * @returns {string} - HTML for pagination controls
 */
function generatePaginationControls(pagination) {
    const {current_page, total_pages, has_prev, has_next} = pagination;
    
    let html = `<div class="pagination-controls">`;
    html += `<button id="prevPage" class="pagination-btn" ${!has_prev ? 'disabled' : ''}>Previous</button>`;
    html += `<span class="page-info">Page ${current_page} of ${total_pages}</span>`;
    html += `<button id="nextPage" class="pagination-btn" ${!has_next ? 'disabled' : ''}>Next</button>`;
    html += `</div>`;
    
    return html;
}

/**
 * Display individual prediction entries
 * @param {Array} predictions - Array of individual prediction objects
 */
function displayIndividualPredictions(predictions) {
    const individualPredContainer = document.getElementById('individualPredictions');
    if (!individualPredContainer) return;
    
    let html = `<h4>Individual Entries</h4>`;
    html += `<div class="predictions-table-container">`;
    html += `<table class="predictions-table">`;
    html += `<thead><tr>
                <th>Entry</th>
                <th>Prediction</th>
                <th>Confidence</th>
             </tr></thead>`;
    html += `<tbody>`;
    
    predictions.forEach((pred, index) => {
        const stateClass = pred.prediction === 1 ? 'sleep-state' : 'rest-state';
        const stateName = pred.prediction === 1 ? 'Sleep' : 'Rest';
        const confidence = (pred.probability * 100).toFixed(2);
        
        html += `<tr>
                    <td>${index + 1}</td>
                    <td class="${stateClass}">${stateName}</td>
                    <td>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar ${stateClass}" style="width: ${confidence}%"></div>
                            <span class="confidence-text">${confidence}%</span>
                        </div>
                    </td>
                 </tr>`;
    });
    
    html += `</tbody></table></div>`;
    
    individualPredContainer.innerHTML = html;
    
    // Add event listeners for pagination buttons
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    if (prevBtn) {
        prevBtn.addEventListener('click', () => changePage('prev'));
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', () => changePage('next'));
    }
}

/**
 * Change page for paginated results
 * @param {string} direction - Direction to change page ('prev' or 'next')
 */
function changePage(direction) {
    // Get the current form data
    const fileInput = document.getElementById('fileInput');
    if (!fileInput || !fileInput.files[0]) return;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Get current page from button data attribute or hidden input
    const currentPage = document.querySelector('.page-info').textContent.split(' ')[1];
    const nextPage = direction === 'next' ? parseInt(currentPage) + 1 : parseInt(currentPage) - 1;
    
    formData.append('page', nextPage);
    
    // Show loading state
    setLoadingState(true);
    
    // Submit for the new page
    fetch('/upload_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        processResponse(data);
        setLoadingState(false);
    })
    .catch(error => {
        showErrorMessage(`Error: ${error.message}`);
        setLoadingState(false);
    });
}

// Export functions to be accessible from other scripts
window.modelFunctions = {
    validateCSVFile,
    showErrorMessage,
    setLoadingState,
    processResponse
};
