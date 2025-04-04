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
const STATES = ["Rest", "Eyes Open", "Eyes Closed"];

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
                
                // Check for required columns
                const missingColumns = FEATURE_COLUMNS.filter(col => !headers.includes(col));
                
                if (missingColumns.length > 0) {
                    reject(`Missing required columns: ${missingColumns.join(', ')}`);
                    return;
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
            predictionCounts[state] = (predictionCounts[state] || 0) + 1;
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
        let html = `<h4>Analysis Results</h4>`;
        html += `<p>Analyzed ${totalSamples} data points</p>`;
        html += `<p>Dominant Brain State: <strong>${dominantState}</strong></p>`;
        html += `<h5>State Distribution:</h5><ul>`;
        
        for (const [state, count] of Object.entries(predictionCounts)) {
            const percentage = ((count / totalSamples) * 100).toFixed(1);
            html += `<li>${state}: ${count} (${percentage}%)</li>`;
        }
        
        html += `</ul>`;
        predictionText.innerHTML = html;
    }
    
    // Display visualizations
    if (plotContainer && data.visualizations) {
        plotContainer.innerHTML = '';
        
        for (const [key, imageBase64] of Object.entries(data.visualizations)) {
            const title = key.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
            
            const plotItem = document.createElement('div');
            plotItem.className = 'plotItem';
            plotItem.innerHTML = `
                <h4>${title}</h4>
                <img src="data:image/png;base64,${imageBase64}" alt="${title}">
            `;
            
            plotContainer.appendChild(plotItem);
        }
    }
}

// Export functions to be accessible from other scripts
window.modelFunctions = {
    validateCSVFile,
    showErrorMessage,
    setLoadingState,
    processResponse
};
