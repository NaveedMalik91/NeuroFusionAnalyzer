// Model1-specific functionality
const modelFunctions = {
    // Show the loading spinner
    setLoadingState: function(isLoading) {
        const loadingSpinner = document.getElementById('loadingSpinner');
        if (loadingSpinner) {
            loadingSpinner.style.display = isLoading ? 'flex' : 'none';
        }
    },
    
    // Show error message
    showErrorMessage: function(message) {
        // Create error alert
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        errorAlert.role = 'alert';
        errorAlert.innerHTML = `
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to document
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            alertContainer.innerHTML = '';
            alertContainer.appendChild(errorAlert);
            
            // Auto dismiss after 5 seconds
            setTimeout(() => {
                errorAlert.classList.remove('show');
                setTimeout(() => {
                    alertContainer.innerHTML = '';
                }, 150);
            }, 5000);
        }
    },
    
    // Validate CSV file
    validateCSVFile: function(file) {
        return new Promise((resolve, reject) => {
            // Check file type
            if (!file.name.endsWith('.csv')) {
                reject('Please upload a CSV file');
                return;
            }
            
            // Check file size (max 5MB)
            if (file.size > 5 * 1024 * 1024) {
                reject('File is too large. Maximum size is 5MB');
                return;
            }
            
            // Read file header to check required columns
            const reader = new FileReader();
            reader.onload = function(e) {
                const contents = e.target.result;
                const lines = contents.split('\n');
                
                if (lines.length < 2) {
                    reject('File is empty or has no data rows');
                    return;
                }
                
                // Check header row
                const headers = lines[0].split(',').map(h => h.trim());
                
                // Required columns
                const requiredColumns = [
                    "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power", 
                    "Spectral Entropy", "Mean Coherence", "BOLD Mean", "BOLD Variance", 
                    "ALFF Mean", "ALFF Variance"
                ];
                
                const missingColumns = requiredColumns.filter(col => !headers.includes(col));
                
                if (missingColumns.length > 0) {
                    reject(`Missing required columns: ${missingColumns.join(', ')}`);
                    return;
                }
                
                resolve();
            };
            
            reader.onerror = function() {
                reject('Error reading file');
            };
            
            reader.readAsText(file);
        });
    },
    
    // Process response from server
    processResponse: function(data) {
        if (data.error) {
            this.showErrorMessage(data.error);
            return;
        }
        
        // Display predictions if available
        if (data.predictions) {
            const predictionsList = document.getElementById('predictionsList');
            if (predictionsList) {
                predictionsList.innerHTML = '';
                
                // Convert numeric predictions to brain state names
                const stateNames = ['Rest', 'Eyes Open', 'Eyes Closed'];
                const predictions = data.predictions.map(p => stateNames[p]);
                
                // Create prediction items
                predictions.forEach((prediction, index) => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item d-flex justify-content-between align-items-center';
                    li.textContent = `Row ${index + 1}: ${prediction}`;
                    
                    // Add badge with color based on prediction
                    const badge = document.createElement('span');
                    badge.className = 'badge';
                    
                    // Different colors for different states
                    if (prediction === 'Rest') {
                        badge.className += ' bg-primary';
                    } else if (prediction === 'Eyes Open') {
                        badge.className += ' bg-success';
                    } else {
                        badge.className += ' bg-warning';
                    }
                    
                    badge.textContent = prediction;
                    li.appendChild(badge);
                    
                    predictionsList.appendChild(li);
                });
            }
        }
        
        // Display visualizations if available
        if (data.visualizations) {
            // Feature distributions
            if (data.visualizations.feature_distributions) {
                const featureDistImg = document.getElementById('featureDistImg');
                if (featureDistImg) {
                    featureDistImg.src = `data:image/png;base64,${data.visualizations.feature_distributions}`;
                    featureDistImg.style.display = 'block';
                }
            }
            
            // Confusion matrix
            if (data.visualizations.confusion_matrix) {
                const confusionMatrixImg = document.getElementById('confusionMatrixImg');
                const confusionMatrixCard = document.getElementById('confusionMatrixCard');
                
                if (confusionMatrixImg && confusionMatrixCard) {
                    confusionMatrixImg.src = `data:image/png;base64,${data.visualizations.confusion_matrix}`;
                    confusionMatrixCard.style.display = 'block';
                }
            } else {
                // Hide confusion matrix if no ground truth labels
                const confusionMatrixCard = document.getElementById('confusionMatrixCard');
                if (confusionMatrixCard) {
                    confusionMatrixCard.style.display = 'none';
                }
            }
            
            // Feature correlations
            if (data.visualizations.feature_correlations) {
                const correlationsImg = document.getElementById('correlationsImg');
                if (correlationsImg) {
                    correlationsImg.src = `data:image/png;base64,${data.visualizations.feature_correlations}`;
                    correlationsImg.style.display = 'block';
                }
            }
            
            // Prediction distributions
            if (data.visualizations.prediction_distributions) {
                const predDistImg = document.getElementById('predDistImg');
                if (predDistImg) {
                    predDistImg.src = `data:image/png;base64,${data.visualizations.prediction_distributions}`;
                    predDistImg.style.display = 'block';
                }
            }
        }
    }
};

// Make the functions available globally
window.modelFunctions = modelFunctions;
