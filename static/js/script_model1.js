document.addEventListener('DOMContentLoaded', function() {
    // Initialize drop area functionality
    initUploadForm();
    
    // Initialize particles.js
    particlesJS("particles-js", {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#ffffff" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: false },
            size: { value: 3, random: true },
            line_linked: {
                enable: true,
                distance: 150,
                color: "#ffffff",
                opacity: 0.4,
                width: 1
            },
            move: {
                enable: true,
                speed: 2,
                direction: "none",
                random: false,
                straight: false,
                out_mode: "out",
                bounce: false
            }
        },
        interactivity: {
            detect_on: "canvas",
            events: {
                onhover: { enable: true, mode: "repulse" },
                onclick: { enable: true, mode: "push" },
                resize: true
            }
        },
        retina_detect: true
    });
});

/**
 * Initialize the upload form and related functionality
 */
function initUploadForm() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const form = document.getElementById('uploadForm');
    const resetBtn = document.querySelector('.reset-btn');
    
    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Handle enter and over events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    // Handle leave and drop events
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle file drop
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        updateFileName();
    }
    
    // Handle file selection
    fileInput.addEventListener('change', updateFileName);
    
    function updateFileName() {
        if (fileInput.files.length) {
            fileName.textContent = fileInput.files[0].name;
        } else {
            fileName.textContent = 'No file selected';
        }
    }
    
    // Handle form reset
    resetBtn.addEventListener('click', function() {
        form.reset();
        fileName.textContent = 'No file selected';
        document.querySelector('.result').style.display = 'none';
    });
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select a file first');
            return;
        }
        
        // Create FormData object
        const formData = new FormData(form);
        
        // Show loading state
        const resultSection = document.querySelector('.result');
        resultSection.style.display = 'flex';
        const overallPercentages = document.getElementById('overall-percentages');
        const modelMetrics = document.getElementById('model-metrics');
        const detailedPredictions = document.getElementById('detailed-predictions');
        
        overallPercentages.innerHTML = '<div>Loading...</div>';
        modelMetrics.innerHTML = '<div>Loading...</div>';
        detailedPredictions.innerHTML = '';
        
        const plotContainer = document.getElementById('plotContainer');
        plotContainer.innerHTML = '<div class="loading">Analyzing data and generating visualizations...</div>';
        
        // Send data to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Server error');
                });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            overallPercentages.innerHTML = '<div>Error processing data</div>';
            modelMetrics.innerHTML = '<div>Error processing data</div>';
            plotContainer.innerHTML = '';
        });
    });
    
    // Toggle detailed predictions visibility
    document.getElementById('togglePredictions').addEventListener('click', function() {
        const detailedPredictions = document.getElementById('detailed-predictions');
        if (detailedPredictions.style.display === 'none') {
            detailedPredictions.style.display = 'block';
        } else {
            detailedPredictions.style.display = 'none';
        }
    });
}

/**
 * Display the results from the server
 */
function displayResults(data) {
    // Show result section
    const resultSection = document.querySelector('.result');
    resultSection.style.display = 'flex';
    
    // 1. Display overall percentages
    const overallPercentages = document.getElementById('overall-percentages');
    overallPercentages.innerHTML = '';
    
    if (data.overall_percentages) {
        const sleepCard = createMetricCard(data.overall_percentages.Sleep + '%', 'Sleep');
        const restCard = createMetricCard(data.overall_percentages.Rest + '%', 'Rest');
        overallPercentages.appendChild(sleepCard);
        overallPercentages.appendChild(restCard);
    }
    
    // 2. Display model metrics
    const modelMetrics = document.getElementById('model-metrics');
    modelMetrics.innerHTML = '';
    
    if (data.model_metrics) {
        const confidenceCard = createMetricCard(data.model_metrics.Model_Confidence + '%', 'Confidence');
        modelMetrics.appendChild(confidenceCard);
    }
    
    // 3. Display detailed predictions
    const detailedPredictions = document.getElementById('detailed-predictions');
    detailedPredictions.innerHTML = '';
    
    if (data.predictions) {
        const entries = Object.entries(data.predictions);
        // Only show first 20 entries if there are too many
        const displayEntries = entries.length > 20 ? entries.slice(0, 20) : entries;
        
        displayEntries.forEach(([entry, probabilities]) => {
            const entryElement = document.createElement('div');
            entryElement.className = 'prediction-entry';
            
            const label = document.createElement('div');
            label.className = 'prediction-entry-label';
            label.textContent = entry;
            
            const probsContainer = document.createElement('div');
            probsContainer.className = 'prediction-probability';
            
            const restProb = (probabilities.Rest * 100).toFixed(2);
            const sleepProb = (probabilities.Sleep * 100).toFixed(2);
            
            probsContainer.innerHTML = `
                <div>Rest: ${restProb}%</div>
                <div>Sleep: ${sleepProb}%</div>
            `;
            
            // Create probability bar visualization
            const barContainer = document.createElement('div');
            barContainer.className = 'probability-bar';
            
            const restBar = document.createElement('div');
            restBar.className = 'rest-probability';
            restBar.style.width = `${restProb}%`;
            
            const sleepBar = document.createElement('div');
            sleepBar.className = 'sleep-probability';
            sleepBar.style.width = `${sleepProb}%`;
            
            barContainer.appendChild(restBar);
            barContainer.appendChild(sleepBar);
            
            entryElement.appendChild(label);
            entryElement.appendChild(probsContainer);
            entryElement.appendChild(barContainer);
            
            detailedPredictions.appendChild(entryElement);
        });
        
        // Add a note if showing truncated results
        if (entries.length > 20) {
            const note = document.createElement('div');
            note.textContent = `Showing first 20 of ${entries.length} entries`;
            note.style.marginTop = '10px';
            note.style.fontSize = '0.9rem';
            note.style.opacity = '0.7';
            detailedPredictions.appendChild(note);
        }
    }
    
    // Display visualizations
    const plotContainer = document.getElementById('plotContainer');
    plotContainer.innerHTML = ''; // Clear previous plots
    
    if (data.visualizations) {
        // Create the visualization elements
        const visuals = data.visualizations;
        
        // Add feature distributions
        if (visuals.feature_distributions) {
            addPlot(plotContainer, 'EEG & fMRI Feature Distributions', visuals.feature_distributions);
        }
        
        // Add confusion matrix if available
        if (visuals.confusion_matrix) {
            addPlot(plotContainer, 'Confusion Matrix', visuals.confusion_matrix);
        }
        
        // Add correlations
        if (visuals.feature_correlations) {
            addPlot(plotContainer, 'EEG-FMRI Feature Correlations', visuals.feature_correlations);
        }
        
        // Add prediction distributions
        if (visuals.prediction_distributions) {
            addPlot(plotContainer, 'Prediction Distributions', visuals.prediction_distributions);
        }
    }
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Create a metric card for displaying statistics
 */
function createMetricCard(value, label) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    
    const valueElement = document.createElement('div');
    valueElement.className = 'metric-value';
    valueElement.textContent = value;
    
    const labelElement = document.createElement('div');
    labelElement.className = 'metric-label';
    labelElement.textContent = label;
    
    card.appendChild(valueElement);
    card.appendChild(labelElement);
    
    return card;
}

/**
 * Add a plot to the container
 */
function addPlot(container, title, base64Data) {
    const plotItem = document.createElement('div');
    plotItem.className = 'plotItem';
    
    const plotTitle = document.createElement('h4');
    plotTitle.textContent = title;
    plotItem.appendChild(plotTitle);
    
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Data}`;
    img.alt = title;
    
    plotItem.appendChild(img);
    container.appendChild(plotItem);
}

// Function to open report document
function openReportDoc() {
    window.open('/static/assets/Report.pdf', '_blank');
}

// Function to open research paper
function openResearchPDF() {
    window.open('/static/assets/Research_paper.pdf', '_blank');
}
