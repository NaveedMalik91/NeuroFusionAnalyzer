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

// Global variable to store prediction data
let allPredictionEntries = [];
let currentPage = 0;
const entriesPerPage = 20;

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
        allPredictionEntries = [];
        currentPage = 0;
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
        
        // Reset pagination
        allPredictionEntries = [];
        currentPage = 0;
        
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
    
    // 3. Store all prediction entries for pagination
    if (data.predictions) {
        allPredictionEntries = Object.entries(data.predictions);
        currentPage = 0;
        
        // Display the first page of predictions
        displayPredictionPage();
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
 * Display a page of prediction entries
 */
function displayPredictionPage() {
    const detailedPredictions = document.getElementById('detailed-predictions');
    detailedPredictions.innerHTML = '';
    
    if (allPredictionEntries.length === 0) {
        detailedPredictions.innerHTML = '<div>No predictions available</div>';
        return;
    }
    
    // Add color legend for the progress bars
    addProgressBarLegend(detailedPredictions);
    
    // Calculate start and end indices for the current page
    const startIndex = currentPage * entriesPerPage;
    const endIndex = Math.min(startIndex + entriesPerPage, allPredictionEntries.length);
    
    // Get entries for the current page
    const pageEntries = allPredictionEntries.slice(startIndex, endIndex);
    
    // Display the entries
    pageEntries.forEach(([entry, probabilities]) => {
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
    
    // Add pagination controls
    addPaginationControls(detailedPredictions);
}

/**
 * Add pagination controls to navigate between pages of predictions
 */
function addPaginationControls(container) {
    const totalPages = Math.ceil(allPredictionEntries.length / entriesPerPage);
    
    if (totalPages <= 1) {
        return; // No pagination needed if only one page
    }
    
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'pagination-controls';
    paginationContainer.style.display = 'flex';
    paginationContainer.style.justifyContent = 'center';
    paginationContainer.style.gap = '15px';
    paginationContainer.style.marginTop = '20px';
    
    // Previous button
    const prevButton = document.createElement('button');
    prevButton.textContent = '← Previous';
    prevButton.style.padding = '8px 15px';
    prevButton.style.borderRadius = '5px';
    prevButton.style.border = 'none';
    prevButton.style.background = 'rgba(138, 43, 226, 0.7)';
    prevButton.style.color = 'white';
    prevButton.style.cursor = 'pointer';
    prevButton.disabled = currentPage === 0;
    if (prevButton.disabled) {
        prevButton.style.opacity = '0.5';
        prevButton.style.cursor = 'not-allowed';
    }
    
    prevButton.addEventListener('click', function() {
        if (currentPage > 0) {
            currentPage--;
            displayPredictionPage();
            container.scrollIntoView({ behavior: 'smooth' });
        }
    });
    
    // Page info
    const pageInfo = document.createElement('div');
    pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    pageInfo.style.display = 'flex';
    pageInfo.style.alignItems = 'center';
    pageInfo.style.fontWeight = '500';
    
    // Next button
    const nextButton = document.createElement('button');
    nextButton.textContent = 'Next →';
    nextButton.style.padding = '8px 15px';
    nextButton.style.borderRadius = '5px';
    nextButton.style.border = 'none';
    nextButton.style.background = 'rgba(138, 43, 226, 0.7)';
    nextButton.style.color = 'white';
    nextButton.style.cursor = 'pointer';
    nextButton.disabled = currentPage >= totalPages - 1;
    if (nextButton.disabled) {
        nextButton.style.opacity = '0.5';
        nextButton.style.cursor = 'not-allowed';
    }
    
    nextButton.addEventListener('click', function() {
        if (currentPage < totalPages - 1) {
            currentPage++;
            displayPredictionPage();
            container.scrollIntoView({ behavior: 'smooth' });
        }
    });
    
    // Add all elements to the pagination container
    paginationContainer.appendChild(prevButton);
    paginationContainer.appendChild(pageInfo);
    paginationContainer.appendChild(nextButton);
    
    // Add pagination container to the main container
    container.appendChild(paginationContainer);
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

// Add a legend for the progress bar colors
function addProgressBarLegend(container) {
    const legendContainer = document.createElement('div');
    legendContainer.className = 'progress-bar-legend';
    legendContainer.style.display = 'flex';
    legendContainer.style.alignItems = 'center';
    legendContainer.style.justifyContent = 'center';
    legendContainer.style.gap = '20px';
    legendContainer.style.marginTop = '15px';
    legendContainer.style.marginBottom = '10px';
    
    // Rest legend item
    const restLegend = document.createElement('div');
    restLegend.style.display = 'flex';
    restLegend.style.alignItems = 'center';
    restLegend.style.gap = '5px';
    
    const restColor = document.createElement('div');
    restColor.style.width = '20px';
    restColor.style.height = '20px';
    restColor.style.borderRadius = '4px';
    restColor.style.background = 'linear-gradient(to right, #3498db, #2980b9)';
    
    const restLabel = document.createElement('span');
    restLabel.textContent = 'Rest State';
    
    restLegend.appendChild(restColor);
    restLegend.appendChild(restLabel);
    
    // Sleep legend item
    const sleepLegend = document.createElement('div');
    sleepLegend.style.display = 'flex';
    sleepLegend.style.alignItems = 'center';
    sleepLegend.style.gap = '5px';
    
    const sleepColor = document.createElement('div');
    sleepColor.style.width = '20px';
    sleepColor.style.height = '20px';
    sleepColor.style.borderRadius = '4px';
    sleepColor.style.background = 'linear-gradient(to right, #e74c3c, #c0392b)';
    
    const sleepLabel = document.createElement('span');
    sleepLabel.textContent = 'Sleep State';
    
    sleepLegend.appendChild(sleepColor);
    sleepLegend.appendChild(sleepLabel);
    
    // Add both legend items to the container
    legendContainer.appendChild(restLegend);
    legendContainer.appendChild(sleepLegend);
    
    container.appendChild(legendContainer);
}
