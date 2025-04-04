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
        const predictionText = document.getElementById('predictionText');
        predictionText.textContent = 'Processing...';
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
            predictionText.textContent = 'Error: ' + (error.message || 'An unexpected error occurred');
            plotContainer.innerHTML = '';
        });
    });
}

/**
 * Display the results from the server
 */
function displayResults(data) {
    // Show result section
    const resultSection = document.querySelector('.result');
    resultSection.style.display = 'flex';
    
    // Show prediction
    const predictionText = document.getElementById('predictionText');
    
    // Process predictions
    if (data.predictions && data.predictions.length) {
        // Map prediction indices to class names (0->Rest, 1->Sleep)
        const classNames = ['Rest', 'Sleep'];
        const predictions = data.predictions.map(p => classNames[p]);
        
        // Count occurrences of each prediction
        const counts = {};
        predictions.forEach(p => { counts[p] = (counts[p] || 0) + 1; });
        
        // Find the most common prediction
        const mostCommon = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
        const percentage = Math.round((counts[mostCommon] / predictions.length) * 100);
        
        predictionText.textContent = `Predicted Brain State: ${mostCommon} (${percentage}% confidence)`;
    } else {
        predictionText.textContent = 'No predictions available';
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
