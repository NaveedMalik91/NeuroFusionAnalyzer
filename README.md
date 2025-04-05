# EEG-fMRI Brain State Analysis Web Application

This web application provides advanced analysis of EEG and fMRI data, using a machine learning fusion model to classify brain states as either "Rest" or "Sleep" based on specific neurological features.

## Overview

The application uses a pre-trained neural network model (fusion_model.h5) to analyze EEG and fMRI data features and generate comprehensive visualizations and predictions about brain states.

## Features

- **Real-time Data Analysis**: Upload CSV files containing EEG and fMRI features for immediate analysis
- **Brain State Classification**: Accurately classifies brain activity as either "Rest" or "Sleep" states
- **Interactive Visualizations**: Generate multiple visualizations to aid in understanding brain activity:
  - EEG & fMRI Feature Distributions
  - Overall Prediction Distribution (Pie Chart)
  - Prediction Probability Heatmap
  - Feature Importance Analysis
  - Feature Correlation Heatmap
  - Prediction Probability Distributions
  - Individual Entry Analysis
  - Time-Series Trends
  
## Usage

1. Upload a CSV file containing exactly 11 columns in this specific order:
   - Delta Power
   - Theta Power
   - Alpha Power
   - Beta Power
   - Gamma Power
   - Spectral Entropy
   - Mean Coherence
   - BOLD Mean
   - BOLD Variance
   - ALFF Mean
   - ALFF Variance

2. The application will validate your input data and process it using the fusion model
3. View the analysis results and visualizations directly in your browser

## Technical Details

- **Framework**: Flask web application
- **Model**: Neural network fusion model (fusion_model.h5)
- **Visualization**: Matplotlib and Seaborn for data visualization
- **Frontend**: HTML, CSS, JavaScript with particles.js for interactive background

## Deployment

This application is configured for deployment on Render with the following specifications:
- Python 3.11+
- Gunicorn web server
- Flask web framework

## Sample Data Format

The application expects a CSV file with the following column structure (example with 3 rows). The data should be scaled with Standard Scaler.

```
Delta Power  Theta  Power Alpha Power Beta Power  Gamma Power  Spectral Entropy  Mean Coherence  BOLD Mean  BOLD Variance  ALFF Mean  ALFF Variance
0.62           0.41        0.35     0.28              0.19          0.88            0.72            0.45         0.21         0.67       0.33


## Error Handling

The application includes comprehensive error handling:
- Validates input CSV format
- Ensures exactly 11 required columns in the correct order
- Reports specific errors with clear messages
- Provides fallback prediction methods if model loading fails

## License

All rights reserved.
