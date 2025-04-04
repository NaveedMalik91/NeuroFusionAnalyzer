"""
Utility functions for EEG-fMRI data analysis
This module contains helper functions for data validation and preprocessing.
"""

import numpy as np
import pandas as pd

# Define expected feature columns
EXPECTED_COLUMNS = [
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power", 
    "Spectral Entropy", "Mean Coherence", "BOLD Mean", "BOLD Variance", 
    "ALFF Mean", "ALFF Variance"
]

# Define brain state classes (as specified by user)
BRAIN_STATES = ['Rest', 'Sleep']

def validate_data(data):
    """
    Validate that the uploaded data contains exactly the required columns in the correct order
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The uploaded data to validate
        
    Returns:
    --------
    bool or str
        True if valid, error message string if invalid
    """
    # Check if the dataframe is empty
    if data.empty:
        return "The uploaded file contains no data"
    
    # Check that columns match exactly the expected columns (including order)
    if list(data.columns) != EXPECTED_COLUMNS:
        return "Invalid data format. Data must contain exactly these columns in this order: " + ", ".join(EXPECTED_COLUMNS)
    
    # Check if numeric columns contain non-numeric data
    for col in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[col]):
            return f"Column '{col}' contains non-numeric data"
    
    # Ensure no extra columns are present
    if len(data.columns) != len(EXPECTED_COLUMNS):
        extra_columns = [col for col in data.columns if col not in EXPECTED_COLUMNS]
        return f"Extra columns not allowed: {', '.join(extra_columns)}"
    
    return True

def preprocess_data(data):
    """
    Extract features from the data for model input
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to preprocess
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed features ready for model input
    """
    # Extract the required features
    features = data[EXPECTED_COLUMNS].values
    
    # We don't do additional preprocessing as the fusion model expects raw features
    # If needed, scaling or normalization could be added here
    
    return features