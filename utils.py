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

# Define brain state classes
BRAIN_STATES = ['Rest', 'Eyes Open', 'Eyes Closed']

def validate_data(data):
    """
    Validate that the uploaded data contains all required columns
    
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
    
    # Check for required columns
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in data.columns]
    if missing_columns:
        return f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if numeric columns contain non-numeric data
    for col in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[col]):
            return f"Column '{col}' contains non-numeric data"
    
    # Check if the optional 'State' column contains valid values
    if 'State' in data.columns:
        valid_states = set(BRAIN_STATES)
        unique_states = set(data['State'].unique())
        invalid_states = unique_states - valid_states
        
        if invalid_states:
            return f"Invalid state values found: {', '.join(invalid_states)}. Valid states are: {', '.join(valid_states)}"
    
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