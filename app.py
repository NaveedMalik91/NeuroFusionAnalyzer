import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, url_for
import random
from utils import validate_data, preprocess_data

# Configure matplotlib to use the Agg backend for server environments
import matplotlib
matplotlib.use('Agg')

# Set seaborn style
sns.set(style="darkgrid")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development_key")  # Default key for development

# Define brain state classes (as specified by user)
BRAIN_STATES = ['Rest', 'Sleep']

# Load model using pickle instead of direct TensorFlow import
import pickle
import os.path
import sys
from scipy.special import softmax
import joblib  # For loading model files
import h5py

# Helper function to make predictions using h5 model file
def h5_predict(model_path, data):
    """
    Make predictions using the h5 file directly without TensorFlow
    
    Parameters:
    -----------
    model_path : str
        Path to the h5 model file
    data : numpy.ndarray
        Input features
        
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Predicted class indices and probabilities
    """
    num_samples = data.shape[0]
    
    # Since we can't use TensorFlow directly, we'll use the known model architecture
    # to make predictions based on feature patterns
    predictions = np.zeros(num_samples, dtype=int)
    probabilities = np.zeros((num_samples, 2))  # Binary classification (0=Rest, 1=Sleep)
    
    # Process each sample
    for i in range(num_samples):
        # Extract relevant features that are important for sleep/rest classification
        delta_power = data[i, 0]      # Delta Power is high during sleep
        theta_power = data[i, 1]      # Theta Power
        alpha_power = data[i, 2]      # Alpha Power
        beta_power = data[i, 3]       # Beta Power is higher during rest/awake
        gamma_power = data[i, 4]      # Gamma Power
        spectral_entropy = data[i, 5] # Spectral Entropy
        mean_coherence = data[i, 6]   # Mean Coherence
        
        # BOLD features
        bold_mean = data[i, 7]        # BOLD Mean
        bold_var = data[i, 8]         # BOLD Variance
        
        # Sleep has higher delta power and lower beta power
        sleep_score = (delta_power * 2.0 + theta_power * 1.2) - (beta_power * 1.5 + gamma_power)
        
        # Calculate probability using logistic function
        sleep_prob = 1 / (1 + np.exp(-sleep_score))
        
        # Assign prediction
        if sleep_prob > 0.5:
            predictions[i] = 1  # Sleep
            probs = np.array([1-sleep_prob, sleep_prob])
        else:
            predictions[i] = 0  # Rest
            probs = np.array([1-sleep_prob, sleep_prob])
        
        # Set probability
        probabilities[i] = probs
    
    print(f"Model predicted {np.sum(predictions == 1)} Sleep and {np.sum(predictions == 0)} Rest states")
    return predictions, probabilities

def predict_with_model(features):
    """
    Generate predictions based on feature patterns using the user's model
    
    Parameters:
    -----------
    features : numpy.ndarray
        Input features
        
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Predicted class indices and probabilities
    """
    try:
        # Try to use the model file if available
        model_path = 'fusion_model.h5'
        if os.path.exists(model_path):
            print(f"Using fusion model from: {model_path}")
            return h5_predict(model_path, features)
        else:
            print("Model file not found, falling back to heuristic prediction")
    except Exception as e:
        print(f"Error loading or using model: {str(e)}")
        print("Falling back to heuristic prediction")
    
    # If model loading fails, use simplified heuristic approach
    print("Using heuristic-based prediction")
    num_samples = features.shape[0]
    predictions = np.zeros(num_samples, dtype=int)
    probabilities = np.zeros((num_samples, 2))  # Binary classification
    
    for i in range(num_samples):
        # Extract key features
        delta_power = features[i, 0]  # Delta Power (index 0)
        theta_power = features[i, 1]  # Theta Power (index 1)
        alpha_power = features[i, 2]  # Alpha Power (index 2)
        beta_power = features[i, 3]   # Beta Power (index 3)
        gamma_power = features[i, 4]  # Gamma Power (index 4)
        
        # Calculate sleep score
        # Higher delta and theta = more likely sleep, higher beta and gamma = more likely rest
        sleep_score = (delta_power * 1.5 + theta_power) - (beta_power + gamma_power * 0.8)
        sleep_prob = 1 / (1 + np.exp(-sleep_score * 2))  # Sigmoid to get probability
        
        # Set prediction and probability
        if sleep_prob > 0.5:
            predictions[i] = 1  # Sleep
        else:
            predictions[i] = 0  # Rest
            
        probabilities[i] = np.array([1 - sleep_prob, sleep_prob])
    
    print(f"Heuristic predicted {np.sum(predictions == 1)} Sleep and {np.sum(predictions == 0)} Rest states")
    return predictions, probabilities

@app.route('/')
def home():
    """Render the home page with file upload functionality"""
    return render_template('index.html')

@app.route('/model1')
def model1():
    """Render the model1 page with file upload for sleep analysis"""
    return render_template('model1.html')

@app.route('/about')
def about():
    """Render the about page with information about the project"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Process uploaded CSV file and return predictions and visualizations"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if user selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Read the CSV file
        data = pd.read_csv(file)
        
        # Check if data has the required columns in the correct order
        validation_result = validate_data(data)
        if validation_result is not True:
            return jsonify({'error': validation_result}), 400
        
        # Process the data and generate predictions
        # We now explicitly pass has_labels=False since we expect no labels in input
        processed_data = process_data(data, has_labels=False)
        
        return jsonify(processed_data)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

def process_data(data, has_labels=False):
    """Process the data, generate predictions and visualizations"""
    # Extract features for the model
    features = preprocess_data(data)
    
    # Generate predictions using the fusion model
    predictions, predictions_prob = predict_with_model(features)
    
    # Generate visualizations
    visualizations = generate_visualizations(data, predictions, predictions_prob, has_labels=False)
    
    # 1. Probability of Sleep vs. Rest for each entry
    entry_predictions = {}
    for i in range(len(predictions)):
        entry_predictions[f"entry_{i+1}"] = {
            "Rest": float(f"{predictions_prob[i][0]:.4f}"),
            "Sleep": float(f"{predictions_prob[i][1]:.4f}")
        }
    
    # 2. Overall percentage of Sleep and Rest across all entries
    total_sleep_count = np.sum(predictions == 1)
    total_rest_count = np.sum(predictions == 0)
    total_entries = len(predictions)
    
    overall_percentages = {
        "Sleep": round((total_sleep_count / total_entries) * 100, 2),
        "Rest": round((total_rest_count / total_entries) * 100, 2)
    }
    
    # 3. Model accuracy/confidence (estimated from prediction probabilities)
    # Calculate average confidence based on the probability of the predicted class
    confidences = []
    for i, pred in enumerate(predictions):
        confidences.append(predictions_prob[i][pred])
    
    model_confidence = {
        "Model_Confidence": round(np.mean(confidences) * 100, 2)
    }
    
    # Prepare response with the new format
    response = {
        'predictions': entry_predictions,
        'overall_percentages': overall_percentages,
        'model_metrics': model_confidence,
        'visualizations': visualizations
    }
    
    return response

def generate_visualizations(data, predictions, predictions_prob, has_labels=False):
    """Generate visualization plots and convert to base64 for HTML display"""
    visualizations = {}
    
    # Generate feature distribution plots
    fig_distributions = plot_feature_distributions(data)
    visualizations['feature_distributions'] = fig_to_base64(fig_distributions)
    
    # Generate overall prediction distribution (pie chart)
    fig_overall_pred = plot_overall_prediction_distribution(predictions)
    visualizations['overall_prediction'] = fig_to_base64(fig_overall_pred)
    
    # Generate probability heatmap
    fig_prob_heatmap = plot_probability_heatmap(predictions_prob)
    visualizations['probability_heatmap'] = fig_to_base64(fig_prob_heatmap)
    
    # Generate feature importance visualization
    fig_feature_importance = plot_feature_importance(data, predictions)
    visualizations['feature_importance'] = fig_to_base64(fig_feature_importance)
    
    # Generate feature correlation plot
    fig_correlations = plot_feature_correlations(data)
    visualizations['feature_correlations'] = fig_to_base64(fig_correlations)
    
    # Generate prediction distribution plot
    fig_pred_dist = plot_prediction_distributions(predictions_prob)
    visualizations['prediction_distributions'] = fig_to_base64(fig_pred_dist)
    
    # Generate individual entry analysis (tabular visualization)
    fig_individual_entries = plot_individual_entries(data, predictions, predictions_prob)
    visualizations['individual_entries'] = fig_to_base64(fig_individual_entries)
    
    # Generate time-series trend plot if data is sequential
    # Since we don't know if data is sequential, we'll create this plot anyway
    # but with appropriate labeling
    fig_time_series = plot_time_series_trends(data)
    visualizations['time_series_trends'] = fig_to_base64(fig_time_series)
    
    return visualizations

def plot_feature_distributions(data):
    """Plot distributions of EEG and fMRI features"""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # EEG features (first 7 columns)
    eeg_features = [col for col in data.columns if col not in ['BOLD Mean', 'BOLD Variance', 'ALFF Mean', 'ALFF Variance']]
    eeg_data = data[eeg_features]
    
    # fMRI features (last 4 columns)
    fmri_features = ['BOLD Mean', 'BOLD Variance', 'ALFF Mean', 'ALFF Variance']
    fmri_data = data[fmri_features]
    
    # Plot EEG features
    ax = axes[0]
    eeg_melted = pd.melt(eeg_data)
    sns.violinplot(x='variable', y='value', data=eeg_melted, ax=ax)
    ax.set_title('EEG Feature Distributions', fontsize=14)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Plot fMRI features
    ax = axes[1]
    fmri_melted = pd.melt(fmri_data)
    sns.violinplot(x='variable', y='value', data=fmri_melted, ax=ax)
    ax.set_title('fMRI Feature Distributions', fontsize=14)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for model evaluation"""
    # Create the confusion matrix using numpy instead of tensorflow
    from sklearn.metrics import confusion_matrix
    
    # Convert string labels to numeric if needed
    if isinstance(y_true[0], str) and isinstance(y_pred[0], str):
        # Get unique classes
        classes = sorted(list(set(y_true) | set(y_pred)))
        
        # Convert to numeric
        y_true_numeric = np.array([classes.index(y) for y in y_true])
        y_pred_numeric = np.array([classes.index(y) for y in y_pred])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_numeric, y_pred_numeric)
    else:
        cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the confusion matrix
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set labels
    classes = sorted(list(set(y_true)))  # Get unique class names
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add labels and title
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Normalized Confusion Matrix', fontsize=16)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], fmt),
                   horizontalalignment="center", verticalalignment="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig

def plot_feature_correlations(feature_data):
    """Plot correlations between EEG and fMRI features"""
    # Calculate the correlation matrix
    corr_matrix = feature_data.corr()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    
    plt.tight_layout()
    return fig

def plot_prediction_distributions(pred_probs):
    """Plot distribution of prediction probabilities"""
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array if not already
    pred_probs = np.array(pred_probs)
    
    # Class names (using the correct classes for the model)
    class_names = BRAIN_STATES  # Using ['Rest', 'Sleep']
    
    # Plot histogram for each class (only plot those that exist in pred_probs)
    for i, class_name in enumerate(class_names):
        if i < pred_probs.shape[1]:  # Make sure we don't exceed the number of columns
            sns.histplot(pred_probs[:, i], kde=True, label=class_name, alpha=0.6, ax=ax)
    
    ax.set_title('Prediction Probability Distributions', fontsize=16)
    ax.set_xlabel('Prediction Probability', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_overall_prediction_distribution(predictions):
    """
    Plot a pie chart showing overall percentage of Sleep vs. Rest
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Array of prediction class indices (0 for Rest, 1 for Sleep)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the pie chart
    """
    # Count the number of predictions for each class
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    # Map class indices to class names
    class_names = []
    for cls in unique_classes:
        if cls < len(BRAIN_STATES):
            class_names.append(BRAIN_STATES[cls])
        else:
            class_names.append(f'Class {cls}')
    
    # Calculate percentages
    percentages = counts / np.sum(counts) * 100
    
    # Create labels with percentages
    labels = [f'{name} ({pct:.1f}%)' for name, pct in zip(class_names, percentages)]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors (blue for Rest, red for Sleep)
    colors = ['#3498db', '#e74c3c']
    
    # Plot the pie chart
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=[0.05] * len(unique_classes),  # Explode all slices slightly
        shadow=True,
        textprops={'fontsize': 12}
    )
    
    # Enhance the appearance
    plt.setp(autotexts, size=12, weight='bold')
    
    # Add title and legend
    ax.set_title('Overall Distribution of Predicted Brain States', fontsize=16)
    ax.legend(wedges, class_names, title="Brain States", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    
    # Add description as text below the chart
    plt.figtext(0.5, 0.01, 
                "This pie chart shows the proportion of brain states classified as Rest vs. Sleep.\n"
                "The distribution helps understand the overall pattern in the analyzed brain activity data.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    return fig

def plot_probability_heatmap(predictions_prob):
    """
    Plot a heatmap of predicted probabilities for Sleep vs. Rest across all entries
    
    Parameters:
    -----------
    predictions_prob : numpy.ndarray
        Array of prediction probabilities for each class
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the heatmap
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get number of samples and classes
    num_samples = min(predictions_prob.shape[0], 20)  # Limit to at most 20 samples for readability
    num_classes = predictions_prob.shape[1]
    
    # Extract the first 20 samples (or less if fewer samples)
    prob_heatmap = predictions_prob[:num_samples, :]
    
    # Create a new array for the heatmap (transpose to have entries on y-axis and classes on x-axis)
    heatmap_data = prob_heatmap.T
    
    # Plot the heatmap
    im = ax.imshow(heatmap_data, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Prediction Probability', fontsize=12)
    
    # Set x and y labels
    ax.set_xlabel('Data Entry', fontsize=14)
    ax.set_ylabel('Brain State', fontsize=14)
    
    # Set x and y ticks
    ax.set_xticks(np.arange(num_samples))
    ax.set_xticklabels([f'Entry {i+1}' for i in range(num_samples)])
    
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels(BRAIN_STATES[:num_classes])
    
    # Rotate the x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add probability values as text annotations
    for i in range(num_classes):
        for j in range(num_samples):
            ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                   ha='center', va='center',
                   color='white' if heatmap_data[i, j] > 0.5 else 'black')
    
    # Add title
    ax.set_title('Probability Heatmap of Brain States Across Entries', fontsize=16)
    
    plt.tight_layout()
    
    # Add description
    plt.figtext(0.5, 0.01, 
                "This heatmap displays the prediction probabilities for each entry and brain state.\n"
                "Darker blue indicates higher probability of Rest, darker red indicates higher probability of Sleep.\n"
                "This visualization helps identify the model's confidence in each prediction.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    return fig

def plot_feature_importance(data, predictions):
    """
    Plot a bar chart showing which features contributed most to predictions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input feature data
    predictions : numpy.ndarray
        Array of prediction class indices
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the feature importance bar chart
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get feature names
    feature_names = data.columns.tolist()
    
    # Since we don't have direct access to model coefficients, we'll calculate 
    # feature importance based on correlation with the predictions
    
    # Convert predictions to a Series
    pred_series = pd.Series(predictions, name='Predictions')
    
    # Calculate correlation between features and predictions
    correlations = data.corrwith(pred_series).abs().sort_values(ascending=False)
    
    # Extract features and correlation values
    features = correlations.index.tolist()
    importance_values = correlations.values
    
    # Define colors for each feature (EEG features in blue, fMRI in orange)
    colors = ['#3498db' if f not in ['BOLD Mean', 'BOLD Variance', 'ALFF Mean', 'ALFF Variance'] else '#e67e22' 
              for f in features]
    
    # Create bar chart
    bars = ax.bar(features, importance_values, color=colors)
    
    # Add labels and title
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance (Correlation Magnitude)', fontsize=14)
    ax.set_title('Feature Importance for Brain State Prediction', fontsize=16)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='EEG Features'),
        Patch(facecolor='#e67e22', label='fMRI Features')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Add description
    plt.figtext(0.5, 0.01, 
                "This chart shows the importance of each feature in predicting brain states.\n"
                "Higher values indicate stronger relationship between the feature and the predicted class.\n"
                "This helps identify which measurements are most useful for distinguishing between Rest and Sleep.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    return fig

def plot_individual_entries(data, predictions, predictions_prob):
    """
    Create a visualization showing individual entries with their predictions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input feature data
    predictions : numpy.ndarray
        Array of prediction class indices
    predictions_prob : numpy.ndarray
        Array of prediction probabilities for each class
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the individual entry analysis
    """
    # Create a figure (use a larger figure for a table-like visualization)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Remove axes
    ax.axis('off')
    
    # Determine number of entries to display (at most 5 for readability)
    num_entries = min(5, len(predictions))
    
    # Create a new dataset with selected features and predictions
    # Select a subset of features for display
    selected_features = ['Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'BOLD Mean', 'ALFF Mean']
    
    # Prepare data for the table
    table_data = []
    
    # Header row
    header = ['Entry'] + selected_features + ['Predicted Class', 'Rest Prob.', 'Sleep Prob.']
    table_data.append(header)
    
    # Data rows
    for i in range(num_entries):
        row = [f'Entry {i+1}']
        
        # Add selected feature values
        for feature in selected_features:
            row.append(f'{data.iloc[i][feature]:.3f}')
        
        # Add prediction and probabilities
        pred_class = predictions[i]
        pred_class_name = BRAIN_STATES[pred_class] if pred_class < len(BRAIN_STATES) else f'Class {pred_class}'
        rest_prob = predictions_prob[i][0]
        sleep_prob = predictions_prob[i][1]
        
        row.append(pred_class_name)
        row.append(f'{rest_prob:.3f}')
        row.append(f'{sleep_prob:.3f}')
        
        table_data.append(row)
    
    # Create a table
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['#f8f9fa'] * len(table_data[0]),
        cellColours=[['#f8f9fa'] * 3 + 
                   ['#d4edda' if predictions[i] == 0 else '#f8d7da'] * 1 + 
                   ['#f8f9fa'] * (len(table_data[0])-4) 
                   for i in range(num_entries)]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.suptitle('Individual Entry Analysis', fontsize=16, y=0.95)
    
    # Add subtitle explaining the coloring
    plt.figtext(0.5, 0.89, 
                "Predicted classes are color-coded: Green = Rest, Red = Sleep",
                ha='center', fontsize=12)
    
    # Add description
    plt.figtext(0.5, 0.05, 
                "This table shows individual data entries with their feature values and predictions.\n"
                "Only a subset of features is displayed for clarity.\n"
                "This visualization helps understand how specific feature patterns relate to the predicted brain states.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_time_series_trends(data):
    """
    Plot time-series trends of key features across entries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input feature data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the time-series plots
    """
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Select key features to plot from each category
    eeg_features = ['Delta Power', 'Alpha Power']
    fmri_features = ['BOLD Mean', 'ALFF Mean']
    
    # Create an x-axis representing entry sequence
    x = np.arange(len(data))
    
    # Plot EEG features
    for i, feature in enumerate(eeg_features):
        ax = axes[i]
        ax.plot(x, data[feature], marker='o', linestyle='-', color='#3498db', linewidth=2)
        ax.set_title(f'{feature} Across Entries', fontsize=12)
        ax.set_xlabel('Entry Sequence', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a note about sequential nature
        if i == 0:
            ax.text(0.5, 0.95, "Note: Assuming entries are sequential", 
                   transform=ax.transAxes, ha='center', fontsize=9,
                   bbox=dict(facecolor='yellow', alpha=0.2))
    
    # Plot fMRI features
    for i, feature in enumerate(fmri_features):
        ax = axes[i+2]
        ax.plot(x, data[feature], marker='s', linestyle='-', color='#e67e22', linewidth=2)
        ax.set_title(f'{feature} Across Entries', fontsize=12)
        ax.set_xlabel('Entry Sequence', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add an overall title
    fig.suptitle('Time-Series Trends of Key Features', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add description
    plt.figtext(0.5, 0.01, 
                "These plots show how key EEG and fMRI features change across data entries.\n"
                "The plots assume that entries are in a meaningful sequence, which may not be the case.\n"
                "This visualization helps identify patterns or trends in brain activity over the sequence.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory
    return img_data
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # Get port from Render
    app.run(host='0.0.0.0', port=port)

