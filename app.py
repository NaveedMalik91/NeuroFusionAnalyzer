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

# Define brain state classes
BRAIN_STATES = ['Rest', 'Eyes Open', 'Eyes Closed']

# Simplified prediction function (mock predictions without TensorFlow)
def mock_predict(features):
    """
    Generate mock predictions for demonstration purposes
    
    Parameters:
    -----------
    features : numpy.ndarray
        Input features
        
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Predicted class indices and probabilities
    """
    num_samples = features.shape[0]
    
    # Generate mock predictions
    predictions = np.array([random.randint(0, 2) for _ in range(num_samples)])
    
    # Generate mock probabilities
    probabilities = np.zeros((num_samples, 3))
    for i in range(num_samples):
        # Create a random probability distribution that sums to 1
        probs = np.random.random(3)
        # Make the predicted class have a higher probability
        probs[predictions[i]] += 1.0
        # Normalize to sum to 1
        probabilities[i] = probs / probs.sum()
    
    return predictions, probabilities

@app.route('/')
def home():
    """Render the home page with file upload functionality"""
    return render_template('index.html')

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
        
        # Check if data has the required columns
        validation_result = validate_data(data)
        if validation_result is not True:
            return jsonify({'error': validation_result}), 400
        
        # Check if the State column exists (for labeled data)
        has_labels = 'State' in data.columns
        
        # Process the data and generate predictions
        processed_data = process_data(data, has_labels)
        
        return jsonify(processed_data)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

def process_data(data, has_labels=True):
    """Process the data, generate predictions and visualizations"""
    # Extract features for the model
    features = preprocess_data(data)
    
    # Generate mock predictions (since we're not using TensorFlow)
    predictions, predictions_prob = mock_predict(features)
    
    # Generate visualizations
    visualizations = generate_visualizations(data, predictions, predictions_prob, has_labels)
    
    # Prepare response
    response = {
        'predictions': predictions.tolist(),
        'visualizations': visualizations
    }
    
    # Include true labels in response if we have them
    if has_labels:
        # Map prediction indices to class names
        class_names = BRAIN_STATES
        predictions_str = [class_names[p] for p in predictions]
        
        # Include true labels in response
        response['true_labels'] = data['State'].tolist()
    
    return response

def generate_visualizations(data, predictions, predictions_prob, has_labels=True):
    """Generate visualization plots and convert to base64 for HTML display"""
    visualizations = {}
    
    # Generate feature distribution plots
    fig_distributions = plot_feature_distributions(data)
    visualizations['feature_distributions'] = fig_to_base64(fig_distributions)
    
    # Generate confusion matrix if we have ground truth labels
    if has_labels:
        # Get true labels
        y_true = data['State'].values
        
        # Map prediction indices to class names
        class_names = ['Rest', 'Eyes Open', 'Eyes Closed']
        predictions_str = [class_names[p] for p in predictions]
        
        # Generate confusion matrix
        fig_confusion = plot_confusion_matrix(y_true, predictions_str)
        visualizations['confusion_matrix'] = fig_to_base64(fig_confusion)
    
    # Generate feature correlation plot
    fig_correlations = plot_feature_correlations(data.drop(columns=['State'] if has_labels else []))
    visualizations['feature_correlations'] = fig_to_base64(fig_correlations)
    
    # Generate prediction distribution plot
    fig_pred_dist = plot_prediction_distributions(predictions_prob)
    visualizations['prediction_distributions'] = fig_to_base64(fig_pred_dist)
    
    return visualizations

def plot_feature_distributions(data):
    """Plot distributions of EEG and fMRI features"""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # EEG features (first 7 columns)
    eeg_features = [col for col in data.columns if col not in ['State', 'BOLD Mean', 'BOLD Variance', 'ALFF Mean', 'ALFF Variance']]
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
    
    # Class names
    class_names = ['Rest', 'Eyes Open', 'Eyes Closed']
    
    # Plot histogram for each class
    for i, class_name in enumerate(class_names):
        sns.histplot(pred_probs[:, i], kde=True, label=class_name, alpha=0.6, ax=ax)
    
    ax.set_title('Prediction Probability Distributions', fontsize=16)
    ax.set_xlabel('Prediction Probability', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
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
    app.run(host='0.0.0.0', port=5000, debug=True)