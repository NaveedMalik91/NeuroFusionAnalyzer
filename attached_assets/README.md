# Project Structure

## Main Directory
- `Neural-Analysis/`: Root directory containing all project files

## Backend Directory
Neural-Analysis/
├── backend/
│   ├── model/
│   │   │   ├── fusion_model.h5
│   ├── static/
│   │   ├── assets/
│   │   │   ├── image.png
│   │   │   ├── image2.jpg
│   │   │   ├── logo1.png
│   │   │   ├── Report.pdf
│   │   │   ├── Research_paper.pdf
│   │   ├── css/
│   │   │   ├── style_model1.css
│   │   │   ├── style.css
│   │   ├── js/
│   │   │   ├── particles.min.js
│   │   │   ├── script.js
│   │   ├── videos/
│   │   │   ├── brainnn.mp4
│   ├── templates/
│   │   ├── index.html
│   │   ├── model1.html
│   ├── app.py
│   ├── Preprocessing.py
│   ├── requirements.txt
├── .gitignore
├── README.md



# EEG-fMRI Brain Signal Analysis and Prediction Model

This project utilizes **EEG** (electroencephalogram) and **fMRI** (functional magnetic resonance imaging) data to predict various tasks using a **CNN-RNN fusion model**. The model is trained to predict task types based on brain signals, and the entire system is deployed via a Flask API. A HTML/CSS/JavaScript frontend is used to interact with the API, where users can input EEG and fMRI features for prediction and visualize the results.

## Tasks:
Task 1: EEG-fMRI Brain Signals Analysis during Sleep and resting state
Task 2: EEG-fMRI Brain Signal Analysis with Eyes open and eyes closed and Resting with eyes closed 

## Project Overview

The system enables:
-**Multimodal brain signal analysis**: Utilizing EEG and fMRI features to predict tasks
- **Model Deployment**: The trained model is served using a Flask API.
- **User Interface**: A simple HTML/CSS/JavaScript-based UI allows users to input EEG and fMRI features and get predictions.

### Features
- **EEG** and **fMRI** data preprocessing.
- Task type prediction using a **trained CNN-RNN fusion model**.
- User-friendly web interface for predictions and visualizations.
- **Label encoding** for task types, making them interpretable and ready for predictions.
- **Data visualization** for EEG and fMRI features.

## Requirements

To run this project, make sure you have the following dependencies installed:

### Backend (Flask API & Model Inference)
- **Python 3.x**
- **TensorFlow**: For deep learning model inference.
- **Flask**: For serving the model via an API.
- **Flask-CORS**: To allow cross-origin requests.
- **scikit-learn**: For preprocessing tasks such as label encoding and scaling.
- **joblib**: For loading saved models and encoders.
- **numpy**: For numerical operations.
- **gunicorn**: For deploying Flask in production.

### Frontend (HTML, CSS, and JavaScript)
- **HTML**: For building the structure of the webpage.
- **CSS**: For styling the webpage.
- **JavaScript**: For dynamic content and API calls.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/naveedmalik914/eeg-fmri-analysis.git
   cd eeg-fmri-analysis


## Project Configuration
- `.gitignore`: Specifies files to ignore in version control
