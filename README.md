# AI-Driven-Solutions-for-Comprehensive-Canine-Healthcare

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Architectures](#model-architectures)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Future Enhancements](#future-enhancements)
   
---

## Overview

- **Symptom Prediction**: Identifying potential diseases based on user-inputted symptoms.
- **Skin Disease Detection**: Analyzing images of canine skin conditions for diagnosis.
- **Nutrition Recommendations**: Generating personalized meal plans based on the dog’s breed, age, weight, and health conditions.
- **Conversational AI**: An interactive chatbot to provide health advice and maintain a conversational flow.

---

## Features

### Core Functionality:
1. **AI-Powered Disease Detection**:
   - Detects diseases like dermatitis, flea allergy, ringworm, and scabies using a convolutional neural network (CNN) built with TensorFlow and Keras.
2. **Nutrition Recommendation**:
   - Uses a CatBoostRegressor model to recommend optimal nutrition plans tailored to a dog’s profile.
3. **Symptom Prediction**:
   - Multi-class classification using a Voting Classifier (Random Forest + XGBoost) for disease prediction based on input symptoms.
4. **Interactive Chatbot**:
   - Intent classification using a TensorFlow-based neural network.
   - TF-IDF for text feature extraction.
   - Integrates LLMs like LlamaIndex for contextual responses.

---

## Dataset

### Sources:
1. **Canine Skin Disease Dataset**:
   - Images of skin conditions categorized into subdirectories (`dermatitis`, `flea_allergy`, etc.).
2. **Dog Nutrition Dataset**:
   - Features: Breed, age, weight, diseases, and their corresponding nutritional requirements.
3. **Symptoms Dataset**:
   - Binary symptom indicators with disease labels.

### Example Data Inputs:
#### Nutrition:
```json
{
    "Breed": "German Shepherd",
    "Age (months)": 24,
    "Weight (kg)": 25.5,
    "Disease": "obesity"
}
```
#### Symptoms:
```json
{
    "Chest Pain": 1,
    "Fever": 1,
    "Headache": 1,
    "Nausea": 0
}
```

---

## Model Architectures

![Main Architecture Diagram](1.png)

 

### Disease Detection (Skin Conditions):
- **Base Model**: Pretrained Xception (ImageNet weights).
- Dense layers with 512 and 256 neurons.
- Softmax activation for multi-class classification.

### Nutrition Recommendation:
- **Model**: CatBoostRegressor with MultiOutputRegressor.
- Trained on customized dog nutrition datasets.

### Symptom Prediction:
- Ensemble Voting Classifier:
  - Random Forest Classifier for feature relationships.
  - XGBoost for handling imbalanced datasets.

### Chatbot:
- Neural Network for intent classification.
- LLMs for generating intelligent responses.

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Libraries: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, OpenCV, and LlamaIndex.

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vetbot.git
   cd vetbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place datasets in the appropriate directories:
   - Skin disease images: `data/images/`
   - Nutrition data: `data/nutrition/dog_nutrition_dataset.csv`
   - Symptoms data: `data/symptoms/vetbot.xlsx`

4. Add your API key to `secrets.yaml`:
   ```yaml
   GROQ_API_KEY: "your_groq_api_key_here"
   ```

---

## Usage

### Train Models:
- **Skin Disease Detection**:
   ```bash
   python train.py
   ```
- **Nutrition Recommendation**:
   ```bash
   python train_model.py
   ```
- **Symptom Prediction**:
   ```bash
   python train_symptoms.py
   ```

### Start VetBot:
   ```bash
   python vetbot.py
   ```

### Predict and Infer:
- **Skin Disease**:
   ```python
   from inference import inference_disease
   result = inference_disease({"image": "path/to/image.jpg", "ears": "normal"})
   ```
- **Nutrition**:
   ```python
   nutrition = inference_nutrition({"Breed": "Labrador", "Age": 24, "Weight": 25.5})
   ```
- **Symptoms**:
   ```python
   predicted_disease = predict_symptom({"Fever": 1, "Headache": 1})
   ```

---

## Evaluation Metrics

- Accuracy, Precision, Recall, AUC (for disease detection).
- MAE, MSE, R² (for nutrition prediction).
- Confusion Matrix and Classification Report (for symptoms prediction).

---

## Results

- **Skin Disease Detection**: Training and validation loss trends available.
- **Nutrition Recommendation**: MAE: XX%, R²: XX%.
- **Symptom Prediction**: Confusion matrix and heatmap visualization.

---

## Future Enhancements

- Add multilingual support and voice recognition.
- Expand datasets for better generalization.
- Develop a web or mobile application for real-time predictions.

---
