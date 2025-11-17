# Diabetes Progression Prediction with MLflow

A machine learning project demonstrating MLflow experiment tracking, model registry, and serving using the Diabetes dataset.

## Project Overview

This project predicts diabetes disease progression using Ridge Regression and demonstrates key MLOps concepts with MLflow:
- Experiment Tracking: Log parameters, metrics, and artifacts
- Model Registry: Version and manage trained models
- Model Serving: Load and serve models for predictions

## Dataset

**Diabetes Dataset** (built-in scikit-learn):
- Samples: 442 patients
- Features: 10 physiological measurements (age, sex, BMI, blood pressure, etc.)
- Target: Quantitative measure of disease progression one year after baseline
- Task: Regression

## Technologies Used

- Python 3.12
- MLflow 3.6.0 - Experiment tracking and model registry
- scikit-learn - Machine learning library
- pandas & numpy - Data manipulation
- matplotlib - Visualization

## Project Structure
```
mlflow_lab/
├── train.py           # Training script with MLflow tracking
├── serve.py           # Model serving script
├── requirements.txt   # Project dependencies
├── README.md          # Project documentation
├── .gitignore        # Git ignore rules
├── mlruns/           # MLflow tracking data (auto-generated)
└── venv/             # Virtual environment
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone git@github.com:PVRPratyusha/mlflow_lab.git
cd mlflow_lab
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Train multiple models with different hyperparameters:
```bash
python train.py
```

**What it does:**
- Loads diabetes dataset
- Trains Ridge Regression with 3 different alpha values (0.1, 1.0, 10.0)
- Logs parameters, metrics, and artifacts to MLflow
- Registers models in MLflow Model Registry
- Creates prediction visualization plots

**Tracked Metrics:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R2 Score
- MSE (Mean Squared Error)

### View Experiments in MLflow UI

Start the MLflow UI:
```bash
mlflow ui
```

Open browser at: http://localhost:5000

Navigate through:
- Experiments: View all runs and compare metrics
- Models: See registered model versions

### Serve Models

Load and use trained models for predictions:
```bash
python serve.py
```

**What it does:**
- Loads the best model (Version 1) from MLflow Model Registry
- Gets sample patient data
- Makes predictions on disease progression
- Displays results

## Results

### Model Performance

| Model (Alpha) | RMSE  | MAE   | R2 Score |
|---------------|-------|-------|----------|
| 0.1          | 53.45 | 43.00 | 0.4609   |
| 1.0          | 55.47 | 46.14 | 0.4192   |
| 10.0         | 66.66 | 58.03 | 0.1612   |

**Best Model:** Ridge Regression with alpha=0.1 (highest R2 score)

## Key MLflow Concepts Demonstrated

### 1. Experiment Tracking
- Created experiment: diabetes_prediction
- Logged multiple runs with different hyperparameters
- Tracked parameters (alpha, train_size, n_features)
- Logged metrics (RMSE, MAE, R2)
- Saved artifacts (prediction plots)

### 2. Model Registry
- Registered model: diabetes_ridge_model
- Created 3 model versions
- Each version linked to specific training run
- Models ready for deployment

### 3. Model Serving
- Loaded models by name and version
- Made predictions on new data
- Demonstrated production-ready inference

## Example Output

### Training
```
============================================================
MLflow Diabetes Prediction - Training Pipeline
============================================================
Loading diabetes dataset...
Dataset shape: (442, 10)
Train set: 353 samples
Test set: 89 samples

Training model with alpha=0.1...
Model Performance:
RMSE: 53.45
MAE: 43.00
R2 Score: 0.4609
Run completed for alpha=0.1
```

### Serving
```
============================================================
MLflow Diabetes Prediction - Model Serving
============================================================
Loading model: models:/diabetes_ridge_model/1
Model loaded successfully!

Prediction Results:
Sample 1: Predicted Progression = 202.84
Sample 2: Predicted Progression = 74.44
Sample 3: Predicted Progression = 178.07
============================================================
```

## Learning Outcomes

Through this project, I learned:
- How to track ML experiments systematically using MLflow
- Managing multiple model versions with Model Registry
- Logging parameters, metrics, and artifacts
- Loading and serving models in production-like scenarios
- Comparing model performance across different hyperparameters

## Dependencies

See requirements.txt for full list:
- mlflow==3.6.0
- scikit-learn==1.3.2
- pandas==2.1.4
- numpy==1.26.2
- matplotlib==3.8.2
- setuptools>=65.0.0

## Contributing

This is an educational project for MLOps learning. Feel free to fork and experiment!

## License

MIT License - Feel free to use this project for learning purposes.

## Author

PVR Pratyusha
- GitHub: github.com/PVRPratyusha

---

Note: This project was created as part of MLOps coursework to demonstrate experiment tracking and model management best practices.