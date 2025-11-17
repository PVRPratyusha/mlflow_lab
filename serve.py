import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes


def load_model(model_name="diabetes_ridge_model", version=1):
    """Load model from MLflow Model Registry"""
    model_uri = f"models:/{model_name}/{version}"
    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print(" Model loaded successfully!")
    return model


def get_sample_data():
    """Get sample data for prediction"""
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    return X.head(5)  # Return first 5 samples


def make_predictions(model, data):
    """Make predictions using the loaded model"""
    predictions = model.predict(data)
    return predictions


def main():
    print("=" * 60)
    print("MLflow Diabetes Prediction - Model Serving")
    print("=" * 60)

    # Load the best model (version 1 - alpha=0.1 had best R2)
    model = load_model(model_name="diabetes_ridge_model", version=1)

    # Get sample data
    print("\nGetting sample data...")
    sample_data = get_sample_data()
    print(f"\nSample data shape: {sample_data.shape}")
    print("\nFirst few samples:")
    print(sample_data)

    # Make predictions
    print("\nMaking predictions...")
    predictions = make_predictions(model, sample_data)

    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results:")
    print("=" * 60)
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: Predicted Progression = {pred:.2f}")

    print("\n" + "=" * 60)
    print("Serving completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()