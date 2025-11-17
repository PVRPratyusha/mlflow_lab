import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set experiment name
mlflow.set_experiment("diabetes_prediction")


def load_data():
    """Load and prepare diabetes dataset"""
    print("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='progression')

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {diabetes.feature_names}")

    return X, y


def train_model(X_train, y_train, alpha=1.0):
    """Train Ridge regression model"""
    print(f"\nTraining model with alpha={alpha}...")
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, y_pred


def plot_predictions(y_test, y_pred):
    """Create prediction vs actual plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Progression')
    plt.ylabel('Predicted Progression')
    plt.title('Diabetes Progression: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('predictions_plot.png')
    plt.close()
    return 'predictions_plot.png'


def main():
    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Try different alpha values
    alphas = [0.1, 1.0, 10.0]

    for alpha in alphas:
        # Start MLflow run
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
            # Log parameters
            mlflow.log_param("model_type", "Ridge Regression")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Train model
            model = train_model(X_train, y_train, alpha=alpha)

            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_test, y_test)

            # Log metrics
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("r2_score", metrics["r2"])
            mlflow.log_metric("mse", metrics["mse"])

            # Create and log plot
            plot_file = plot_predictions(y_test, y_pred)
            mlflow.log_artifact(plot_file)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"diabetes_ridge_model"
            )

            print(f"\n✅ Run completed for alpha={alpha}")
            print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Diabetes Prediction - Training Pipeline")
    print("=" * 60)
    main()
    print("\n" + "=" * 60)
    print("Training completed! Check MLflow UI at http://localhost:5000")
    print("=" * 60)