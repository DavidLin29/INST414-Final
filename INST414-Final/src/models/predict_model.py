# src/models/predict_model.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.train_model import load_and_preprocess_data  # Fixed import path

def load_trained_model(model_filepath):
    """Load the trained model from a file."""
    return joblib.load(model_filepath)

def preprocess_input_data(filepath):
    """Preprocess input data in the same way as during the training phase."""
    # Use the same preprocessing function from train_model
    X, _, df = load_and_preprocess_data(filepath)
    return X, df

def make_predictions(model, data_filepath):
    """Make predictions using the trained model."""
    # Preprocess the input data (ensure consistency with the training set)
    X, df = preprocess_input_data(data_filepath)
    
    # Make predictions using the model
    predictions = model.predict(X)
    
    return predictions, df

def save_predictions(predictions, df, output_filepath):
    """Save the predictions to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    df["Predicted_Conversion"] = predictions
    df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to {output_filepath}")
    
def visualize_predictions(df):
    """Visualize some of the predictions."""
    # Visualize the predictions, e.g., distribution of predicted values
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Predicted_Conversion", data=df, palette="coolwarm")
    plt.xlabel("Predicted Conversion")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Conversions")
    plt.show()
    
    # Visualize the relationship between age category and predictions
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Age Category", hue="Predicted_Conversion", data=df, palette="viridis")
    plt.xlabel("Age Category")
    plt.ylabel("Count")
    plt.title("Predicted Conversions by Age Category")
    plt.xticks(rotation=45)
    plt.legend(title="Conversion")
    plt.tight_layout()
    plt.show()

def main(model_filepath, data_filepath, output_filepath):
    """Main function to load model, make predictions, and save them."""
    # Load the trained model
    model = load_trained_model(model_filepath)
    
    # Make predictions
    predictions, df = make_predictions(model, data_filepath)
    
    # Save predictions to file
    save_predictions(predictions, df, output_filepath)
    
    # Optional: Visualize predictions
    visualize_predictions(df)

if __name__ == "__main__":
    # Try different path formats
    base_paths = ["INST414-Final", ""]
    
    model_filepath = None
    data_filepath = None
    
    for base in base_paths:
        model_path = os.path.join(base, "models", "trained_model_rf.pkl")
        data_path = os.path.join(base, "data", "processed", "feature_data.csv")
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            model_filepath = model_path
            data_filepath = data_path
            break
    
    if model_filepath is None or data_filepath is None:
        raise FileNotFoundError("Could not find model or data files. Please run train_model.py first.")
    
    # Determine output path based on data path
    if "INST414-Final" in data_filepath:
        output_filepath = "INST414-Final/data/output/predictions.csv"
    else:
        output_filepath = "data/output/predictions.csv"
    
    main(model_filepath, data_filepath, output_filepath)