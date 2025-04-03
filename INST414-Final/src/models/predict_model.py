# src/model/predict_model.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.model.train_model import load_and_preprocess_data  # Import preprocessing functions

def load_trained_model(model_filepath):
    """Load the trained model from a file."""
    return joblib.load(model_filepath)

def preprocess_input_data(filepath):
    """Preprocess input data in the same way as during the training phase."""
    # Load and preprocess the data just like in the training pipeline
    financial_marketing = pd.read_csv(filepath)
    
    # FEATURE ENGINEERING
    # Group up ages for more insights
    financial_marketing["Age Category"] = pd.cut(financial_marketing["Age"],
                                                 bins=[13, 23, 33, 43, 53, 63, 73],
                                                 labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"],
                                                 right=False)
    # Engagement Score normalization
    financial_marketing["EngagementScore"] = MinMaxScaler().fit_transform(financial_marketing[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", "EmailOpens", "EmailClicks", "SocialShares"]]).sum(axis=1)
    # Email engagement ratio
    financial_marketing["EmailEngagementRatio"] = financial_marketing["EmailClicks"].div(financial_marketing["EmailOpens"].replace(0, 0), fill_value=0)

    # Get the features and preprocess them for prediction
    X = financial_marketing[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    
    return X

def make_predictions(model, data_filepath):
    """Make predictions using the trained model."""
    # Preprocess the input data (ensure consistency with the training set)
    X = preprocess_input_data(data_filepath)
    
    # Make predictions using the model
    predictions = model.predict(X)
    
    return predictions

def save_predictions(predictions, output_filepath):
    """Save the predictions to a CSV file."""
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to {output_filepath}")
    
def visualize_predictions(predictions, data_filepath):
    """Visualize some of the predictions (optional)."""
    # Load data for the same records
    financial_marketing = pd.read_csv(data_filepath)
    
    # Add predictions to the dataframe for visualization
    financial_marketing["Predictions"] = predictions

    # Visualize the predictions, e.g., distribution of predicted values
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Predictions", data=financial_marketing, palette="coolwarm")
    plt.xlabel("Predicted Conversion")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Conversions")
    plt.show()

def main(model_filepath, data_filepath, output_filepath):
    """Main function to load model, make predictions, and save them."""
    # Load the trained model
    model = load_trained_model(model_filepath)
    
    # Make predictions
    predictions = make_predictions(model, data_filepath)
    
    # Save predictions to file
    save_predictions(predictions, output_filepath)
    
    # Optional: Visualize predictions
    visualize_predictions(predictions, data_filepath)

if __name__ == "__main__":
    # Example usage
    model_filepath = 'src/model/trained_model_rf.pkl'  # Path to the trained model file
    data_filepath = 'input_data.csv'  # Path to the input CSV file
    output_filepath = 'predictions_output.csv'  # Output file path to save predictions

    main(model_filepath, data_filepath, output_filepath)
