# src/models/train_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath):
    """ Load and preprocess the CSV data. """
    df = pd.read_csv(filepath)
    
    # Preprocessing steps from your earlier code
    if "AdvertisingPlatform" in df.columns and "AdvertisingTool" in df.columns:
        df.drop(columns=["AdvertisingPlatform", "AdvertisingTool"], inplace=True)
    
    # Create Age Category
    df["Age Category"] = pd.cut(df["Age"], bins=[13, 23, 33, 43, 53, 63, 73], 
                               labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"], right=False)
    
    # Create Engagement Score
    df["EngagementScore"] = MinMaxScaler().fit_transform(df[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", 
                                                           "EmailOpens", "EmailClicks", "SocialShares"]]).sum(axis=1)
    
    # Create Email Engagement Ratio
    df["EmailEngagementRatio"] = df["EmailClicks"].div(df["EmailOpens"].replace(0, 0), fill_value=0)
    
    # Feature selection
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", 
            "PreviousPurchases", "EmailEngagementRatio"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    y = df["Conversion"]
    
    return X, y, df

def train_random_forest(X_train, y_train, X_test, y_test):
    """ Train a Random Forest model and evaluate it. """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    return rf, report

def save_model(model, filename):
    """Save the trained model to a file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main(data_filepath, model_filepath):
    """Main function to train and save the model."""
    # Load and preprocess data
    X, y, df = load_and_preprocess_data(data_filepath)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model, report = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Save the model
    save_model(model, model_filepath)
    
    return model, report

if __name__ == "__main__":
    # Try different path formats
    data_paths = [
        "INST414-Final/data/processed/feature_data.csv",
        "data/processed/feature_data.csv"
    ]
    
    data_filepath = None
    for path in data_paths:
        if os.path.exists(path):
            data_filepath = path
            break
    
    if data_filepath is None:
        raise FileNotFoundError("Could not find the processed data file. Please run build_features.py first.")
    
    # Determine model path based on data path
    if "INST414-Final" in data_filepath:
        model_filepath = "INST414-Final/models/trained_model_rf.pkl"
    else:
        model_filepath = "models/trained_model_rf.pkl"
    
    main(data_filepath, model_filepath)