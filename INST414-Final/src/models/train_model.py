# src/model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath):
    """ Load and preprocess the CSV data. """
    df = pd.read_csv(filepath)
    
    # Preprocessing steps from your earlier code
    df.drop(columns=["AdvertisingPlatform", "AdvertisingTool"], inplace=True)
    df["Age Category"] = pd.cut(df["Age"], bins=[13, 23, 33, 43, 53, 63, 73], labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"], right=False)
    df["EngagementScore"] = MinMaxScaler().fit_transform(df[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", "EmailOpens", "EmailClicks", "SocialShares"]]).sum(axis=1)
    df = df[(df['EngagementScore'] >= df['EngagementScore'].quantile(0.25)) & (df['EngagementScore'] <= df['EngagementScore'].quantile(0.75))]  # Removing outliers
    
    # Feature selection
    X = df[["EngagementScore", "Income", "AdSpend", "Age Category", "CampaignChannel", "PreviousPurchases"]]
    X = pd.get_dummies(X, columns=["Age Category", "CampaignChannel"], drop_first=True)
    y = df["Conversion"]
    
    return X, y

def train_random_forest(X_train, y_train, X_test, y_test):
    """ Train a Random Forest model and evaluate it. """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    return classification_report(y_test, y_pred)
