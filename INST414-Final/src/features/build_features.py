# src/features/build_features.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    """Create new features for model training."""
    # Create Age Category
    df["Age Category"] = pd.cut(df["Age"], bins=[13, 23, 33, 43, 53, 63, 73],
                                labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"], right=False)
    
    # Create Engagement Score
    df["EngagementScore"] = MinMaxScaler().fit_transform(df[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", 
                                                           "EmailOpens", "EmailClicks", "SocialShares"]]).sum(axis=1)
    
    # Create Email Engagement Ratio
    df["EmailEngagementRatio"] = df["EmailClicks"].div(df["EmailOpens"].replace(0, 0), fill_value=0)
    
    return df

if __name__ == "__main__":
    # Load the raw data using the simplified path
    data_path = "INST414-Final/data/raw/digital_marketing_campaign_dataset.csv"
    
    # Ensure the path exists
    if not os.path.exists(data_path):
        # Try alternate path format
        data_path = "data/raw/digital_marketing_campaign_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found. Tried: INST414-Final/data/raw/digital_marketing_campaign_dataset.csv and data/raw/digital_marketing_campaign_dataset.csv")
        
    df = pd.read_csv(data_path)
    
    # Optional: Clean data if needed
    if "AdvertisingPlatform" in df.columns and "AdvertisingTool" in df.columns:
        df.drop(columns=["AdvertisingPlatform", "AdvertisingTool"], inplace=True)
    
    # Create features
    df = create_features(df)
    
    # Determine the processed directory path
    if "INST414-Final" in data_path:
        processed_dir = "INST414-Final/data/processed"
    else:
        processed_dir = "data/processed"
    
    # Create the processed data directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the processed data
    output_path = os.path.join(processed_dir, "feature_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")