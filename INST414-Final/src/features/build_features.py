import pandas as pd
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
    df = pd.read_csv("data/processed/cleaned_data.csv")
    df = create_features(df)
    df.to_csv("data/processed/feature_data.csv", index=False)
