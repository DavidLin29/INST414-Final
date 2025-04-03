import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize(df):
    """Generate visualizations for data analysis."""
    sns.boxplot(x=df["Age Category"])
    plt.title("Age Category Distribution")
    plt.show()

    sns.scatterplot(x="Income", y="EngagementScore", hue="Age Category", data=df, alpha=0.7)
    plt.title("Income vs. Engagement Score by Age Category")
    plt.show()

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/processed/feature_data.csv")
    visualize(df)
