# INST414-Final

Hey this is my final project for INST 414.

Summary: This project analyzes social media engagement data to predict user conversions (e.g., purchases or sign-ups) based on demographics and interaction metrics. It involves processing raw data into features like `Age Category` and `EngagementScore`, training a machine learning model (e.g., Random Forest), and visualizing both raw data patterns and prediction outcomes. Built with Python, it uses Pandas for data handling, Scikit-learn for modeling, and Seaborn/Matplotlib for visualization.

## Dependencies
To run this project, you’ll need the following tools and libraries:
- **Python** (version 3.8 or higher)
- **Pandas** (version 1.5.0) - For data manipulation and CSV handling
- **Scikit-learn** (version 1.2.0) - For feature scaling (MinMaxScaler) and model training/prediction
- **Click** (version 8.1.0) - For command-line interface in preprocessing
- **Python-dotenv** (version 1.0.0) - For loading environment variables
- **Joblib** (version 1.2.0) - For saving/loading trained models
- **Seaborn** (version 0.12.0) - For statistical data visualization
- **Matplotlib** (version 3.7.0) - For plotting graphs
- **NumPy** (version 1.23.0) - For numerical operations (implicit via Pandas/Scikit-learn)


#Setup Instructions
Git Clone the File
Pip Install all the Dependencies
run python scripts/preprocess.py data/raw/raw_data.csv data/processed/feature_data.csv
run python src/models/train_model.py  # Train the model (adjust if different)
run python scripts/visualize.py        # Explore the processed data
run python scripts/predict.py          # Generate predictions and visualizations