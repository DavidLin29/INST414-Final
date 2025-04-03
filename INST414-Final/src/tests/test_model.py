# tests/test_model.py

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model.train_model import load_and_preprocess_data, train_random_forest

@pytest.fixture
def sample_data():
    """Fixture to load and preprocess sample data for testing."""
    # Assuming your CSV file is in the root directory of your project
    file_path = "digital_marketing_campaign_dataset.csv"
    X, y = load_and_preprocess_data(file_path)
    return X, y

def test_model_training(sample_data):
    """Test if the model trains without errors and produces classification report."""
    X, y = sample_data
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model and evaluate
    report = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Assert that the classification report contains key metrics (accuracy, precision, recall)
    assert "accuracy" in report
    assert "precision" in report
    assert "recall" in report
    
    print("Model training and evaluation passed!")
