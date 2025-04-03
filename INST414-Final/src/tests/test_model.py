# src/tests/test_model.py

import pytest
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.train_model import load_and_preprocess_data, train_random_forest

@pytest.fixture
def sample_data():
    """Fixture to load and preprocess sample data for testing."""
    # Try different path formats
    paths = [
        "INST414-Final/data/processed/feature_data.csv",
        "data/processed/feature_data.csv"
    ]
    
    file_path = None
    for path in paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        pytest.skip("Could not find the processed data file. Please run build_features.py first.")
    
    X, y, _ = load_and_preprocess_data(file_path)
    return X, y

def test_model_training(sample_data):
    """Test if the model trains without errors and produces classification report."""
    X, y = sample_data
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model and evaluate
    model, report = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Assert that the classification report contains key metrics
    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report
    
    print("Model training and evaluation passed!")