from dagster import op, job
import pandas as pd
import json
import os
import mlflow
import xgboost as xgb
from datetime import datetime

# Get the directory of the current script (pipeline/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Fetch raw input data
@op
def load_test_data():
    # Create paths to the sample files
    features_path = os.path.join(script_dir, '..', 'data', 'feature_names_1.json')
    X_test_path = os.path.join(script_dir, '..', 'data', 'X_test_sample_1.json') # A hardcoded path for this example, but we could use a config file to get the path from the sensor

    # Load feature names
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    # Load sample data. We assume it's raw and it needs to be transformed, but we know it's already transformed
    X_test = pd.read_json(X_test_path, lines=True)
    X_test = X_test[feature_names]

    return X_test

# 2. Apply feature transformations (if any)
@op
def transform_features(X_test):
    # Dummy transformation. In real life, we would apply feature engineering here
    transformed_X = X_test.copy()

    return transformed_X

# 3. Load trained model artifact from MLflow
@op
def load_model():
    # Set the tracking URI to the local directory where MLflow runs are stored
    mlruns_path = os.path.abspath(os.path.join(script_dir, '..', 'experiment', 'mlruns'))
    mlflow.set_tracking_uri(f'file:{mlruns_path}')

    # Load the model from MLflow Model Registry
    model_uri = 'models:/ChurnXGBoostModel/1'
    model = mlflow.xgboost.load_model(model_uri)

    return model

# 4. Generate predictions and store them in a CSV file
@op
def generate_predictions(model, transformed_X):
    # Generate predictions
    predictions = model.predict(xgb.DMatrix(transformed_X))

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame({'prediction': predictions})

    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'predictions_{timestamp}.csv'

    # Save to CSV
    output_path = os.path.join(script_dir, 'output', output_filename)
    predictions_df.to_csv(output_path, index=False)

    return output_path

@job
def churn_prediction_job():
    X = load_test_data()
    X_trans = transform_features(X)
    model = load_model()
    output_path = generate_predictions(model, X_trans)


if __name__ == '__main__':
    result = churn_prediction_job.execute_in_process()
    print('Pipeline finished. Check the output folder for output.')