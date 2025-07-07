import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import json

# Create paths to the sample files
model_bin_path =  '../model/xgb_churn_model_1.bin'
features_path = '../data/feature_names_1.json'
X_test_path = '../data/X_test_sample_1.json'

# Load feature names
with open(features_path, 'r') as f:
    feature_names = json.load(f)

# Load data
X_test = pd.read_json(X_test_path, lines=True)
X_test = X_test[feature_names]

# Load XGBoost model
model = xgb.Booster()
model.load_model(model_bin_path)

# Set MLflow experiment name
mlflow.set_experiment('churn_xgboost_experiment')

with mlflow.start_run(run_name='churn_xgboost_model_run'):
    # Log model parameters (none were provided so we'll log a placeholder)
    mlflow.log_param('model_type', 'xgboost.Booster')

    # Log a test metric (dummy accuracy, since we don't have y_true)
    mlflow.log_metric('dummy_metric', 1.0)

    # Log the model with input/output signature for reproducibility
    input_example = X_test.head(1)
    signature = mlflow.models.infer_signature(X_test, model.predict(xgb.DMatrix(X_test)))
    mlflow.xgboost.log_model(
        xgb_model=model,
        name='model',
        signature=signature,
        input_example=input_example,
        registered_model_name='ChurnXGBoostModel'
    )
