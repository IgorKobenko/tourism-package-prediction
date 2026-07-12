# ---------------------------------------------------------------
# MODEL TRAINING SCRIPT (production)
# Loads train/test splits from the HF data space, tunes an XGBoost
# pipeline with GridSearchCV, logs every experiment to MLflow, and
# registers the best model on the Hugging Face model hub.
# ---------------------------------------------------------------

# for data manipulation
import pandas as pd

# for preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

# for folders / env variables
import os

# for Hugging Face authentication and upload
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# for experiment tracking
import mlflow

# MLflow server is started by the GitHub Actions workflow on port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# ------------------ 1. LOAD TRAIN/TEST FROM THE HF DATA SPACE ------------------
Xtrain = pd.read_csv("hf://datasets/ivkobenko/tourism-package-prediction/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/ivkobenko/tourism-package-prediction/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/ivkobenko/tourism-package-prediction/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/ivkobenko/tourism-package-prediction/ytest.csv")
print("Data loaded:", Xtrain.shape, Xtest.shape)

# ------------------ 2. FEATURE DEFINITION ------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# ------------------ 3. CLASS-IMBALANCE HANDLING ------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ------------------ 4. PREPROCESSING + MODEL PIPELINE ------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------ 5. HYPERPARAMETER GRID ------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7, 0.9],
    "xgbclassifier__reg_lambda": [0.5, 1.0, 2.0],
}

# ------------------ 6. TUNING + MLFLOW TRACKING ------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5,
                               scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain.values.ravel())

    # Log every tested parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best parameters in the parent run
    mlflow.log_params(grid_search.best_params_)
    print("Best parameters:", grid_search.best_params_)

    # ------------------ 7. EVALUATE THE BEST MODEL ------------------
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)
    print("Test-set report:")
    print(classification_report(ytest, y_pred_test))

    mlflow.log_metrics({
        "train_accuracy":  train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall":    train_report["1"]["recall"],
        "train_f1-score":  train_report["1"]["f1-score"],
        "test_accuracy":   test_report["accuracy"],
        "test_precision":  test_report["1"]["precision"],
        "test_recall":     test_report["1"]["recall"],
        "test_f1-score":   test_report["1"]["f1-score"],
    })

    # ----------- 8. SAVE + REGISTER THE BEST MODEL ------------------
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path) 
    # joblib.dump serializes the entire fitted pipeline

    # Log the model file as an MLflow artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")
    #Attaches the file to the MLflow run 


    # Register (upload) the model on the Hugging Face model hub
    repo_id = "ivkobenko/tourism-package-model"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating a new one...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    # Uploads the joblib file. As the filename stays the same, each pipeline run overwrites the previous model — 
    # - the hub always holds the latest trained version, and the Streamlit app (hf_hub_download in app.py) always fetches whatever is current
    
    print("Best model registered on the Hugging Face model hub.")
