import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import mlflow

# threshold picked from the out-of-fold analysis in the notebook (must match app.py)
CLASSIFICATION_THRESHOLD = 0.40
MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-package")

# splits come from the data-prep job artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus"]
numeric_features = [c for c in Xtrain.columns if c not in categorical_features]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# weight the minority class (buyers)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

model_pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(scale_pos_weight=class_weight, colsample_bytree=0.8,
                      random_state=42, eval_metric="logloss"),
)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run(run_name="pipeline_training"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # log every combination that was tried
    res = grid_search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"combo_{i:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", res["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", res["std_test_score"][i])

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    print("Best params:", grid_search.best_params_)
    print(f"Best CV F1: {grid_search.best_score_:.3f}")

    # evaluate with the chosen threshold
    metrics = {}
    for split, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        proba = best_model.predict_proba(X)[:, 1]
        pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        report = classification_report(y, pred, output_dict=True)
        metrics.update({
            f"{split}_accuracy": report["accuracy"],
            f"{split}_precision": report["1"]["precision"],
            f"{split}_recall": report["1"]["recall"],
            f"{split}_f1": report["1"]["f1-score"],
            f"{split}_roc_auc": roc_auc_score(y, proba),
        })
        if split == "test":
            print(classification_report(y, pred))
    mlflow.log_metrics(metrics)

    # save next to app.py so Streamlit can load it, and keep a copy in MLflow
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"Model saved to {MODEL_PATH}")
