# Wellness Tourism Package Prediction - MLOps Pipeline

Predicts whether a "Visit with Us" customer will buy the Wellness Tourism Package, with an automated
training and deployment pipeline.

## Structure
| Path | What it does |
|---|---|
| `tourism_project/data/tourism.csv` | raw customer dataset |
| `tourism_project/model_building/data_register.py` | validates the schema and prints a data summary |
| `tourism_project/model_building/prep.py` | cleans the data and creates the train/test split |
| `tourism_project/model_building/train.py` | tunes XGBoost, tracks runs in MLflow, saves the best model |
| `tourism_project/deployment/app.py` | Streamlit app used by the sales team |
| `.github/workflows/pipeline.yml` | GitHub Actions: register -> prep -> train -> commit model |

## How it runs
Every push to `main` triggers the workflow. The trained model is committed to
`tourism_project/deployment/best_tourism_model_v1.joblib`, and the Streamlit Community Cloud app
picks it up automatically.
