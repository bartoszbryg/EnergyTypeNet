# EnergyTypeNet

I built EnergyTypeNet to predict whether a building is Residential, Commercial or Industrial from its energy consumption data. The core idea was to go beyond just using sklearn and actually implement some models from scratch so I could understand what is happening inside them. I built three from scratch: an attention-weighted nearest neighbour classifier using exponential kernel weighting, a One-vs-Rest logistic regression trained with gradient descent and L2 regularisation, and a Softmax regression with a joint weight matrix and categorical cross-entropy loss. On top of those I trained sklearn Logistic Regression, MLP and XGBoost, then compared all six using 5-fold stratified cross-validation. I also built soft voting and stacking ensembles to see if combining models beats any single one.

Everything is packaged into a REST API, a CI pipeline, MLflow experiment tracking and a Streamlit dashboard that works on any CSV you upload, not just this dataset. The research part answers a specific question I had: is the accuracy ceiling caused by too little data or by the classes being too similar in feature space. Notebook 06 runs a synthetic experiment across different sample sizes and class separability levels that proves its the latter, so collecting more data would not fix it. One finding that surprised me was that the extended feature set produced near-perfect cross-validation scores, which looked impressive until I realised those extra features basically encode the label directly, so the honest benchmark is actually the two-feature setup where all models sit between 60 and 67 percent. XGBoost wins but only by a few points, and the custom attention classifier matches sklearn logistic regression despite being written entirely in NumPy.

---

## What it does

Given data about a building (energy consumption, square footage, number of occupants etc.) the system predicts whether it is Residential, Commercial or Industrial. Six models compete on the same data so you can see exactly where each one wins and loses.

The more interesting part is the dashboard. You can upload any CSV file, pick a target column, and the same pipeline runs on your data automatically. No code needed.

---

## Dataset background

The **Energy Consumption Dataset** contains information about energy usage across different building types. Each row describes one building using capacity, usage and environmental attributes. The training set has 1000 buildings and the test set has 100 buildings, which makes it useful for building a model on one split and validating predictions on a separate holdout split.

The original dataset can support both classification and regression tasks. In this project I focus on classification, using building information to predict the building type.

**Features included in the dataset:**

- Building Type
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Day of Week
- Energy Consumption

**Possible classification targets:**

- Building Type
- Day of Week

**Possible regression targets:**

- Energy Consumption
- Appliances Used
- Average Temperature
- Square Footage

### Real-world relevance

This problem is relevant for construction teams, architectural designers, building managers and utility companies. A model like this can help estimate expected energy usage, identify building type from consumption patterns, support energy-efficiency planning and improve billing or operational decisions. For example, square footage, number of occupants and appliance usage can help estimate whether a building is likely to have high energy demand.

The dataset also connects to sustainability. Understanding energy demand patterns matters because buildings contribute to energy costs, greenhouse gas emissions and the use of non-renewable resources. Better prediction tools can help companies and residents plan more efficiently.

### Potential challenges

The dataset does not include every factor that affects energy usage. Missing features such as building materials, insulation quality, location, climate, country, building age and resident behavior can all change energy consumption. Some buildings may also be more densely populated than others, which can make building type harder to predict.

Another challenge is class overlap. Different building types can have similar energy consumption and square footage, so the model cannot always separate them cleanly. This is one reason the honest two-feature benchmark has a clear accuracy ceiling.

### Reflection

Machine learning on building energy data can be useful for future construction, architecture and power-management systems. The results show that the model can learn meaningful patterns, but also that model quality depends heavily on the features available. Training on more data can help in some cases, but adding better attributes would likely improve this project more than simply collecting more rows with the same fields.

---

## Models

Three of these are built from scratch without sklearn:

| Model | Type | Notes |
|---|---|---|
| LogisticRegressionOvR | Custom | One-vs-Rest, gradient descent, L2 regularisation |
| LogisticRegressionSoftmax | Custom | Joint weight matrix W, categorical cross-entropy |
| AttentionClassifier | Custom | Exponential kernel weighting, vectorised (m,n,d) broadcast |
| LogisticRegression | sklearn | Multinomial, grid search over C |
| MLPClassifier | sklearn | Grid search over architecture and activation |
| XGBClassifier | XGBoost | Grid search over depth, learning rate, subsampling |

Ensemble methods (soft voting and stacking) are explored in notebook 05.

---

## Project structure

```
data/
  train_energy_data.csv         1000 buildings for training
  test_energy_data.csv          100 buildings, used once for final evaluation

notebooks/
  01_exploratory_data_analysis  class balance, distributions, correlations
  02_model_training_evaluation  training, hyperparameter search, results
  03_feature_engineering        ANOVA ranking, PCA, t-SNE, ablation study
  04_model_interpretability     learning curves, ROC/AUC, permutation importance, calibration, PR curves
  05_ensemble_stacking          voting and stacking vs individual models
  06_synthetic_experiment       tests whether the accuracy ceiling is overlap or sample size

src/
  data.py                       load_raw, load_features, make_engineered_features
  models.py                     the three custom classifiers
  evaluation.py                 cross_validate_custom, plot_decision_boundaries, plot_confusion_matrices
  train.py                      trains all candidates, selects best by CV, saves artifact + MLflow run
  predict.py                    command line inference from a CSV file
  api.py                        FastAPI prediction endpoint
  synthetic_experiment.py       standalone script for the separability experiment

dashboard.py                    Streamlit app with two modes (see below)
Dockerfile                      packages the API for deployment
tests/                          unit tests for custom models and data loading
docs/                           deployment guide and DVC notes
```

---

## Dashboard

Three modes in one app:

**EnergyTypeNet mode** has 9 pages: overview, EDA, model comparison, decision boundaries, confusion matrices, ROC curves, precision-recall curves with threshold sweep, learning curves and a live prediction tool where sliders update all 5 models in real time.

**Custom Dataset mode** lets you upload any CSV. You pick the target column and feature columns, categorical columns get one-hot encoded automatically, and you get model comparison, confusion matrices, ROC curves, PR curves, PCA decision boundary projection and live prediction with a JSON download.

**AI Dataset Assistant mode** profiles any uploaded CSV, suggests likely target and feature columns, recommends classification vs regression, trains tabular baseline models, ranks features, generates a short dataset report and answers basic questions using computed dataset statistics.

```bash
streamlit run dashboard.py
```

---

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate        # Mac/Linux

pip install -r requirements.txt
```

---

## Running things

Run notebooks in order 01 to 06 using Jupyter.

Train and save the best model:
```bash
python -m src.train --feature-set core --no-mlflow
```

Predict from the command line:
```bash
python -m src.predict --input data/test_energy_data.csv
```

Start the API:
```bash
uvicorn src.api:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

Run tests:
```bash
pytest -q
```

Track experiments with MLflow (run without --no-mlflow flag, then):
```bash
mlflow ui
# opens at http://localhost:5000
```

---

## Key findings

XGBoost reaches 0.67 test accuracy on the two core features, others sit between 0.60 and 0.64. The synthetic experiment in notebook 06 confirms the ceiling comes from class overlap in feature space rather than insufficient data. Adding more buildings of the same type would not help. The extended feature set produces near-perfect CV scores which is suspicious and likely means those extra features encode the label too strongly, so the two-feature benchmark is the honest comparison.

---

## Deployment

The API is containerised with Docker and can be deployed on Render or Railway using the included Dockerfile. The dashboard deploys to Streamlit Cloud by connecting the GitHub repo and pointing at dashboard.py. See docs/DEPLOYMENT.md for step by step instructions.
