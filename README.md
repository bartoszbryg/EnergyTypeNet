# EnergyTypeNet

EnergyTypeNet is a machine-learning project for predicting whether a building is **Residential**, **Commercial**, or **Industrial** from energy-consumption and building-attribute data.

The project started as a building-type classification pipeline, but I expanded it into a broader machine-learning system with custom NumPy models, sklearn/XGBoost baselines, model diagnostics, ensemble learning, API deployment, MLflow tracking, and a reusable AI Dataset Assistant for uploaded CSV files.

The main goal of the project is not only to get good accuracy, but also to understand how different learning algorithms behave internally. Instead of only applying ready-made sklearn models, I implemented several models from scratch using NumPy and exposed them through sklearn-compatible estimator APIs where possible.

---

## Main Capabilities

* Predict building type from energy-consumption and building-attribute data.
* Implement multiple machine-learning models from scratch with NumPy.
* Compare custom models against sklearn, neural-network, and XGBoost baselines.
* Train soft-voting and stacking ensembles.
* Run feature engineering, feature selection, model interpretation, and diagnostic notebooks.
* Track experiments with MLflow.
* Train and serialize production models with joblib.
* Serve predictions through a FastAPI endpoint.
* Explore results through a Streamlit dashboard.
* Upload custom CSV files and run a lightweight AutoML-style workflow.
* Generate dataset reports and grounded natural-language explanations.
* Optionally stream local LLM answers with Ollama when running locally.

---

## Dataset Background

The Energy Consumption Dataset contains building-level data with usage, capacity, and environmental attributes.

Features in the original dataset include:

* Building Type
* Square Footage
* Number of Occupants
* Appliances Used
* Average Temperature
* Day of Week
* Energy Consumption

The main supervised task in this project is **building-type classification**, where the target variable is `Building Type`.

Possible additional task formulations include:

| Task Type      | Possible Targets                                                         |
| -------------- | ------------------------------------------------------------------------ |
| Classification | Building Type, Day of Week                                               |
| Regression     | Energy Consumption, Appliances Used, Average Temperature, Square Footage |

This problem is relevant to building management, construction planning, architecture, utility billing, and energy-efficiency analysis. The dataset is also useful for discussing model limitations because important real-world factors such as climate, location, insulation, building age, materials, and resident behavior are not included.

---

## EnergyTypeNet Models

### Custom Models Implemented from Scratch

| Model                          | Implementation                               | Purpose                                                                     |
| ------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------- |
| `AttentionClassifier`          | NumPy + sklearn-compatible estimator API     | Kernel-weighted nearest-neighbor style classifier                           |
| `LogisticRegressionOvR`        | NumPy                                        | One-vs-Rest logistic regression with gradient descent and L2 regularization |
| `LogisticRegressionSoftmax`    | NumPy                                        | Multiclass softmax regression with cross-entropy loss                       |
| `DecisionTreeClassifierCustom` | NumPy                                        | CART-style classifier with Gini or entropy splits                           |
| `DecisionTreeRegressorCustom`  | NumPy                                        | CART-style regressor using MSE reduction                                    |
| `SVMClassifierCustom`          | NumPy + random Fourier features for RBF mode | Binary soft-margin SVM with hinge-loss optimization                         |
| `GaussianNaiveBayes`           | NumPy                                        | Probabilistic classifier for continuous numeric features                    |
| `MultinomialNaiveBayes`        | NumPy                                        | Count-feature Naive Bayes for text or frequency-style features              |
| `BernoulliNaiveBayes`          | NumPy                                        | Binary-feature Naive Bayes with optional thresholding                       |
| `BayesianLinearRegression`     | NumPy                                        | Bayesian regression with predictive mean and variance                       |

### Production Training Candidates

The production training script also compares stronger standard baselines:

| Model               | Notes                                                                 |
| ------------------- | --------------------------------------------------------------------- |
| Logistic Regression | Standardized sklearn pipeline                                         |
| MLPClassifier       | sklearn neural-network baseline                                       |
| XGBoost             | Gradient-boosted tree model                                           |
| Soft Voting         | Combines Logistic Regression, MLP, and XGBoost                        |
| Stacking            | Meta-learner over Logistic Regression, MLP, and XGBoost probabilities |

### AI Dataset Assistant Baselines

The reusable AI Dataset Assistant can train classification and regression baselines for uploaded CSV files.

| Classification      | Regression                  |
| ------------------- | --------------------------- |
| Dummy baseline      | Dummy baseline              |
| Logistic Regression | Ridge Regression            |
| KNN                 | KNN Regressor               |
| SVM                 | SVR                         |
| Random Forest       | Random Forest Regressor     |
| Gradient Boosting   | Gradient Boosting Regressor |
| MLP Neural Network  | MLP Regressor               |
| XGBoost             | XGBoost Regressor           |

---

## Advanced Model Suite

This branch adds a broader classical machine-learning model suite beyond the original custom attention and logistic-regression models.

The advanced suite includes:

* custom CART decision tree classifier
* custom CART decision tree regressor
* custom linear/RBF SVM classifier
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Bayesian Linear Regression
* additional model tests for the new estimators

This makes the project stronger as a learning portfolio because it demonstrates how core ML algorithms work internally rather than only using library implementations.

---

## Dashboard

Run the dashboard with:

```bash
streamlit run dashboard.py
```

The Streamlit app has three modes.

### EnergyTypeNet Mode

Uses the bundled energy-consumption dataset and project models. It includes:

* overview metrics
* exploratory data analysis
* model comparison
* decision boundaries
* confusion matrices
* ROC/AUC curves
* precision-recall curves
* learning curves
* live prediction controls

### Custom Dataset Mode

Lets a user upload a CSV, manually choose target/features, and run a reusable tabular-modeling workflow with visual diagnostics.

### AI Dataset Assistant Mode

Turns any uploaded CSV into a guided AutoML-style analysis.

It can:

* profile rows, columns, dtypes, missing values, and duplicate rows
* suggest likely target columns
* infer classification vs regression
* suggest usable feature columns
* rank features with mutual information
* recommend a compact feature set
* train classification or regression baselines
* compare full selected features against compact selected features
* generate a short dataset report
* answer questions about model quality, missingness, important features, task type, overfitting, and leakage

The assistant uses deterministic, computed-statistic answers by default. Local LLM streaming is optional and only runs when Ollama is installed and active on the user's machine.

The upload workflow is guarded for normal public use: it expects CSV input, removes empty rows and columns, checks whether a target can be modeled, rejects continuous numeric targets accidentally used as classification labels, and shows friendly messages when the selected dataset cannot be prepared.

---

## Local LLM Explanations

The app is free-first by design.

* Deterministic dataset explanations work locally and in public deployments without any API key.
* Local Ollama streaming is free, but only works on a machine with Ollama installed.
* Hosted LLM streaming can be added later, but it may cost money because API providers usually charge per token.
* Public demos should keep hosted LLM mode disabled by default unless usage limits are added.

To test local LLM streaming:

```bash
ollama pull llama3.1
ollama run llama3.1
streamlit run dashboard.py
```

Then open the AI Dataset Assistant, enable local LLM explanation if Ollama is running, keep the model as `llama3.1`, and ask a dataset question.

---

## Project Structure

```text
data/
  train_energy_data.csv              Training split for EnergyTypeNet
  test_energy_data.csv               Holdout split for EnergyTypeNet
  sample_building_operations.csv     Small sample CSV for the AI Dataset Assistant

notebooks/
  01_exploratory_data_analysis.ipynb
  02_model_training_evaluation.ipynb
  03_feature_engineering.ipynb
  04_model_interpretability.ipynb
  05_ensemble_stacking.ipynb
  06_synthetic_experiment.ipynb
  07_advanced_model_suite.ipynb
  08_decision_tree_models.ipynb
  09_svm_naive_bayes_models.ipynb
  10_bayesian_regression.ipynb

src/
  api.py                             FastAPI prediction service
  automl.py                          CSV profiling, target/feature suggestions and baseline training
  data.py                            Energy dataset loading and feature engineering
  evaluation.py                      Evaluation and plotting helpers
  llm_assistant.py                   Optional local Ollama prompt/streaming helpers
  models.py                          Custom NumPy classifiers and regressors
  predict.py                         CLI prediction helpers
  synthetic_experiment.py            Synthetic separability experiment
  train.py                           Production model training script

tests/
  test_api.py
  test_automl.py
  test_data.py
  test_llm_assistant.py
  test_models.py

docs/
  DEPLOYMENT.md
  DVC.md

dashboard.py                         Streamlit dashboard
Dockerfile                           API container
requirements.txt                     Python dependencies
pytest.ini                           Test configuration
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Common Commands

Run tests:

```bash
pytest -q
```

Compile-check Python files:

```bash
python -m compileall src tests dashboard.py
```

Train and save the production model:

```bash
python -m src.train --feature-set core --no-mlflow
```

Train with MLflow logging:

```bash
python -m src.train --feature-set core
mlflow ui
```

Predict from a CSV:

```bash
python -m src.predict --input data/test_energy_data.csv
```

Start the API:

```bash
uvicorn src.api:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

Run the synthetic experiment:

```bash
python -m src.synthetic_experiment
```

---

## API Example

Start the API:

```bash
uvicorn src.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"square_footage\":25000,\"number_of_occupants\":20,\"appliances_used\":30,\"average_temperature\":72,\"day_of_week\":\"Weekday\",\"energy_consumption\":4100}"
```

---

## Validation

Before committing or deploying, run:

```bash
python -m pip check
pytest -q
python -m compileall src tests dashboard.py
```

Expected current result:

```text
No broken requirements found.
tests passed
compileall passed
```

Warnings from FastAPI/Starlette internals or sklearn MLP convergence on tiny test data are not currently project failures.

---

## Deployment

### Streamlit Dashboard

The easiest public deployment path is Streamlit Community Cloud:

1. Push the latest code to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select this repository and the target branch.
5. Set the main file path to `dashboard.py`.
6. Deploy.

The public Streamlit version supports CSV upload, profiling, target/feature suggestions, baseline training, feature ranking, model comparison, dataset reports, and deterministic dataset Q&A.

Local Ollama streaming only works when running the project locally with Ollama installed.

### FastAPI

The API can be containerized with Docker:

```bash
docker build -t energytypenet .
docker run -p 8000:8000 energytypenet
```

See `docs/DEPLOYMENT.md` for deployment notes.

---

## Current Findings

The core EnergyTypeNet task is intentionally honest about feature limitations.

The two-feature benchmark using `Energy Consumption` and `Square Footage` is the cleanest comparison because it avoids features that may encode the label too directly. The synthetic separability experiment supports the idea that the accuracy ceiling is mainly caused by class overlap in feature space, not simply by a shortage of rows.

The advanced model suite adds more algorithmic depth by comparing several learning families: linear models, tree models, margin-based models, probabilistic classifiers, Bayesian regression, neural networks, boosted trees, and ensembles.

The AI Dataset Assistant extends the project beyond this one dataset by making the workflow reusable for other tabular CSV files while keeping explanations grounded in computed statistics.

---

## Suggested Next Branches

Good future improvements:

* `deploy-streamlit`: add live app link and screenshots after deployment
* `regularization-suite`: add L1/L2 regularization experiments, Lasso, Ridge, and ElasticNet comparisons
* `pytorch-tabular-models`: add custom PyTorch classifier/regressor and training curves
* `dataset-chat-agent`: add chat history and richer follow-up questions
* `hosted-llm-provider`: add optional API-key based hosted LLM streaming with usage controls
