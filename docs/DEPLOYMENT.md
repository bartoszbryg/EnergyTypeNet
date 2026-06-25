# Deployment Guide

This project now has three ways to use the model outside notebooks.

## 1. Train and Save a Model

```powershell
python -m src.train --feature-set core
```

This creates:

```text
artifacts/model.joblib
artifacts/metrics.json
```

## 2. Command-Line Prediction

Create a small CSV with the required columns:

```csv
Building Type,Square Footage,Number of Occupants,Appliances Used,Average Temperature,Day of Week,Energy Consumption
Residential,25000,48,25,22.5,Weekday,4100
```

Then run:

```powershell
python -m src.predict --input sample.csv
```

The output is JSON with the predicted class and class probabilities.

## 3. FastAPI Service

Start the API:

```powershell
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example request body for `POST /predict`:

```json
{
  "square_footage": 25000,
  "number_of_occupants": 48,
  "appliances_used": 25,
  "average_temperature": 22.5,
  "day_of_week": "Weekday",
  "energy_consumption": 4100
}
```

### API Smoke Tests

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example output:

```text
status
------
ok
```

Prediction request:

```powershell
$body = @{
  square_footage = 25000
  number_of_occupants = 20
  appliances_used = 30
  average_temperature = 72
  day_of_week = "Weekday"
  energy_consumption = 4100
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Example output:

```text
class      probabilities
-----      -------------
Commercial @{Residential=0.292...; Commercial=0.548...; Industrial=0.159...}
```

Automated API tests:

```powershell
pytest tests/test_api.py -q
```

Example output:

```text
2 passed
```

## 4. Streamlit Dashboard

```powershell
streamlit run dashboard.py
```

The dashboard includes:

- dataset overview
- data explorer
- scatter visualization
- single-row prediction tool
- saved metrics view

## 5. Docker

Build:

```powershell
docker build -t energytypenet .
```

Run:

```powershell
docker run -p 8000:8000 energytypenet
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## 6. Online Deployment Options

Good beginner-friendly options:

- Render: deploy the FastAPI app with the Dockerfile.
- Railway: deploy the FastAPI app from GitHub.
- Hugging Face Spaces: easiest for the Streamlit dashboard.

For an industry-style portfolio, deploy both:

- FastAPI API for backend inference.
- Streamlit dashboard for an interactive demo.
