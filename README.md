# Credit Risk Prediction API

This project is a FastAPI-based machine learning API that predicts credit risk for loan applicants using a trained classification model.

## Features
- Predicts credit risk (Good / Bad)
- Returns risk probability
- REST API built using FastAPI
- Trained ML model integrated using scikit-learn

## Tech Stack
- Python
- FastAPI
- scikit-learn
- Pandas
- Joblib
- Uvicorn

## API Endpoints
### GET /
Health check endpoint.

### POST /predict
Accepts applicant details and returns credit risk prediction.

## Input Fields
- Age
- Job
- Credit amount
- Duration
- Sex
- Housing
- Saving accounts
- Checking account
- Purpose

## Output
- risk_prediction (0 or 1)
- risk_probability
- risk_label (Good / Bad)

## Deployment
The API is deployed using Render.

## Author
Samriddha Chakraborty
