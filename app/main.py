from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import json
import numpy as np
import os

# ── Load model + metadata ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "metadata.json")

pipeline = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

FEATURES              = metadata["features"]
LOAN_PURPOSE_CLASSES  = metadata["loan_purpose_classes"]

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Predicts the probability that a loan applicant will default, "
        "based on financial profile and loan terms. "
        "Built with scikit-learn (Random Forest) and deployed via Render."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100, example=34)
    income: float = Field(..., gt=0, example=55000)
    loan_amount: float = Field(..., gt=0, example=12000)
    loan_term: int = Field(..., description="Loan term in months", example=36)
    interest_rate: float = Field(..., ge=0, le=100, example=11.5)
    credit_score: int = Field(..., ge=300, le=850, example=640)
    employment_years: int = Field(..., ge=0, example=4)
    num_prev_loans: int = Field(..., ge=0, example=2)
    missed_payments: int = Field(..., ge=0, example=0)
    loan_purpose: Literal["business", "personal", "education", "home_improvement", "medical"] = Field(
        ..., example="personal"
    )

    @validator("loan_term")
    def validate_term(cls, v):
        allowed = [12, 24, 36, 48, 60]
        if v not in allowed:
            raise ValueError(f"loan_term must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    default_prediction: int = Field(..., description="1 = likely to default, 0 = unlikely")
    default_probability: float = Field(..., description="Probability of default (0.0 – 1.0)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH based on probability")
    model_version: str


# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def build_features(data: LoanApplication) -> list:
    purpose_enc = LOAN_PURPOSE_CLASSES.index(data.loan_purpose)

    debt_to_income = data.loan_amount / data.income

    monthly_payment_burden = (
        (data.loan_amount * data.interest_rate / 100) / data.loan_term
    ) / (data.income / 12)

    credit_risk_flag = int(data.credit_score < 580)
    high_missed_flag = int(data.missed_payments >= 2)

    return [
        data.age,
        data.income,
        data.loan_amount,
        data.loan_term,
        data.interest_rate,
        data.credit_score,
        data.employment_years,
        data.num_prev_loans,
        data.missed_payments,
        debt_to_income,
        purpose_enc,
        monthly_payment_burden,
        credit_risk_flag,
        high_missed_flag,
    ]


def risk_label(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.60:
        return "MEDIUM"
    return "HIGH"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Loan Default Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model": metadata["model_type"],
        "roc_auc": metadata["roc_auc"],
        "test_accuracy": metadata["test_accuracy"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    try:
        features = build_features(application)
        X = np.array(features).reshape(1, -1)

        prediction   = int(pipeline.predict(X)[0])
        probability  = round(float(pipeline.predict_proba(X)[0][1]), 4)

        return PredictionResponse(
            default_prediction=prediction,
            default_probability=probability,
            risk_level=risk_label(probability),
            model_version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
