# Loan Default Prediction API

A machine learning API that predicts the probability of loan default based on an applicant's financial profile and loan terms. Built with scikit-learn, served via FastAPI, and deployed on Render.

---

## The Problem

Lenders face real financial exposure when borrowers default. Manual credit assessment is slow and inconsistent. This project builds an automated scoring system that takes a loan application and returns a default probability and risk tier, giving analysts a fast, data-driven starting point before final approval.

---

## How It Works

1. **Model** — Random Forest classifier trained on 3,000 synthetic loan records with engineered features (debt-to-income ratio, monthly payment burden, credit risk flags)
2. **API** — FastAPI app exposes a `/predict` endpoint that accepts a JSON loan application and returns a prediction, probability, and risk level (LOW / MEDIUM / HIGH)
3. **Deployment** — Live on Render, accessible via HTTP from any client

---

## Project Structure

```
loan-default-api/
├── app/
│   ├── main.py              # FastAPI app — routes, schemas, feature engineering
│   └── model/
│       ├── loan_model.pkl   # Trained pipeline (scaler + Random Forest)
│       └── metadata.json    # Feature list, label encoding, eval metrics
├── notebook/
│   └── loan_default_training.ipynb   # Full training pipeline with EDA
├── data/
│   ├── loan_data.csv             # Synthetic training dataset
│   ├── eda_plot.png
│   ├── evaluation_plot.png
│   └── feature_importance.png
├── requirements.txt
├── Procfile
├── render.yaml
└── README.md
```

---

## Model Performance

| Metric       | Score  |
|-------------|--------|
| ROC-AUC     | ~0.87  |
| Test Accuracy | ~82%  |
| CV (5-fold AUC) | ~0.86 ± 0.01 |

Top predictors: `credit_score`, `missed_payments`, `debt_to_income`, `monthly_payment_burden`

---

## API Endpoints

| Method | Route      | Description                    |
|--------|------------|--------------------------------|
| GET    | `/`        | Root — confirms API is running |
| GET    | `/health`  | Model info and eval metrics    |
| POST   | `/predict` | Submit loan application, get prediction |
| GET    | `/docs`    | Interactive Swagger UI         |

### Example Request

```bash
curl -X POST "https://your-render-url.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "income": 55000,
    "loan_amount": 12000,
    "loan_term": 36,
    "interest_rate": 11.5,
    "credit_score": 640,
    "employment_years": 4,
    "num_prev_loans": 2,
    "missed_payments": 0,
    "loan_purpose": "personal"
  }'
```

### Example Response

```json
{
  "default_prediction": 0,
  "default_probability": 0.2341,
  "risk_level": "LOW",
  "model_version": "1.0.0"
}
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/baba-yega/loan-default-api.git
cd loan-default-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (run the notebook first to generate loan_model.pkl)
jupyter notebook notebook/loan_default_training.ipynb

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open docs
# http://127.0.0.1:8000/docs
```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Your API will be live at `https://loan-default-api.onrender.com`

> **Note:** Render's free tier spins down after inactivity. First request after sleep may take ~30 seconds.

---

## Tech Stack

- **Python 3.10**
- **scikit-learn** — Random Forest, StandardScaler, Pipeline
- **FastAPI** — REST API framework
- **Pydantic** — Input validation
- **joblib** — Model serialisation
- **Render** — Cloud deployment

---

## Author

**Charles Oselukwue** — AI/ML Engineering Intern  
[github.com/baba-yega](https://github.com/baba-yega) · [linkedin.com/in/charles-oselukwue](https://linkedin.com/in/charles-oselukwue)
