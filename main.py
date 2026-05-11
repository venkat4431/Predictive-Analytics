from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random, math, datetime

app = FastAPI(title="PredictIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def seasonal_sales(month: int, base: float, noise: float = 0.15) -> float:
    seasonal = 1 + 0.3 * math.sin((month - 3) * math.pi / 6)
    return round(base * seasonal * (1 + random.uniform(-noise, noise)), 2)

def generate_time_series(months: int = 24, base: float = 50000):
    data = []
    today = datetime.date.today()
    for i in range(months, 0, -1):
        d = today - datetime.timedelta(days=i*30)
        trend = base * (1 + 0.02 * (months - i))
        value = seasonal_sales(d.month, trend)
        data.append({"date": d.strftime("%Y-%m"), "value": value,
                     "label": d.strftime("%b %Y")})
    return data

def forecast(history: list, steps: int = 6):
    vals = [h["value"] for h in history]
    avg_growth = (vals[-1] / vals[0]) ** (1 / len(vals)) if vals[0] > 0 else 1
    last_date = datetime.datetime.strptime(history[-1]["date"], "%Y-%m")
    result = []
    for i in range(1, steps+1):
        next_date = last_date + datetime.timedelta(days=i*30)
        base_val = vals[-1] * (avg_growth ** i)
        noise = random.uniform(0.96, 1.04)
        result.append({
            "date": next_date.strftime("%Y-%m"),
            "label": next_date.strftime("%b %Y"),
            "value": round(base_val * noise, 2),
            "lower": round(base_val * noise * 0.92, 2),
            "upper": round(base_val * noise * 1.08, 2),
            "is_forecast": True,
        })
    return result

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"message": "PredictIQ API running"}

@app.get("/api/sales/history")
def sales_history():
    return generate_time_series(24, 48000)

@app.get("/api/sales/forecast")
def sales_forecast():
    history = generate_time_series(24, 48000)
    return {
        "history": history,
        "forecast": forecast(history, 6),
        "model": "Seasonal Trend Decomposition",
        "accuracy": 94.2,
        "generated_at": datetime.datetime.now().isoformat(),
    }

@app.get("/api/customer/churn")
def churn_prediction():
    segments = ["Enterprise", "Mid-Market", "SMB", "Startup"]
    return [
        {
            "segment": seg,
            "churn_risk": round(random.uniform(0.05, 0.45), 3),
            "customer_count": random.randint(50, 800),
            "avg_revenue": round(random.uniform(500, 8000), 2),
            "risk_level": "High" if random.random() > 0.6 else ("Medium" if random.random() > 0.4 else "Low"),
        }
        for seg in segments
    ]

class PriceInput(BaseModel):
    airline: str
    origin: str
    destination: str
    days_before: int
    duration_hours: float
    stops: int
    travel_class: str

@app.post("/api/predict/flight-price")
def predict_flight(data: PriceInput):
    base = 180
    base += (14 - min(data.days_before, 14)) * 12
    base += data.duration_hours * 22
    base += data.stops * 45
    if data.travel_class == "Business": base *= 2.8
    elif data.travel_class == "First": base *= 4.5
    if data.airline in ["Emirates","Singapore Airlines","Qatar Airways"]: base *= 1.35
    noise = random.uniform(0.93, 1.07)
    price = round(base * noise, 2)
    return {
        "predicted_price": price,
        "confidence": round(random.uniform(0.87, 0.96), 3),
        "price_range": {"low": round(price*0.91, 2), "high": round(price*1.11, 2)},
        "recommendation": "Book Now" if data.days_before < 7 else ("Good Price" if price < 350 else "Wait for Deal"),
        "feature_importance": {
            "days_before_departure": 0.31,
            "flight_duration": 0.24,
            "airline": 0.18,
            "travel_class": 0.16,
            "stops": 0.11,
        }
    }

@app.get("/api/dashboard/kpis")
def kpis():
    return {
        "total_revenue": {"value": 2847392, "change": 12.4, "unit": "USD"},
        "avg_order_value": {"value": 284, "change": 3.8, "unit": "USD"},
        "customer_ltv": {"value": 1840, "change": 7.2, "unit": "USD"},
        "churn_rate": {"value": 4.7, "change": -1.2, "unit": "%"},
        "forecast_accuracy": {"value": 94.2, "change": 0.8, "unit": "%"},
        "models_active": {"value": 8, "change": 2, "unit": ""},
    }

@app.get("/api/models")
def list_models():
    return [
        {"name": "Sales Forecaster", "type": "Time Series", "accuracy": 94.2, "status": "active", "last_trained": "2025-05-10"},
        {"name": "Churn Predictor", "type": "Classification", "accuracy": 91.7, "status": "active", "last_trained": "2025-05-09"},
        {"name": "Price Optimizer", "type": "Regression", "accuracy": 88.4, "status": "active", "last_trained": "2025-05-08"},
        {"name": "Demand Forecaster", "type": "Time Series", "accuracy": 89.1, "status": "training", "last_trained": "2025-05-11"},
        {"name": "Anomaly Detector", "type": "Unsupervised", "accuracy": 95.6, "status": "active", "last_trained": "2025-05-07"},
    ]
