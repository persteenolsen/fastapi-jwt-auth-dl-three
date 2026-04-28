from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta

from features import FEATURES, transform


# -----------------------------
# INIT APP
# -----------------------------
# Create FastAPI application with metadata (used in Swagger docs)
app = FastAPI(
    title="FastAPI + JWT + Deep Learning + House Price Prediction (v6)",
    description="28-04-2026 - FastAPI + JWT + Deep Learning + House Price Prediction with Ames Housing Dataset - Neural Network trained by PyTorch and exported to ONNX",
    version="6.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
)


# ---------------- JWT CONFIG ----------------
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---------------- LOAD NORMALIZATION ----------------
mean = np.load("mean.npy")
std = np.load("std.npy")

# ---------------- ONNX MODEL ----------------
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------- INPUT MODEL ----------------
class HouseInput(BaseModel):
    Gr_Liv_Area: float
    Overall_Qual: float
    Year_Built: float
    Garage_Cars: float
    Full_Bath: float
    Bedroom_AbvGr: float
    Lot_Area: float

# ---------------- JWT ----------------
def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# -------- ROOT ENDPOINT --------
# Simple health/info endpoint
@app.get("/")
def root():
    return {"message": "House Price Prediction API v6 + PyTorch + ONNX"}

# ---------------- LOGIN ----------------
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):

    if form.username != "admin" or form.password != "password":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_token({"sub": form.username}),
        "token_type": "bearer"
    }

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(input_data: HouseInput, token: str = Depends(oauth2_scheme)):

    verify_token(token)

    # ---------------- RAW INPUT ----------------
    data = input_data.dict()

    # ---------------- FEATURE ENGINEERING ----------------
    features_dict = {
        "Gr_Liv_Area": data["Gr_Liv_Area"],
        "Overall_Qual": data["Overall_Qual"],
        "Year_Built": data["Year_Built"],
        "Garage_Cars": data["Garage_Cars"],
        "Full_Bath": data["Full_Bath"],
        "Bedroom_AbvGr": data["Bedroom_AbvGr"],
        "Lot_Area": data["Lot_Area"],

        "HouseAge": 2026 - data["Year_Built"],
        "HasGarage": 1 if data["Garage_Cars"] > 0 else 0
    }

    # ---------------- FEATURE ORDER (MUST MATCH TRAINING) ----------------
    x = np.array([[features_dict[f] for f in FEATURES]], dtype=np.float32)

    # ---------------- NORMALIZATION (CRITICAL FIX) ----------------
    x = (x - mean) / std

    # ---------------- ONNX INFERENCE ----------------
    pred_log = session.run([output_name], {input_name: x})[0]

    # reverse log transform
    price = np.expm1(pred_log)[0][0]

    return {
        "predicted_price": float(price)
    }