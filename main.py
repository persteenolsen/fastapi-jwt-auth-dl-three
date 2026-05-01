from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import numpy as np
import onnxruntime as ort
import os
from jose import jwt, JWTError
from features import FEATURES

# ----------------------------- INIT APP -----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(
    title="FastAPI + JWT + Deep Learning + House Price Prediction (v6)",
    description="01-05-2026 - FastAPI app with deep learning model serving house price predictions based on Ames Housing dataset.",
    version="6.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
    openapi_tags=[
        {
            "name": "Authorization",
            "description": "JWT Token for API access",
        }
    ]
)

# ---------------------------- ENV ----------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# ---------------------------- LOAD MODEL ----------------------------
mean = np.load("mean.npy").astype(np.float32)
std = np.load("std.npy").astype(np.float32)

# SAFE normalization (fixes divide-by-zero risk)
std_safe = np.where(std == 0, 1e-8, std)

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------------------- MODELS ----------------------------
class HouseInput(BaseModel):
    Gr_Liv_Area: float = Field(..., gt=100, lt=10000)
    Overall_Qual: float = Field(..., gt=0, lt=10)
    Year_Built: float = Field(..., gt=1800, lt=2026)
    Garage_Cars: float = Field(..., gt=0, lt=10)
    Full_Bath: float = Field(..., gt=0, lt=10)
    Bedroom_AbvGr: float = Field(..., gt=0, lt=10)
    Lot_Area: float = Field(..., gt=1000, lt=50000)

# ---------------------------- JWT ----------------------------
def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---------------------------- ROOT ----------------------------
@app.get("/")
def root():
    return {"message": "House Price Prediction API"}

# ---------------------------- LOGIN ----------------------------
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != ADMIN_USERNAME or form.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_token({"sub": form.username}),
        "token_type": "bearer"
    }

# ================================================================
# 🔧 FIXED FEATURE ENGINEERING (CRITICAL ALIGNMENT WITH TRAIN.PY)
# ================================================================
def transform(data: dict):
    data = data.copy()
    data["HouseAge"] = 2026 - data["Year_Built"]
    data["HasGarage"] = 1 if data["Garage_Cars"] > 0 else 0
    return data

# ---------------------------- PREDICT ----------------------------
@app.post("/predict")
def predict(input_data: HouseInput, token: str = Depends(oauth2_scheme)):

    verify_token(token)

    # -------------------------- FEATURE ENGINEERING --------------------------
    features_dict = transform(input_data.dict())

    # -------------------------- FEATURE VECTOR --------------------------
    x_list = [features_dict[f] for f in FEATURES]
    x = np.array([x_list], dtype=np.float32)

    # -------------------------- NORMALIZATION --------------------------
    x = (x - mean) / std_safe

    # -------------------------- ONNX PREDICTION --------------------------
    pred_log = session.run(
        [output_name],
        {input_name: x.astype(np.float32)}
    )[0]

    # SAFE OUTPUT HANDLING (fix shape issues)
    price = float(np.expm1(pred_log).reshape(-1)[0])

    # -------------------------- CLAMP --------------------------
    max_price = 755000
    min_price = 50000

    price = max(min(price, max_price), min_price)

    return {"predicted_price": price}