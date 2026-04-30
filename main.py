from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import numpy as np
import onnxruntime as ort
import os
from jose import jwt, JWTError
from features import FEATURES, transform

# ----------------------------- INIT APP -----------------------------
# OAuth2 password bearer token definition
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# FastAPI initialization
app = FastAPI(
    title="FastAPI + JWT + Deep Learning + House Price Prediction (v6)",
    description="30-04-2026 - FastAPI app with deep learning model serving house price predictions based on Ames Housing dataset.",
    version="6.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
    openapi_tags=[  # Adding a tag for the JWT token
        {
            "name": "Authorization",
            "description": "JWT Token for API access",
        }
    ]
)

# ---------------------------- ENVIRONMENT VARIABLES ----------------------------
load_dotenv()

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Admin credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# ---------------------------- LOAD MEAN, STD, AND MODEL ----------------------------
mean = np.load("mean.npy")
std = np.load("std.npy")

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

# ---------------------------- JWT HANDLING ----------------------------

def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---------------------------- ROOT ENDPOINT ----------------------------
@app.get("/")
def root():
    return {"message": "House Price Prediction API"}

# ---------------------------- LOGIN ENDPOINT ----------------------------
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != ADMIN_USERNAME or form.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_token({"sub": form.username}),
        "token_type": "bearer"
    }

# ---------------------------- PREDICT ENDPOINT ----------------------------
# ---------------------------- PREDICT ENDPOINT ----------------------------
@app.post("/predict")
def predict(input_data: HouseInput, token: str = Depends(oauth2_scheme)):
    # Verify JWT token
    verify_token(token)

    # -------------------------- FEATURE ENGINEERING --------------------------
    features_dict = transform(input_data.dict())

    # -------------------------- BUILD FEATURE VECTOR --------------------------
    # Ensure correct order and visibility
    x_list = [features_dict[f] for f in FEATURES]

    # 🔍 Debug: print feature mapping (remove later)
    print("---- FEATURE DEBUG ----")
    for f, v in zip(FEATURES, x_list):
        print(f"{f}: {v}")
    print("------------------------")

    x = np.array([x_list], dtype=np.float32)

    # -------------------------- SAFETY CHECK --------------------------
    if len(mean) != len(FEATURES) or len(std) != len(FEATURES):
        raise HTTPException(
            status_code=500,
            detail="Mismatch between FEATURES and normalization vectors"
        )

    # 🔍 Debug: print mean/std alignment (remove later)
    print("---- NORMALIZATION DEBUG ----")
    for f, m, s in zip(FEATURES, mean, std):
        print(f"{f}: mean={m:.2f}, std={s:.2f}")
    print("-----------------------------")

    # -------------------------- NORMALIZATION --------------------------
    x = (x - mean) / std

    # -------------------------- PREDICTION --------------------------
    pred_log = session.run([output_name], {input_name: x.astype(np.float32)})[0]

    # Reverse the log transformation
    price = np.expm1(pred_log)[0][0]

    # -------------------------- CLAMPING --------------------------
    max_price = 755000
    min_price = 50000

    if price > max_price:
        price = max_price
    if price < min_price:
        price = min_price

    return {"predicted_price": float(price)}