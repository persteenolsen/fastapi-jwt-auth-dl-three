# 🏠 v6 - House Price Prediction API (FastAPI + PyTorch + JWT + Ames Dataset)

Last updated 

- 03-05-2026

A production-style machine learning backend system that predicts house prices using a PyTorch neural network trained on the Ames Housing dataset and served through a secure FastAPI API with JWT authentication.

This project demonstrates a full ML engineering pipeline: data preprocessing → feature engineering → model training → ONNX export → secure API inference.

# 👨‍💻 Things I learned

- For Ames Housing Dataset which are messy and noisy tabular data using a Neural Network was an interesting experience

- My v7 using Linear Regression performs "better" with Ames Housing Dataset, but both models have their pros and cons 

However, I got experience with PyTorch and compared PyToch Neural Network v6 with Linear Regression v7. PyTorch would be a good choice for massive datsets and image classification

Take a look at the section Model Tuning

---

# 🚀 Features

- End-to-end ML pipeline using real-world Ames Housing dataset  
- Feature engineering (HouseAge, HasGarage derived features)  
- Robust feature alignment between training and inference  
- Input normalization for stable neural network training  
- PyTorch neural network regression model  
- Log-transformed target (log1p) for stable regression learning  
- ONNX export for fast production inference  
- Secure REST API using FastAPI  
- JWT authentication (OAuth2 password flow)  
- Named-feature JSON input (no manual ordering required)  
- Consistent preprocessing across training and inference  
- Serverless-ready deployment design (Vercel compatible)

---

# 🧱 Tech Stack

- Python 3.12  
- FastAPI  
- PyTorch  
- ONNX Runtime  
- NumPy  
- Pandas  
- scikit-learn  
- python-jose (JWT auth)  
- Uvicorn  
- joblib  

---

# 📁 Project Structure

```
.
├── main.py                 # FastAPI inference API (JWT + ONNX)
├── train.py               # Model training + ONNX export
├── features.py            # Shared feature definitions + transforms
├── AmesHousing.csv        # Dataset
├── model.onnx             # Exported inference model
├── mean.npy               # Feature normalization mean
├── std.npy                # Feature normalization std
├── requirements.txt
└── .env                   # Environment variables (not committed)
```

---

# ⚙️ Installation

```bash
git clone https://github.com/your-repo/house-price-api.git
cd house-price-api

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements/dev.txt
```

Verify setup:

```bash
python -c "import fastapi, numpy, onnxruntime; print('VENV OK')"
```

Expected output:

```
VENV OK
```

---

# 🔐 Environment Variables (.env)

```
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

ADMIN_USERNAME=admin
ADMIN_PASSWORD=password

```

---

# 🏋️ Training the Model

```bash
python train.py
```

Outputs:
- model.onnx  
- mean.npy  
- std.npy  

---

## 🔧 Model Tuning (NEW)

During development, the model was tuned to improve stability and realism of predictions.

Key tuning changes:

- Reduced a hidden layer
- Reduced hidden layer size (16 → 8 neurons)
- Lowered learning rate (0.01 → 0.004)
- Increased training epochs (500 → 1100)
- Added weight decay (L2 regularization)
- Introduced early stopping for training stability
- Improved numerical stability in normalization

Result:

- Smooth, monotonic price curves
- Stable age depreciation behavior
- More consistent size scaling
- Reduced high-range prediction jumps
- Better generalization without overfitting

---

# 🔬 Training Pipeline

- Load Ames Housing dataset  
- Feature engineering:
  - HouseAge (derived)
  - HasGarage (binary feature)
- Select final feature set  
- Normalize features (mean/std scaling)  
- Train PyTorch neural network  
- Optimize regression using log1p target  
- Export model to ONNX format  
- Save normalization parameters for inference  

---

# 🚀 Run API Locally

```bash
uvicorn main:app --reload
```

Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

# 🔐 Authentication

## Login

POST `/login`

```
username=admin
password=password
```

Response:

```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer"
}
```

## Use Token

```
Authorization: Bearer <token>
```

---

# 📡 Prediction Endpoint

POST `/predict`

## Example requests

 "Gr_Liv_Area" = 1500

```json
{
  "Gr_Liv_Area": 1500,
  "Overall_Qual": 7,
  "Year_Built": 2005,
  "Garage_Cars": 2,
  "Full_Bath": 2,
  "Bedroom_AbvGr": 3,
  "Lot_Area": 8000
}
```

## Response

```json
{
  "predicted_price": 209812.421875
}
```


"Gr_Liv_Area" = 1200 gives lower price like expected

```json
{
  "Gr_Liv_Area": 1200,
  "Overall_Qual": 7,
  "Year_Built": 2005,
  "Garage_Cars": 2,
  "Full_Bath": 2,
  "Bedroom_AbvGr": 3,
  "Lot_Area": 8000
}
```

## Response

```json
{
  "predicted_price": 192813.921875
}
```
---

## Clamping Predicted Price

To avoid unrealistic house price predictions, the model clamps the output to a maximum value of **$755,000**. After predicting, the price is reverse log-transformed and if it exceeds the cap, it's set to this maximum value.

Example:

1. Model predicts a price.
2. The price is transformed using `np.expm1()`.
3. If the price exceeds $755,000, it's clamped to that value.

This ensures predictions stay within a realistic range.

---

# 🧠 Key Design Feature

### Named-feature system (safe ML design)

- No manual feature ordering in API  
- Centralized feature definition (`features.py`)  
- Automatic transformation + alignment  
- Prevents silent prediction errors  

---

# 🔁 Training vs Inference Consistency

## Training:
- defines feature space in `features.py`
- applies normalization (mean/std)
- trains PyTorch model
- exports ONNX + normalization params  

## Inference:
- receives named JSON input  
- applies same feature transform  
- applies identical normalization  
- runs ONNX model  
- returns real-world prediction  

---

# 🧪 System Flow

1. User logs in → receives JWT  
2. Sends house feature JSON  
3. API:
   - verifies JWT  
   - builds feature vector  
   - applies normalization  
   - runs ONNX inference  
   - returns predicted price  

---

# 🧠 What this project demonstrates

- Production-style ML system architecture  
- Feature engineering for tabular regression  
- Train/inference consistency design  
- Secure API authentication (JWT)  
- ONNX-based model deployment  
- Serverless-compatible ML inference design  

---

# 👨‍💻 Author

Built as a machine learning engineering project demonstrating production-style ML system design using FastAPI, PyTorch, and ONNX for real-world deployment scenarios.