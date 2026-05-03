import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from features import FEATURES

# =====================================================
# SETTING THE RANDOM SEED FOR REPRODUCIBILITY
# =====================================================
# The random seed ensures consistent results across multiple runs of the script.
# By fixing the seed for Python, NumPy, and PyTorch, we guarantee that model 
# weights, data shuffling, and other random processes are the same each time.
# This is important for reproducibility and comparison of results.
# Then tuning the model architecture, learning rate and epochs will be more easy and consistent.
#  
# Example (manual predictions with fixed seed):
# 🔎 Gr_Liv_Area=900 -> $182,959
# 🔎 Gr_Liv_Area=1000 -> $187,534
# 🔎 Gr_Liv_Area=1100 -> $192,223
# 🔎 Gr_Liv_Area=1200 -> $197,028

import random

# Set the random seed for reproducibility
seed_value = 42  # You can choose any integer you like

# For Python random module
random.seed(seed_value)

# For NumPy
np.random.seed(seed_value)

# For PyTorch CPU
torch.manual_seed(seed_value)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def build_features(row):
    row = row.copy()
    row["HouseAge"] = 2026 - row["Year_Built"]
    row["HasGarage"] = 1 if row["Garage_Cars"] > 0 else 0
    return row


def make_input(row, scaler):
    x = np.array([row[f] for f in FEATURES], dtype=np.float32)
    x_scaled = (x - scaler.mean_) / scaler.scale_
    return x_scaled.astype(np.float32)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("AmesHousing.csv")

df = df.rename(columns={
    "Gr Liv Area": "Gr_Liv_Area",
    "Overall Qual": "Overall_Qual",
    "Year Built": "Year_Built",
    "Garage Cars": "Garage_Cars",
    "Full Bath": "Full_Bath",
    "Bedroom AbvGr": "Bedroom_AbvGr",
    "Lot Area": "Lot_Area"
})

df = df.apply(build_features, axis=1)
df = df[FEATURES + ["SalePrice"]].dropna()

# =====================================================
# CORRELATION CHECK
# =====================================================
print("\n📊 Correlation with SalePrice:")
print(df[FEATURES + ["SalePrice"]].corr()["SalePrice"].sort_values())

# =====================================================
# LINEAR BASELINE
# =====================================================
print("\n📏 Linear Regression baseline:")
linreg = LinearRegression()
linreg.fit(df[FEATURES], df["SalePrice"])

coef = dict(zip(FEATURES, linreg.coef_))
print("Gr_Liv_Area coefficient:", coef["Gr_Liv_Area"])

# =====================================================
# TARGET
# =====================================================
y = np.log1p(df["SalePrice"].values.astype(np.float32))

# =====================================================
# FEATURES
# =====================================================
X = df[FEATURES].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save("mean.npy", scaler.mean_)
np.save("std.npy", scaler.scale_)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
split = int(0.8 * len(X_scaled))

X_train = torch.tensor(X_scaled[:split], dtype=torch.float32)
y_train = torch.tensor(y[:split], dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_scaled[split:], dtype=torch.float32)
y_test = torch.tensor(y[split:], dtype=torch.float32).view(-1, 1)

# =====================================================
# MODEL DEFINITION
# =====================================================
class HouseModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            
            # Reduced hidden layer size to prevent overfitting, but still allows learning complex patterns
            nn.Linear(n_features, 4),
            nn.ReLU(),
           
            # Removed second hidden layer to simplify the model and reduce overfitting risk. 
            # The first layer can still capture non-linear relationships.
            
            # Output layer remains the same to predict the log price
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)

model = HouseModel(len(FEATURES))

# =====================================================
# TRAIN SETUP
# =====================================================
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
     
     # Higher LR can cause divergence, but too low may be slow. 0.004 is a good balance for this model
     lr=0.005,          
     weight_decay=1e-4
)

# =====================================================
# TRAIN LOOP
# =====================================================
# Increased epochs to allow the model to learn better, but with the simplified architecture 
# and regularization, it should not overfit
epochs = 1000

for epoch in range(epochs):
    model.train()

    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# =====================================================
# EVALUATION
# =====================================================
model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(X_test), y_test)

print("\n📊 Final Test Loss:", test_loss.item())

# =====================================================
# EXPORT ONNX
# =====================================================
dummy = torch.randn(1, len(FEATURES), dtype=torch.float32)

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)

print("\n✅ model.onnx exported")
print("✅ mean.npy + std.npy saved")

# =====================================================
# TEST CASE 1: PYTORCH
# =====================================================
print("\n🔎 Manual predictions (PyTorch):")

base = {
    "Gr_Liv_Area": 900,
    "Overall_Qual": 7,
    "Year_Built": 2005,
    "Garage_Cars": 2,
    "Full_Bath": 2,
    "Bedroom_AbvGr": 3,
    "Lot_Area": 8000
}

test_values = [900, 1000, 1100, 1200]

model.eval()
with torch.no_grad():
    for val in test_values:
        row = base.copy()
        row["Gr_Liv_Area"] = val

        row = build_features(row)

        x_scaled = make_input(row, scaler)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        pred = model(x_tensor).item()
        price = np.expm1(pred)

        print(f"Gr_Liv_Area={val} -> ${price:,.0f}")

# =====================================================
# TEST CASE 2: ONNX VALIDATION
# =====================================================
print("\n🔎 ONNX verification:")

import onnxruntime as ort
ort_session = ort.InferenceSession("model.onnx")

model.eval()
with torch.no_grad():
    for val in test_values:
        row = base.copy()
        row["Gr_Liv_Area"] = val

        row = build_features(row)

        x_scaled = make_input(row, scaler)

        torch_pred = model(torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)).item()

        ort_out = ort_session.run(None, {"input": x_scaled.reshape(1, -1)})
        onnx_pred = ort_out[0][0][0]

        print(f"{val}: diff={abs(torch_pred - onnx_pred):.8f}")