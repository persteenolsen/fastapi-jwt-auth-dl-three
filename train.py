import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from features import FEATURES

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
# MODEL (IMPROVED CAPACITY)
# =====================================================
class HouseModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
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
    lr=0.0005,          # lower LR for stability
    weight_decay=1e-4
)

# =====================================================
# TRAIN LOOP
# =====================================================
epochs = 500

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