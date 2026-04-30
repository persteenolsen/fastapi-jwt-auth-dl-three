import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import StandardScaler
from features import FEATURES

# -------------------- LOAD DATA --------------------
df = pd.read_csv("AmesHousing.csv")

# ---------------- FEATURE ENGINEERING ----------------
df["HouseAge"] = 2026 - df["Year Built"]
df["HasGarage"] = (df["Garage Cars"] > 0).astype(int)

# Rename for consistency
df = df.rename(columns={
    "Gr Liv Area": "Gr_Liv_Area",
    "Overall Qual": "Overall_Qual",
    "Year Built": "Year_Built",
    "Garage Cars": "Garage_Cars",
    "Full Bath": "Full_Bath",
    "Bedroom AbvGr": "Bedroom_AbvGr",
    "Lot Area": "Lot_Area"
})

# ---------------- SELECT FEATURES ----------------
df = df[FEATURES + ["SalePrice"]].dropna()

# ---------------- CHECK DISTRIBUTION OF SALEPRICE ----------------
print("\nSummary statistics for SalePrice:")
print(df['SalePrice'].describe())

# ---------------- TARGET NORMALIZATION ----------------
y = np.log1p(df["SalePrice"].values.astype(np.float32))

# ---------------- FEATURE NORMALIZATION ----------------
X = df[FEATURES].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the mean and std values for inference
np.save("mean.npy", scaler.mean_)
np.save("std.npy", scaler.scale_)

# ---------------- TRAIN/TEST SPLIT ----------------
split = int(0.8 * len(X_scaled))
X_train = torch.tensor(X_scaled[:split])
y_train = torch.tensor(y[:split]).view(-1, 1)
X_test = torch.tensor(X_scaled[split:])
y_test = torch.tensor(y[split:]).view(-1, 1)

# ---------------- MODEL ----------------
class HouseModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),  # Using smaller number of units
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = HouseModel(len(FEATURES))

# ---------------- TRAIN SETUP ----------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAIN LOOP ----------------
epochs = 200
mono_loss_weight = 0.05  # Lowering monotonic loss weight

for epoch in range(epochs):
    model.train()

    pred = model(X_train)
    base_loss = loss_fn(pred, y_train)
    mono_loss = 0  # For now no monotonic constraint

    total_loss = base_loss + mono_loss * mono_loss_weight
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: base_loss={base_loss.item():.4f}, mono_loss={mono_loss.item() if mono_loss != 0 else 0:.4f}, total_loss={total_loss.item():.4f}")

# ---------------- EVALUATION ----------------
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)

print("\n📊 Final Test Loss:", test_loss.item())

# ---------------- EXPORT ONNX ----------------
dummy = torch.randn(1, len(FEATURES))

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)

print("\n✅ Training complete")
print("✅ model.onnx exported")
print("✅ mean.npy + std.npy saved")