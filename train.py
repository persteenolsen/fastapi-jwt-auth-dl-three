import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from features import FEATURES

# ---------------- LOAD DATA ----------------
df = pd.read_csv("AmesHousing.csv")

# ---------------- FEATURE ENGINEERING ----------------
df["HouseAge"] = 2026 - df["Year Built"]
df["HasGarage"] = (df["Garage Cars"] > 0).astype(int)

# rename for consistency
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

X = df[FEATURES].values.astype(np.float32)
y = np.log1p(df["SalePrice"].values.astype(np.float32))

# ---------------- NORMALIZATION (CRITICAL FIX) ----------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8  # avoid divide-by-zero

X = (X - X_mean) / X_std

# save for inference
np.save("mean.npy", X_mean)
np.save("std.npy", X_std)

# ---------------- TRAIN/TEST SPLIT ----------------
split = int(0.8 * len(X))

X_train = torch.tensor(X[:split])
y_train = torch.tensor(y[:split]).view(-1, 1)

X_test = torch.tensor(X[split:])
y_test = torch.tensor(y[split:]).view(-1, 1)

# ---------------- MODEL ----------------
class HouseModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = HouseModel(len(FEATURES))

# ---------------- TRAIN SETUP ----------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ---------------- TRAIN LOOP ----------------
epochs = 200

for epoch in range(epochs):
    model.train()

    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: {loss.item():.4f}")

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