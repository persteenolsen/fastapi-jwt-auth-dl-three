import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import StandardScaler
from features import FEATURES

# ---------------- LOAD DATA ----------------
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
# Check summary statistics of the SalePrice distribution
print("\nSummary statistics for SalePrice:")
print(df['SalePrice'].describe())

# ---------------- TARGET NORMALIZATION ----------------
# Log-transform the target SalePrice to compress large values
y = np.log1p(df["SalePrice"].values.astype(np.float32))

# ---------------- FEATURE NORMALIZATION ----------------
X = df[FEATURES].values.astype(np.float32)

# Normalize the features using StandardScaler from sklearn
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