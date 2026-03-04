#Using the model
import numpy as np, pandas as pd, joblib
pipe = joblib.load("ridge_hic_model.joblib")
X = np.load("combined_features.npy")             # (N, 1280)
y = pd.read_csv("labels.csv")["hic_min"].values  # (N,)
y_pred = pipe.predict(X)

#Visualize predicted values vs actual values of the model
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y, y_pred, s=16, alpha=0.7)
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("Actual HIC (min)"); plt.ylabel("Predicted HIC (min)")
plt.title("Predicted vs Actual (Ridge)")
plt.tight_layout(); plt.savefig("plot_pred_vs_actual.png", dpi=200)

plt.figure(figsize=(5,3.2))
res = y_pred - y
plt.hist(res, bins=30, alpha=0.8)
plt.xlabel("Residual (pred - actual) [min]"); plt.ylabel("Count")
plt.title("Residuals")
plt.tight_layout(); plt.savefig("plot_residuals.png", dpi=200)
