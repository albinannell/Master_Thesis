#Checking info about the created model

import joblib

model = joblib.load("ridge_hic_model.joblib")
print(model)                   # Shows the Pipeline and parameters
print(model.named_steps)       # Dict-like access to steps
print(type(model.named_steps["ridge"]))  # Ridge class

import numpy as np

# Load saved features and labels (optional)
X = np.load("combined_features.npy")            # shape (N, 1280) if ESM2-150M
pred = model.predict(X)                         # shape (N,)
print(pred[:5])                                 # first 5 predictions (minutes)

#Direction and magnitue of each featrure after scaling
ridge = model.named_steps["ridge"]
print("Intercept (minutes):", ridge.intercept_)    # scalar

coef = ridge.coef_                                  # shape (n_features,)
print("Coefficient vector shape:", coef.shape)
print("First 10 coefficients:", coef[:10])

#Norms and basic stats
import numpy as np

coef = ridge.coef_
print("L2-norm of coefficients:", np.linalg.norm(coef))
print("Mean |coef|:", np.mean(np.abs(coef)))
print("Max |coef|:", np.max(np.abs(coef)))
print("Index of max |coef|:", np.argmax(np.abs(coef)))

#top‑k positive/negative coefficients
k = 10
idx_sorted = np.argsort(coef)        # ascending
print("Most negative:", idx_sorted[:k])
print("Most positive:", idx_sorted[-k:])
print("Top positive coeff values:", coef[idx_sorted[-k:]])
print("Top negative coeff values:", coef[idx_sorted[:k]])

