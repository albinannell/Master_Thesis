import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Load simplified data
# -------------------------
df = pd.read_excel("OriginalData.xlsx", engine="openpyxl")
print("Columns found:", list(df.columns))

# --- Clean HIC robustly (coerce errors to NaN) ---
hic_raw = (
    df["HIC retention time (min)"]
      .astype(str)
      .str.replace(",", ".", regex=False)
      .str.strip()
)

# Convert to numeric; invalid strings -> NaN
hic_num = pd.to_numeric(hic_raw, errors="coerce")

# Build a keep mask: valid HIC and non-empty VH/VL
vh_clean = df["VH Protein"].astype(str).str.replace(" ", "").str.replace("-", "")
vl_clean = df["VL Protein"].astype(str).str.replace(" ", "").str.replace("-", "")

keep_mask = (~hic_num.isna()) & (vh_clean.str.len() > 0) & (vl_clean.str.len() > 0)

dropped = (~keep_mask).sum()
if dropped > 0:
    print(f"Filtering: dropping {dropped} row(s) with non-numeric HIC or empty VH/VL.")

# Filter the dataframe & assign cleaned columns
df = df.loc[keep_mask].copy()
df["HIC retention time (min)"] = hic_num.loc[keep_mask].astype(float)
df["VH_clean"] = vh_clean.loc[keep_mask]
df["VL_clean"] = vl_clean.loc[keep_mask]

# Optional - For auditability in your thesis, you can save which rows were removed
dropped_rows = ~keep_mask
if dropped_rows.any():
    cols_to_show = ["Clone name", "HIC retention time (min)", "VH Protein", "VL Protein"]
    df_original = pd.read_excel("OriginalData.xlsx", engine="openpyxl")
    pd.DataFrame(df_original.loc[dropped_rows, cols_to_show]).to_csv(
        "dropped_rows_due_to_bad_HIC_or_sequence.csv", index=False
    )
    print("Saved dropped row details → dropped_rows_due_to_bad_HIC_or_sequence.csv")

# -------------------------
# Load ESM-2 (650M model)
# -------------------------
model_name = "facebook/esm2_t30_150M_UR50D"     # Later: "facebook/esm2_t33_650M_UR50D" or keep 150M if performance is similar
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cpu")
model.to(device)

# -------------------------
# Simple embedding function
# -------------------------
@torch.no_grad()
def embed_seq(seq):
    """Embed a single sequence using mean of token embeddings."""
    tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = model(**tokens)

    # Remove [CLS]/BOS and [EOS] tokens (position 1:-1)
    hidden = out.last_hidden_state[:, 1:-1, :]
    attn = tokens["attention_mask"][:, 1:-1]

    # Mean pooling
    embed = (hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1).unsqueeze(-1)
    return embed.cpu().numpy().flatten()

# -------------------------
# Compute embeddings
# -------------------------
vh_embeddings = []
vl_embeddings = []

for _, row in df.iterrows():
    vh_embeddings.append(embed_seq(row["VH_clean"]))
    vl_embeddings.append(embed_seq(row["VL_clean"]))


vh_embeddings = torch.from_numpy(np.stack(vh_embeddings, axis=0))  # (N, 1280)
vl_embeddings = torch.from_numpy(np.stack(vl_embeddings, axis=0))  # (N, 1280)


print("VH embedding shape:", vh_embeddings.shape)
print("VL embedding shape:", vl_embeddings.shape)

# Save if you want
#torch.save(vh_embeddings, "VH_embeddings.pt")
#torch.save(vl_embeddings, "VL_embeddings.pt")

# Combine VH and VL embeddings
features = np.concatenate([
    vh_embeddings.numpy(),
    vl_embeddings.numpy()
], axis=1)

# Save
np.save("combined_features.npy", features)

# Labels (HIC minutes)
y = df["HIC retention time (min)"].values.astype(float)

# Features are already in 'features' (shape = [N, 2560]) from your code above.
X = features
print("X shape:", X.shape, "y shape:", y.shape)

# =========================
# Save features + labels
# =========================

# 1) Save feature matrix
np.save("combined_features.npy", features)
print("Saved → combined_features.npy", features.shape)

# 2) Build labels.csv aligned to X rows
labels_df = pd.DataFrame({
    "hic_min": df["HIC retention time (min)"].values.astype(float)
})
if "Clone name" in df.columns:
    labels_df["Clone name"] = df["Clone name"].values
labels_df.to_csv("labels.csv", index=False)
print("Saved → labels.csv", labels_df.shape)

print(f"\nDataset ready → X: {X.shape}, y: {y.shape}\n")


# =========================
# Ridge Regression (CV baseline) with tidy output
# =========================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# (Optional) Spearman correlation
try:
    from scipy.stats import spearmanr
    HAS_SPEARMAN = True
except Exception:
    HAS_SPEARMAN = False

# 1) Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Pipeline: Standardize → Ridge
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("ridge", Ridge(random_state=42))
])

# 3) Hyperparameter grid + robust CV
alphas = np.logspace(-3, 3, 13)   # 1e-3 … 1e3
param_grid = {"ridge__alpha": alphas}
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_mean_absolute_error",  # optimize MAE in CV
    n_jobs=-1,
    refit=True
)
gs.fit(X_tr, y_tr)

# 4) Evaluate on held-out test
y_pred = gs.predict(X_te)
mae  = mean_absolute_error(y_te, y_pred)
r2   = r2_score(y_te, y_pred)
if HAS_SPEARMAN:
    rho, rho_p = spearmanr(y_te, y_pred)
else:
    rho, rho_p = (None, None)

# 5) Tidy printout
print("=== Ridge Regression (CV Baseline) ===")
print(f"Samples: train={len(y_tr)}, test={len(y_te)}, features={X.shape[1]}")
print(f"Best alpha (CV): {gs.best_params_['ridge__alpha']:.6g}")
print(f"CV best score (−MAE): {-gs.best_score_:.4f}  "
      f"(lower is better, RepeatedKFold)")
print("\n-- Held-out Test Metrics --")
print(f"MAE (min): {mae:.4f}")
print(f"R^2:       {r2:.4f}")
if HAS_SPEARMAN:
    print(f"Spearman ρ: {rho:.4f}  (p={rho_p:.3g})")
else:
    print("Spearman ρ: (scipy not installed)")

# 6) Save the fitted model (optional)
try:
    import joblib
    joblib.dump(gs.best_estimator_, "ridge_hic_model.joblib")
    print("\nSaved → ridge_hic_model.joblib")
except Exception:
    print("\nModel not saved (install 'joblib' to enable saving).")