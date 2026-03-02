import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Load simplified data
# -------------------------
df = pd.read_excel("OriginalDataSimplified.xlsx", engine="openpyxl")
print("Columns found:", list(df.columns))

# Fix European decimals (e.g., "8,9")
df["HIC"] = df["HIC"].astype(str).str.replace(",", ".").astype(float)

# Clean sequences (remove spaces/hyphens just in case)
df["VH_clean"] = df["VH Protein"].str.replace(" ", "").str.replace("-", "")
df["VL_clean"] = df["VL Protein"].str.replace(" ", "").str.replace("-", "")

# -------------------------
# Load ESM-2 (650M model)
# -------------------------
model_name = "facebook/esm2_t33_650M_UR50D"
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
torch.save(vh_embeddings, "VH_embeddings.pt")
torch.save(vl_embeddings, "VL_embeddings.pt")

# Combine VH and VL embeddings
features = np.concatenate([
    vh_embeddings.numpy(),
    vl_embeddings.numpy()
], axis=1)

# Save
np.save("combined_features.npy", features)