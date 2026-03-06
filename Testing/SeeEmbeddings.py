import torch

vh = torch.load("VH_embeddings.pt")
vl = torch.load("VL_embeddings.pt")

print(vh.shape)
print(vh)
