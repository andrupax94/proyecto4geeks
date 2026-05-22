import torch

from src.models.hybrid_cnn_v2 import HybridCNN

MODEL_PATH = "models/final/model_total_V4_32.pt"

device = torch.device("cpu")

model = HybridCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()