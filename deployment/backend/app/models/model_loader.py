import torch

from src.models.hybrid_cnn_v3 import ImprovedMFCCCNN

MODEL_PATH = "models/final/model_total_V4_32.pt"

mode = "mel_waveform"
NUM_CLASSES = 2

device = torch.device("cpu")

model = ImprovedMFCCCNN(num_classes=NUM_CLASSES, dropout=0.25, mode=mode).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()