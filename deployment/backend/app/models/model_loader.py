from pathlib import Path
import sys
import torch

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[4]

# agregar proyecto4geeks al path
sys.path.append(str(BASE_DIR))

from src.models.hybrid_cnn_v3 import ImprovedMFCCCNN

MODEL_PATH = BASE_DIR / "models" / "final" / "best_alertable_v3.pt"

device = torch.device("cpu")

mode = "mel_waveform"
NUM_CLASSES = 2

model = ImprovedMFCCCNN(
    num_classes=NUM_CLASSES,
    dropout=0.25,
    mode=mode
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()