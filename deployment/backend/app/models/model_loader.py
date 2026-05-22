import pickle
import torch
from pathlib import Path
from src.models.hybrid_cnn_v3 import ImprovedMFCCCNN

# Ruta base automática
ROOT_DIR = Path(__file__).resolve().parents[4]
print(ROOT_DIR)

FINAL_MODEL_DIR = ROOT_DIR / "models" / "final"
INTERIM_DIR = ROOT_DIR / "data" / "interim" / "processed_dataset"

# Modelo binario
MODEL_BINARY_PATH = FINAL_MODEL_DIR / "best_alertable_v3.pt"
LABEL_MAPPING_BINARY = INTERIM_DIR / "label_mapping_alertableV2.pkl"

# Modelo alertable
MODEL_ALERTABLE_PATH = FINAL_MODEL_DIR / "best_human_label_v6.pt"
LABEL_MAPPING_ALERTABLE = INTERIM_DIR / "label_mapping_human_labelV2.pkl"

# Modelo no alertable
MODEL_NO_ALERTABLE_PATH = FINAL_MODEL_DIR / "best_no_alertable_v4.pt"
LABEL_MAPPING_NO_ALERTABLE = INTERIM_DIR / "label_mapping_human_no_alertableV2.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.25

def load_label_mapping(path: Path) -> tuple[dict, dict, int]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "label2idx" in payload:
        label2idx = payload["label2idx"]
        idx2label = payload["idx2label"]
    else:
        label2idx = payload
        idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label, len(label2idx)

def detect_mode_from_state_dict(state_dict: dict) -> str:
    keys = set(state_dict.keys())
    has_mel = any(k.startswith("cnn_mel.") for k in keys)
    has_mfcc = any(k.startswith("cnn_mfcc.") for k in keys)
    has_waveform = any(k.startswith("cnn_wave.") for k in keys)
    if has_mel and has_waveform:
        return "mel_waveform"
    elif has_mel and has_mfcc:
        return "mel_mfcc"
    elif has_mel:
        return "mel_only"
    elif has_mfcc:
        return "mfcc_only"
    raise ValueError("No se encontraron keys conocidas en el state_dict.")

def load_model(model_path: Path, num_classes: int, dropout: float, device) -> tuple[ImprovedMFCCCNN, str]:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    mode = detect_mode_from_state_dict(state_dict)
    model = ImprovedMFCCCNN(num_classes=num_classes, dropout=dropout, mode=mode).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, mode

# Inicializar modelos de forma perezosa (lazy loading) o al arrancar
print("📦 Cargando modelos y mappings en memoria...")
try:
    label2idx_bin, idx2label_bin, num_classes_bin = load_label_mapping(LABEL_MAPPING_BINARY)
    model_bin, mode_bin = load_model(MODEL_BINARY_PATH, num_classes_bin, DROPOUT, device)
    
    label2idx_alert, idx2label_alert, num_classes_alert = load_label_mapping(LABEL_MAPPING_ALERTABLE)
    model_alert, mode_alert = load_model(MODEL_ALERTABLE_PATH, num_classes_alert, DROPOUT, device)
    
    label2idx_noalert, idx2label_noalert, num_classes_noalert = load_label_mapping(LABEL_MAPPING_NO_ALERTABLE)
    model_noalert, mode_noalert = load_model(MODEL_NO_ALERTABLE_PATH, num_classes_noalert, DROPOUT, device)
    
    print("✅ Modelos cargados exitosamente!")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    raise e
