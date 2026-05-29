import pickle
import torch
import os
from pathlib import Path
from src.models.hybrid_cnn_v3 import ImprovedMFCCCNN
from src.models.hybrid_cnn_v4 import HybridCNNV5

# Ruta base automática
ROOT_DIR = Path(__file__).resolve().parents[4]
FINAL_MODEL_DIR = ROOT_DIR / "models" / "final"
INTERIM_DIR = ROOT_DIR / "data" / "interim" / "processed_dataset"

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
    return "mel_only"

def load_model(model_path: Path, num_classes: int, dropout: float, device, target_type: str, version: int) -> tuple[torch.nn.Module, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    mode = detect_mode_from_state_dict(state_dict)
    
    # Determinar arquitectura
    use_v5 = False
    if target_type == "alertable" and version > 3:
        use_v5 = True
    elif target_type == "human_label" and version > 6:
        use_v5 = True
    elif target_type == "no_alertable" and version > 4:
        use_v5 = True

    if use_v5:
        # IMPORTANTE: Determinar si el modelo es binario o multiclase para HybridCNNV5
        # Miramos el tamaño de la última capa del clasificador en el state_dict
        classifier_key = "classifier.4.weight" if "classifier.4.weight" in state_dict else "classifier.3.weight"
        if classifier_key in state_dict:
            out_features = state_dict[classifier_key].shape[0]
            task = "binary" if out_features == 1 else "multiclass"
            # Si es multiclase, num_classes debe coincidir con out_features
            current_num_classes = out_features if task == "multiclass" else num_classes
        else:
            task = "binary"
            current_num_classes = num_classes

        print(f"🏗️ Cargando HybridCNNV5 | Task: {task} | Classes: {current_num_classes} | Mode: {mode}")
        model = HybridCNNV5(num_classes=current_num_classes, dropout=dropout, mode=mode, task=task).to(device)
    else:
        print(f"🏗️ Cargando ImprovedMFCCCNN | Classes: {num_classes} | Mode: {mode}")
        model = ImprovedMFCCCNN(num_classes=num_classes, dropout=dropout, mode=mode).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, mode

class ModelLoader:
    def __init__(self):
        self.device = device
        self._cache = {}

    def get_binary_model(self, version: int = 3):
        model_key = f"binary_v{version}"
        if model_key not in self._cache:
            path = FINAL_MODEL_DIR / f"best_alertable_v{version}.pt"
            if not path.exists():
                path = FINAL_MODEL_DIR / "best_alertable_v3.pt"
                version = 3
            
            mapping_path = INTERIM_DIR / "label_mapping_alertableV2.pkl"
            l2i, i2l, n_classes = load_label_mapping(mapping_path)
            model, mode = load_model(path, n_classes, DROPOUT, self.device, "alertable", version)
            self._cache[model_key] = (model, mode, i2l)
        return self._cache[model_key]

    def get_alertable_model(self, version: int = 6):
        model_key = f"alertable_v{version}"
        if model_key not in self._cache:
            path = FINAL_MODEL_DIR / f"best_human_label_v{version}.pt"
            if not path.exists():
                path = FINAL_MODEL_DIR / "best_human_label_v6.pt"
                version = 6
            
            mapping_path = INTERIM_DIR / "label_mapping_human_labelV2.pkl"
            l2i, i2l, n_classes = load_label_mapping(mapping_path)
            model, mode = load_model(path, n_classes, DROPOUT, self.device, "human_label", version)
            self._cache[model_key] = (model, mode, i2l)
        return self._cache[model_key]

    def get_no_alertable_model(self, version: int = 4):
        model_key = f"no_alertable_v{version}"
        if model_key not in self._cache:
            path = FINAL_MODEL_DIR / f"best_no_alertable_v{version}.pt"
            if not path.exists():
                path = FINAL_MODEL_DIR / "best_no_alertable_v4.pt"
                version = 4
            
            mapping_path = INTERIM_DIR / "label_mapping_human_no_alertableV2.pkl"
            l2i, i2l, n_classes = load_label_mapping(mapping_path)
            model, mode = load_model(path, n_classes, DROPOUT, self.device, "no_alertable", version)
            self._cache[model_key] = (model, mode, i2l)
        return self._cache[model_key]

loader = ModelLoader()
