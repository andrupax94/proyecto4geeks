import tempfile
import torch
import soundfile as sf
import torchaudio.transforms as T
from pathlib import Path
from src.data.preprocess import Preprocess, PreprocessConfig
from app.models.model_loader import loader

# Configurar el preprocesador
SAMPLE_RATE = 16000
N_MELS = 128
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 160
PEAK_TARGET = 0.99

pp_config = PreprocessConfig(
    sample_rate=SAMPLE_RATE,
    target_duration=None,
    normalize_peak=True,
    peak_target=PEAK_TARGET,
    n_mels=N_MELS,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    save_audio=False,
    save_mel=False,
    save_mfcc=False,
    save_waveform=False,
    augment_inference=False,
)

pp = Preprocess(config=pp_config)
device = pp.device

def load_and_preprocess(audio_path: Path, augment: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel, mfcc = pp.process_audio_file(audio_path, augment_inference=augment)
    audio_np, sr = sf.read(str(audio_path))
    waveform = torch.tensor(audio_np, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE).to(device)
        waveform = resampler(waveform)
    peak = waveform.abs().max().clamp_min(1e-8)
    waveform = (waveform / peak * PEAK_TARGET).float()
    return mel.float(), mfcc.float(), waveform

@torch.inference_mode()
def _run_model(model, mode, mel, mfcc, waveform) -> torch.Tensor:
    mel_b = mel.unsqueeze(0)
    mfcc_b = mfcc.unsqueeze(0)
    waveform_b = waveform.unsqueeze(0)
    
    try:
        if mode == "mel_only":
            return model(mel=mel_b)
        elif mode == "mel_mfcc":
            return model(mel=mel_b, mfcc=mfcc_b)
        elif mode == "mel_waveform":
            try:
                return model(mel=mel_b, waveform=waveform_b)
            except TypeError:
                return model(mel_b, waveform_b)
    except Exception as e:
        print(f"⚠️ Error ejecutando modelo en modo {mode}: {e}")
        return model(mel_b)

async def predict_audio(file, version_bin: int = 3, version_specific: int = 6):
    # Guardar archivo subido en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        content = await file.read()
        temp.write(content)
        temp_path = Path(temp.name)

    try:
        # ── Etapa 1: binario ─────────────────────────────────────────────────
        # Obtener modelo dinámicamente según versión
        model_bin, mode_bin, idx2label_bin = loader.get_binary_model(version_bin)
        
        mel, mfcc, waveform = load_and_preprocess(temp_path, augment=True)
        logits_bin = _run_model(model_bin, mode_bin, mel, mfcc, waveform)
        
        if logits_bin.shape[-1] == 1:
            probs_bin = torch.sigmoid(logits_bin).squeeze(0).cpu()
            conf = float(probs_bin[0])
            is_alertable = conf > 0.5
            bin_pred_conf = conf if is_alertable else 1 - conf
            binary_top_k = [
                {"label": "alertable", "confidence": conf},
                {"label": "no_alertable", "confidence": 1 - conf}
            ]
        else:
            probs_bin = torch.softmax(logits_bin, dim=-1).squeeze(0).cpu()
            top_probs_bin, top_idxs_bin = probs_bin.topk(2)
            bin_pred_label = idx2label_bin[int(top_idxs_bin[0])]
            bin_pred_conf = float(top_probs_bin[0])
            
            if isinstance(bin_pred_label, bool):
                is_alertable = bin_pred_label
            else:
                is_alertable = str(bin_pred_label).lower() not in ("false", "no_alertable", "no alertable", "0")
                
            binary_top_k = []
            for i, p in zip(top_idxs_bin, top_probs_bin):
                label_val = idx2label_bin[int(i)]
                label_str = "alertable" if label_val is True or str(label_val).lower() in ("true", "1") else "no_alertable"
                binary_top_k.append({"label": label_str, "confidence": float(p)})
        
        # ── Etapa 2: clasificador de clase ────────────────────────────────────
        if is_alertable:
            # Cargar modelo alertable dinámicamente
            model_specific, mode_specific, idx2label_specific = loader.get_alertable_model(version_specific)
            mel2, mfcc2, waveform2 = load_and_preprocess(temp_path, augment=False)
        else:
            # Cargar modelo no alertable dinámicamente (aquí podrías añadir version_noalert si fuera necesario)
            model_specific, mode_specific, idx2label_specific = loader.get_no_alertable_model(4)
            mel2, mfcc2, waveform2 = load_and_preprocess(temp_path, augment=True)
            
        logits2 = _run_model(model_specific, mode_specific, mel2, mfcc2, waveform2)
        probs2 = torch.softmax(logits2, dim=-1).squeeze(0).cpu()
        top_k_count = min(5, len(idx2label_specific))
        top_probs2, top_idxs2 = probs2.topk(top_k_count)
        
        pred_label = idx2label_specific[int(top_idxs2[0])]
        pred_conf = float(top_probs2[0])
        top_k_list = [
            {"label": str(idx2label_specific[int(i)]), "confidence": float(p)} 
            for i, p in zip(top_idxs2, top_probs2)
        ]
        
        return {
            "filename": file.filename,
            "is_alertable": is_alertable,
            "binary_label": "alertable" if is_alertable else "no_alertable",
            "binary_confidence": bin_pred_conf,
            "binary_top_k": binary_top_k,
            "prediction": pred_label,
            "confidence": pred_conf,
            "top_k": top_k_list,
            "versions": {
                "binary": version_bin,
                "specific": version_specific
            }
        }
    finally:
        # Limpiar archivo temporal
        if temp_path.exists():
            temp_path.unlink()
