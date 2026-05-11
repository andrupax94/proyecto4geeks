from src.models.audio_dataset import ProcessedAudioDataset
from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING

ds = ProcessedAudioDataset(
    metadata_csv=PROCESSED_METADATA,
    label_mapping_path=LABEL_MAPPING,
    split="train",
    use_mfcc=True,
    use_scalars=False,
)

print(f"Número de clases en dataset: {len(ds.label_mapping)}")
print(f"Primeras 10 clases:")
for i, (label, code) in enumerate(list(ds.label_mapping.items())[:10]):
    print(f"  {code}: {label}")