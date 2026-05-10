"""
diagnostic_shapes.py

Diagnostica los shapes exactos que llegan de tu ProcessedAudioDataset.
Usa esto para entender qué está pasando con MEL y MFCC.
"""

import torch
import sys
from pathlib import Path

# Añade el path de tu proyecto
# sys.path.insert(0, str(Path(__file__).parent / "src"))

# Intenta importar desde tu proyecto
try:
    from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn
    from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING
    HAS_REAL_DATASET = True
except ImportError:
    print("⚠️ No se puede importar ProcessedAudioDataset")
    print("   Usando dataset dummy para demostración")
    HAS_REAL_DATASET = False
    
    from improved_dataset import create_realistic_loaders


def diagnostic_real_dataset():
    """Diagnostica tu dataset real"""
    print("=" * 80)
    print("📊 DIAGNOSTIC: ProcessedAudioDataset")
    print("=" * 80)
    print()
    
    try:
        # Crea dataset
        train_ds = ProcessedAudioDataset(
            metadata_csv=PROCESSED_METADATA,
            label_mapping_path=LABEL_MAPPING,
            split="train",
            use_mfcc=True,
            use_scalars=True,
        )
        
        print(f"✅ Dataset cargado")
        print(f"   Total samples: {len(train_ds)}")
        print(f"   Num classes: {len(train_ds.label_mapping)}")
        print()
        
        # Obtén un sample
        batch, label, filename = train_ds[0]
        
        print("📦 SAMPLE INDIVIDUAL (sin collate):")
        print(f"   Filename: {filename}")
        print(f"   Label: {label}")
        print()
        
        print("   Shapes antes de collate_fn:")
        print(f"     MEL:     {batch.get('mel').shape if batch.get('mel') is not None else None}")
        print(f"     MFCC:    {batch.get('mfcc').shape if batch.get('mfcc') is not None else None}")
        print(f"     Scalars: {batch.get('scalars').shape if batch.get('scalars') is not None else None}")
        print()
        
        print("   Estadísticas:")
        if batch.get('mel') is not None:
            print(f"     MEL mean:     {batch['mel'].mean():.4f} ± {batch['mel'].std():.4f}")
        if batch.get('mfcc') is not None:
            print(f"     MFCC mean:    {batch['mfcc'].mean():.4f} ± {batch['mfcc'].std():.4f}")
        if batch.get('scalars') is not None:
            print(f"     Scalars mean: {batch['scalars'].mean():.4f} ± {batch['scalars'].std():.4f}")
        print()
        
        # DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            train_ds,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=crnn_collate_fn,
        )
        
        batch, labels, filenames = next(iter(loader))
        
        print("📦 BATCH (después de collate_fn):")
        print(f"   Batch size: 4")
        print()
        
        print("   Shapes DESPUÉS de collate_fn:")
        print(f"     MEL:     {batch.get('mel').shape if batch.get('mel') is not None else None}")
        print(f"     MFCC:    {batch.get('mfcc').shape if batch.get('mfcc') is not None else None}")
        print(f"     Scalars: {batch.get('scalars').shape if batch.get('scalars') is not None else None}")
        print(f"     Labels:  {labels.shape}")
        print()
        
        print("   Estadísticas:")
        if batch.get('mel') is not None:
            print(f"     MEL mean:     {batch['mel'].mean():.4f} ± {batch['mel'].std():.4f}")
        if batch.get('mfcc') is not None:
            print(f"     MFCC mean:    {batch['mfcc'].mean():.4f} ± {batch['mfcc'].std():.4f}")
        if batch.get('scalars') is not None:
            print(f"     Scalars mean: {batch['scalars'].mean():.4f} ± {batch['scalars'].std():.4f}")
        print()
        
        # Test forward pass
        from optimized_crnn_model_v2 import OptimizedAudioCNN
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = OptimizedAudioCNN(
            num_classes=len(train_ds.label_mapping),
            use_mfcc=True,
            use_scalars=True,
            scalar_dim=len(train_ds.available_scalar_columns) if hasattr(train_ds, 'available_scalar_columns') else 5,
        ).to(device)
        
        print(f"🧠 MODEL: OptimizedAudioCNN v2")
        print(f"   Parámetros: {model.num_params:,}")
        print()
        
        # Forward
        model.debug_shapes = True
        
        mel = batch['mel'].to(device)
        mfcc = batch.get('mfcc')
        if mfcc is not None:
            mfcc = mfcc.to(device)
        scalars = batch.get('scalars')
        if scalars is not None:
            scalars = scalars.to(device)
        
        print(f"🔄 FORWARD PASS:")
        try:
            output = model(mel=mel, mfcc=mfcc, scalars=scalars)
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ SUCCESS!")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def diagnostic_dummy_dataset():
    """Diagnostica con dataset dummy"""
    print("=" * 80)
    print("📊 DIAGNOSTIC: Dummy Dataset (Realistic)")
    print("=" * 80)
    print()
    
    train_ds, test_ds, train_loader, test_loader = create_realistic_loaders(
        num_train=100,
        num_test=20,
        batch_size=4,
    )
    
    print(f"✅ Dataset cargado")
    print(f"   Total train: {len(train_ds)}")
    print(f"   Total test: {len(test_ds)}")
    print()
    
    batch, labels, filenames = next(iter(train_loader))
    
    print("📦 BATCH:")
    print(f"   MEL shape:     {batch['mel'].shape}")
    print(f"   MFCC shape:    {batch['mfcc'].shape}")
    print(f"   Scalars shape: {batch['scalars'].shape}")
    print(f"   Labels shape:  {labels.shape}")
    print()
    
    # Test forward
    from optimized_crnn_model_v2 import OptimizedAudioCNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OptimizedAudioCNN(
        num_classes=10,
        use_mfcc=True,
        use_scalars=True,
        scalar_dim=5,
    ).to(device)
    
    print(f"🧠 MODEL: OptimizedAudioCNN v2")
    print(f"   Parámetros: {model.num_params:,}")
    print()
    
    model.debug_shapes = True
    
    mel = batch['mel'].to(device)
    mfcc = batch['mfcc'].to(device)
    scalars = batch['scalars'].to(device)
    
    print(f"🔄 FORWARD PASS:")
    try:
        output = model(mel=mel, mfcc=mfcc, scalars=scalars)
        print(f"\n   ✅ Output shape: {output.shape}")
        print(f"   ✅ SUCCESS!")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")


def main():
    print()
    
    if HAS_REAL_DATASET:
        diagnostic_real_dataset()
    else:
        diagnostic_dummy_dataset()
    
    print()
    print("=" * 80)
    print("💡 RECOMENDACIONES:")
    print("=" * 80)
    print()
    print("Si los shapes no coinciden:")
    print("  1. Verifica tu audio_dataset.py")
    print("  2. Revisa la función crnn_collate_fn")
    print("  3. Usa optimized_crnn_model_v2 (es robusto a shapes inesperados)")
    print()


if __name__ == "__main__":
    main()
