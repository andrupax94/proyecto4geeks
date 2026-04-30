# 🛰️ Architecture — Deforestation Detection System

## 1. Overview

This project detects deforestation by comparing satellite imagery across time and identifying changes in vegetation.

The system follows a modular architecture with clear separation between:

* Data processing
* Feature engineering
* Model training & inference
* Experiment tracking
* Deployment (web app isolated)

---

## 2. High-Level Architecture

```text
        ┌────────────────────────────┐
        │   Satellite Data Sources   │
        │  (NASA, ESA, Landsat...)  │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │      Data Ingestion        │
        │   (download + storage)     │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │     Preprocessing Layer    │
        │ (cleaning, alignment, etc) │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │   Feature Engineering      │
        │   (NDVI, spectral index)   │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │        ML Models           │
        │ (U-Net / CNN / Temporal)   │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │     Evaluation Layer       │
        │ (metrics + validation)     │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │     Prediction Service     │
        │   (API / batch inference)  │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │        Web Application     │
        │   (visualization layer)    │
        └────────────────────────────┘
```

---

## 3. Data Layer

### Sources

* NASA EarthData (Landsat)
* ESA Sentinel imagery
* External datasets (e.g. forest monitoring)

### Storage Strategy

* `data/raw/` → immutable source data
* `data/interim/` → partially processed
* `data/processed/` → model-ready

### Key Decisions

* Raw data is never modified
* All transformations are reproducible via scripts in `src/data/`

---

## 4. Preprocessing Layer

Handles:

* Image alignment (multi-temporal consistency)
* Cloud filtering (if applicable)
* Resizing / normalization
* Patch extraction (tiling large satellite images)

**Output:** Clean, aligned image pairs (t1, t2)

---

## 5. Feature Engineering

Optional but critical for performance.

Examples:

* NDVI (Normalized Difference Vegetation Index)
* Spectral band combinations
* Temporal differences (Δ vegetation)

**Why it matters:**
Raw RGB is often not enough in satellite imagery.

---

## 6. Model Layer

### Core Task

Semantic segmentation of deforested areas.

### Candidate Models

* U-Net (baseline)
* DeepLabV3+
* CNN-based change detection
* Temporal models (LSTM / 3D CNN)

### Input

* Image at time T1
* Image at time T2

### Output

* Binary mask (deforestation vs no deforestation)

---

## 7. Training Pipeline

Located in:

```
src/models/train.py
```

Steps:

1. Load processed dataset
2. Apply augmentations
3. Train model
4. Save checkpoints → `models/checkpoints/`
5. Save final model → `models/final/`

---

## 8. Evaluation

Metrics:

* IoU (Intersection over Union)
* F1-score
* Precision / Recall

Artifacts stored in:

```
reports/metrics/
reports/figures/
```

---

## 9. Experiment Tracking

Experiments stored in:

```
experiments/
```

Tracks:

* Hyperparameters
* Model version
* Dataset version
* Metrics

Optional tools:

* MLflow
* Weights & Biases

---

## 10. Inference Layer

Implemented in:

```
src/models/predict.py
```

Supports:

* Batch inference
* Single image prediction

Output:

* Segmentation mask
* Overlay visualization

---

## 11. Deployment Architecture

### Separation Principle

Deployment is fully isolated from the ML core.

```
deployment/webapp/
```

### Components

#### Frontend

* Displays satellite images
* Shows deforestation overlays
* Allows user input (image upload / region selection)

#### Backend (FastAPI)

* `/predict` endpoint
* Loads trained model
* Returns predictions

---

## 12. Visualization Layer

Located in:

```
src/visualization/
```

Capabilities:

* Before vs After comparison
* Overlay masks
* Area estimation (deforested region)

---

## 13. Testing

Basic unit tests:

```
tests/
```

Focus:

* Data loading
* Preprocessing correctness
* Model input/output consistency

---

## 14. Design Principles

* **Modularity** → each layer is independent
* **Reproducibility** → pipelines are script-based
* **Scalability** → easy to swap models or datasets
* **Separation of concerns** → ML vs deployment isolated
* **Traceability** → experiments tracked

---

## 15. Future Improvements

* Add geospatial indexing (GeoTIFF support)
* Integrate real-time satellite feeds
* Deploy model with GPU inference
* Add alert system for detected deforestation
* Incorporate climate data (multimodal learning)

---

## 16. Summary

This architecture is designed to:

* Handle real-world satellite data
* Support experimentation and iteration
* Transition smoothly from research to production

It balances simplicity (for development) with extensibility (for scaling).
