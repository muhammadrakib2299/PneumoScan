# PneumoScan — Project Plan

## Project Title

**PneumoScan: Multi-Class Pneumonia Detection from Chest X-Rays Using Transfer Learning and Explainable AI**

A deep learning system that classifies chest X-rays into Normal, Bacterial Pneumonia, and Viral Pneumonia using transfer learning — with explainable AI (Grad-CAM + LIME) to help clinicians understand and trust AI-assisted diagnoses.

---

## 1. Problem Statement

- Pneumonia kills ~2.5 million people/year globally, ~700,000 are children under 5
- Bangladesh has roughly 1 radiologist per 100,000+ people
- Rural clinics often have X-ray machines but no trained radiologist on-site
- Misdiagnosis rates are high during high-volume shifts
- Distinguishing bacterial vs viral pneumonia matters — bacterial needs antibiotics, viral does not

**This project builds an AI-assisted screening tool that reads chest X-rays in <1 second and provides explainable predictions with confidence scores.**

---

## 2. Target Users & Benefits

| User Type | Problem Solved | Benefit |
|-----------|---------------|---------|
| Radiologists | Fatigue-related missed diagnoses | AI second opinion reduces error rate |
| Rural clinic doctors | No radiologist available | AI screening where specialists don't exist |
| Hospital administrators | Slow triage, high costs | Faster triage = fewer deaths, lower costs |
| Public health researchers | Manual epidemiological tracking | Automated pneumonia pattern detection |
| Medical AI researchers | Lack of reproducible benchmarks | Open-source, reproducible pipeline |
| Emergency departments | Delayed critical diagnoses | Instant prioritization of severe cases |

### Real-World Impact

- **Speed**: AI reads an X-ray in <1 second vs 5–15 minutes for a human
- **Equity**: Specialist-level screening in places that can't afford radiologists
- **Treatment accuracy**: Bacterial vs viral distinction guides correct treatment (antibiotics vs supportive care)
- **Education**: Grad-CAM heatmaps teach junior doctors what pneumonia patterns look like
- **Scalability**: One model can serve unlimited clinics simultaneously

---

## 3. Dataset

**Source**: Kaggle Chest X-Ray Images (Pneumonia) — [Paul Mooney's Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | BACTERIAL | VIRAL | Total |
|-------|--------|-----------|-------|-------|
| Train | ~1,341 | ~2,530 | ~1,345 | ~5,216 |
| Val | 8 | 8 | 8 | 24 |
| Test | 234 | 242 | 148 | 624 |

### Class Extraction Strategy

The original dataset labels pneumonia images with filename prefixes:
- `person1_bacteria_1.jpeg` → **Bacterial Pneumonia**
- `person1_virus_1.jpeg` → **Viral Pneumonia**
- NORMAL folder → **Normal**

### Data Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance (bacterial > viral > normal) | Class weights in loss function + targeted augmentation |
| Tiny validation set (only 24 images) | Merge val into train, use stratified K-Fold CV |
| Varying image sizes | Resize all to 224×224 with aspect ratio preservation |
| Limited dataset size (~5,864 total) | Aggressive augmentation + transfer learning |

### Augmentation Pipeline

```
- Random horizontal flip
- Random rotation (±15°)
- Random zoom (±10%)
- Random brightness/contrast adjustment (±15%)
- Random translation (±10%)
```

---

## 4. Model Architecture

### 4.1 Model Lineup (5 + Ensemble)

| # | Model | Parameters | ImageNet Top-1 | Why Include |
|---|-------|-----------|----------------|-------------|
| 1 | Custom CNN (baseline) | ~500K | N/A | Control group — proves transfer learning helps |
| 2 | ResNet-50 | 25.6M | 76.1% | Classic, well-understood, strong baseline |
| 3 | EfficientNet-B0 | 5.3M | 77.1% | Best accuracy-per-parameter ratio |
| 4 | DenseNet-121 | 8.0M | 74.4% | CheXNet architecture — medical imaging standard |
| 5 | MobileNetV2 | 3.4M | 71.3% | Edge deployment feasibility proof |
| 6 | Ensemble (Top-3) | Combined | — | Voting/averaging of best 3 models |

### 4.2 Training Strategy

**Phase 1 — Feature Extraction (Frozen Base)**
- Freeze all pretrained layers
- Train only the classification head
- 10 epochs, learning rate: 1e-3
- Purpose: Adapt the head to chest X-ray features

**Phase 2 — Fine-Tuning (Partial Unfreeze)**
- Unfreeze top 30% of base model layers
- 20 epochs, learning rate: 1e-5
- Purpose: Adapt high-level features to medical domain

### 4.3 Classification Head Architecture

```
GlobalAveragePooling2D
→ Dense(256, relu) + BatchNorm + Dropout(0.3)
→ Dense(128, relu) + BatchNorm + Dropout(0.3)
→ Dense(3, softmax)    # 3-class output
```

### 4.4 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Categorical crossentropy (with class weights) |
| LR Schedule | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | patience=7, restore_best_weights=True |
| Validation | Stratified 5-Fold CV (merge original val into train) |

### 4.5 Ensemble Strategy

- Select top-3 performing models by validation accuracy
- **Soft voting**: Average the probability outputs from all 3 models
- **Weighted voting**: Weight by each model's validation AUC score
- Compare both approaches and report the better one

---

## 5. Evaluation Framework

### 5.1 Metrics

| Metric | Why |
|--------|-----|
| **Accuracy** | Overall correctness |
| **Precision (per class)** | How many flagged cases are truly positive |
| **Recall / Sensitivity (per class)** | How many actual cases are caught — critical for medical screening |
| **F1-Score (per class)** | Harmonic mean of precision and recall |
| **AUC-ROC (One-vs-Rest)** | Performance across all thresholds |
| **AUC-PR (Precision-Recall)** | Better than ROC for imbalanced classes |
| **Confusion Matrix** | Per-class error analysis |
| **Specificity** | True negative rate — important for "Normal" class |
| **Cohen's Kappa** | Agreement beyond chance — accounts for class imbalance |

### 5.2 Multi-Threshold Analysis

- Plot precision, recall, and F1 at confidence thresholds from 0.3 to 0.9
- Determine optimal threshold for clinical use (maximize sensitivity while keeping specificity ≥ 90%)
- Report the "clinical operating point" per class

### 5.3 Cross-Model Comparison

Generate a single comparison table:

```
| Model          | Accuracy | F1 (Macro) | AUC-ROC | Params  | Inference (ms) |
|----------------|----------|------------|---------|---------|----------------|
| Custom CNN     | ...      | ...        | ...     | 500K    | ...            |
| ResNet-50      | ...      | ...        | ...     | 25.6M   | ...            |
| EfficientNet   | ...      | ...        | ...     | 5.3M    | ...            |
| DenseNet-121   | ...      | ...        | ...     | 8.0M    | ...            |
| MobileNetV2    | ...      | ...        | ...     | 3.4M    | ...            |
| Ensemble       | ...      | ...        | ...     | —       | ...            |
```

---

## 6. Explainability (XAI)

### 6.1 Grad-CAM

- Generate heatmap overlays for all 3 classes across all 5 models
- Show which lung regions the model focuses on
- Compare: Do different architectures attend to different areas?
- Generate a grid visualization: `[Original | Normal-CAM | Bacterial-CAM | Viral-CAM]`

### 6.2 LIME (Local Interpretable Model-agnostic Explanations)

- Superpixel-based explanation for individual predictions
- Shows which image regions contribute positively/negatively to each class
- Complements Grad-CAM — LIME is model-agnostic, Grad-CAM is gradient-based
- Side-by-side comparison: Grad-CAM vs LIME for same images

### 6.3 Explainability Outputs

For each test image prediction:
```
- Predicted class + confidence score
- Grad-CAM heatmap overlay
- LIME superpixel explanation
- Top-3 class probabilities bar chart
```

---

## 7. Deployment

### 7.1 TensorFlow Lite Export

- Convert best model to `.tflite` format
- Quantize (int8) for mobile/edge deployment
- Report size reduction and accuracy impact:
  ```
  Original: ~30MB → TFLite: ~8MB → Quantized: ~3MB
  ```

### 7.2 Gradio Web Demo

- Upload any chest X-ray image
- Get: predicted class, confidence %, Grad-CAM heatmap, LIME explanation
- Deployable on Hugging Face Spaces (free hosting)

### 7.3 Single-Image Inference Script

```bash
python src/predict.py --image path/to/xray.jpg --model best
```
Output: class prediction, confidence score, Grad-CAM saved to outputs/

---

## 8. Technology Stack

### Core

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Core language |
| TensorFlow | 2.15+ | Training engine (Keras integrated) |
| OpenCV | 4.9+ | Image I/O and preprocessing |
| NumPy | 1.26+ | Array operations |
| pandas | 2.1+ | Results tables and analysis |
| matplotlib | 3.8+ | Visualizations |
| seaborn | 0.13+ | Statistical plots and heatmaps |
| scikit-learn | 1.4+ | Metrics, stratified splits, utilities |

### Explainability

| Tool | Purpose |
|------|---------|
| tf-keras-vis | Grad-CAM, Grad-CAM++, ScoreCAM |
| lime | LIME image explanations |

### Deployment

| Tool | Purpose |
|------|---------|
| Gradio | Interactive web demo |
| TensorFlow Lite | Mobile/edge model export |

### Development

| Tool | Purpose |
|------|---------|
| Google Colab (free) | GPU training (T4 GPU) |
| Git + GitHub | Version control |
| Jupyter Notebook | Experimentation and visualization |

---

## 9. Compute Strategy (Free GPU)

**Primary: Google Colab Free Tier**
- GPU: NVIDIA T4 (16GB VRAM) — sufficient for all models at batch_size=32
- RAM: ~12GB — enough for this dataset
- Session limit: ~12 hours — save checkpoints frequently

**Backup: Kaggle Notebooks**
- GPU: NVIDIA P100 (16GB) — 30 hours/week
- Dataset already on Kaggle — zero download time

**Strategy to handle session limits:**
- Save model checkpoints after every epoch to Google Drive
- Resume training from checkpoint if session disconnects
- Train one model per session to avoid timeout issues

---

## 10. Folder Structure

```
pneumoscan/
│
├── data/
│   ├── raw/
│   │   └── chest_xray/
│   │       ├── train/
│   │       │   ├── NORMAL/
│   │       │   ├── BACTERIA/          # Extracted from PNEUMONIA
│   │       │   └── VIRUS/             # Extracted from PNEUMONIA
│   │       ├── val/
│   │       │   ├── NORMAL/
│   │       │   ├── BACTERIA/
│   │       │   └── VIRUS/
│   │       └── test/
│   │           ├── NORMAL/
│   │           ├── BACTERIA/
│   │           └── VIRUS/
│   └── sample/                        # 5–10 sample images for quick testing
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_baseline_custom_cnn.ipynb
│   ├── 03_resnet50_transfer.ipynb
│   ├── 04_efficientnet_transfer.ipynb
│   ├── 05_densenet121_transfer.ipynb
│   ├── 06_mobilenetv2_transfer.ipynb
│   ├── 07_ensemble_model.ipynb
│   ├── 08_evaluation_comparison.ipynb
│   └── 09_explainability_gradcam_lime.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters in one place
│   ├── data_loader.py                 # Dataset pipeline + augmentation
│   ├── preprocessing.py               # Filename-based class extraction
│   ├── models.py                      # All model builders
│   ├── train.py                       # Training loop with callbacks
│   ├── evaluate.py                    # Metrics, confusion matrix, ROC, PR
│   ├── gradcam.py                     # Grad-CAM implementation
│   ├── lime_explain.py                # LIME implementation
│   ├── ensemble.py                    # Ensemble prediction logic
│   ├── predict.py                     # Single-image inference script
│   └── utils.py                       # Plotting helpers, common utilities
│
├── models/
│   └── saved/
│       ├── custom_cnn.keras
│       ├── resnet50_finetuned.keras
│       ├── efficientnet_finetuned.keras
│       ├── densenet121_finetuned.keras
│       ├── mobilenetv2_finetuned.keras
│       └── ensemble_config.json       # Ensemble weights and model list
│
├── outputs/
│   ├── figures/
│   │   ├── eda/                       # Dataset visualizations
│   │   ├── training_curves/           # Loss + accuracy per model
│   │   ├── confusion_matrices/        # Per-model confusion matrices
│   │   ├── roc_curves/                # ROC curves (OvR)
│   │   ├── pr_curves/                 # Precision-Recall curves
│   │   ├── gradcam/                   # Grad-CAM heatmap overlays
│   │   ├── lime/                      # LIME explanations
│   │   └── comparison/                # Cross-model comparison charts
│   ├── reports/
│   │   ├── model_comparison.csv       # Final metrics table
│   │   └── classification_reports/    # Per-model sklearn reports
│   └── tflite/
│       ├── best_model.tflite
│       └── best_model_quantized.tflite
│
├── app/
│   └── demo.py                        # Gradio web demo
│
├── requirements.txt
├── .gitignore
├── PLAN.md
├── TODO.md
└── README.md
```

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Colab session timeout during training | Checkpoint to Google Drive after every epoch |
| Overfitting on small dataset | Augmentation + dropout + early stopping + class weights |
| Class imbalance skewing predictions | Weighted loss + stratified splits + per-class metrics |
| Grad-CAM not available for some layers | Use tf-keras-vis (supports all Keras models) |
| Low viral pneumonia recall | Oversample viral class or use focal loss |
| TFLite accuracy degradation | Test quantized model separately and report delta |

---

## 12. Success Criteria

The project is considered successful when:

1. **At least one model achieves ≥90% accuracy** on the 3-class test set
2. **Ensemble outperforms any single model** (even marginally)
3. **Grad-CAM heatmaps** visually align with known pneumonia patterns (lower lobes, bilateral infiltrates)
4. **LIME explanations** are consistent with Grad-CAM findings
5. **Gradio demo** is functional and deployable
6. **TFLite model** runs inference successfully with <5% accuracy drop
7. **All results are reproducible** — random seeds set, code is clean and documented
