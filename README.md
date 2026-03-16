# PneumoScan

**Transfer Learning for Pneumonia Detection: From Agricultural Vision to Medical Imaging**

A deep learning pipeline that classifies chest X-rays into **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia** using transfer learning — with explainable AI (Grad-CAM + LIME) and a deployable web demo.

> This project proves that computer vision pipelines developed for agricultural tasks can transfer effectively to life-saving medical imaging applications.

---

## Highlights

- **5 models compared**: Custom CNN, ResNet-50, EfficientNet-B0, DenseNet-121, MobileNetV2
- **Ensemble model**: Soft/weighted voting of top-3 models
- **3-class classification**: Normal vs Bacterial Pneumonia vs Viral Pneumonia
- **Explainable AI**: Grad-CAM heatmaps + LIME superpixel explanations
- **Deployable**: Gradio web demo + TFLite mobile export
- **Free GPU training**: Runs entirely on Google Colab / Kaggle free tier

---

## Results

### Model Comparison

| Model | Accuracy | F1 (Macro) | AUC-ROC | Parameters | Inference |
|-------|----------|-----------|---------|------------|-----------|
| Custom CNN | — | — | — | ~500K | — |
| ResNet-50 | — | — | — | 25.6M | — |
| EfficientNet-B0 | — | — | — | 5.3M | — |
| DenseNet-121 | — | — | — | 8.0M | — |
| MobileNetV2 | — | — | — | 3.4M | — |
| **Ensemble** | — | — | — | — | — |

> Results will be filled after training is complete.

### Grad-CAM Visualization

```
[Original X-Ray] → [Normal Heatmap] → [Bacterial Heatmap] → [Viral Heatmap]
```

> Sample Grad-CAM outputs will be added after the explainability phase.

---

## Dataset

**Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Normal | Bacterial | Viral | Total |
|-------|--------|-----------|-------|-------|
| Train | ~1,341 | ~2,530 | ~1,345 | ~5,216 |
| Test | 234 | 242 | 148 | 624 |

3 classes extracted from filename prefixes (`bacteria_` / `virus_` / NORMAL folder).

---

## Project Structure

```
pneumoscan/
├── data/
│   ├── raw/chest_xray/          # Kaggle dataset (3-class)
│   └── sample/                  # Quick-test images
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
├── src/
│   ├── config.py                # Centralized hyperparameters
│   ├── data_loader.py           # tf.data pipeline + augmentation
│   ├── preprocessing.py         # 3-class extraction from filenames
│   ├── models.py                # All 5 model builders
│   ├── train.py                 # Training loop with callbacks
│   ├── evaluate.py              # Metrics + comparison
│   ├── gradcam.py               # Grad-CAM implementation
│   ├── lime_explain.py          # LIME implementation
│   ├── ensemble.py              # Ensemble logic
│   ├── predict.py               # Single-image inference
│   └── utils.py                 # Plotting helpers
├── models/saved/                # Trained .keras files
├── outputs/
│   ├── figures/                 # All visualizations
│   ├── reports/                 # CSV metrics and comparisons
│   └── tflite/                  # Exported lightweight models
├── app/
│   └── demo.py                  # Gradio web demo
├── requirements.txt
├── PLAN.md
├── TODO.md
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PneumoScan.git
cd PneumoScan
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place in `data/raw/chest_xray/`.

### 4. Run Preprocessing

```bash
python src/preprocessing.py
```

This reorganizes the dataset from 2-class (Normal/Pneumonia) to 3-class (Normal/Bacteria/Virus) based on filename prefixes.

### 5. Train Models

Run notebooks 02–07 sequentially in Google Colab, or:

```bash
python src/train.py --model all
```

### 6. Evaluate

```bash
python src/evaluate.py --model all
```

### 7. Run Web Demo

```bash
python app/demo.py
```

Opens a Gradio interface at `http://localhost:7860` — upload any chest X-ray to get a prediction with Grad-CAM heatmap.

### 8. Single Image Inference

```bash
python src/predict.py --image path/to/xray.jpg --model best
```

---

## Technology Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow 2.15+, Keras |
| **Image Processing** | OpenCV 4.9+ |
| **ML Utilities** | scikit-learn 1.4+, NumPy, pandas |
| **Visualization** | matplotlib 3.8+, seaborn 0.13+ |
| **Explainability** | tf-keras-vis (Grad-CAM), LIME |
| **Deployment** | Gradio, TensorFlow Lite |
| **Training** | Google Colab (free T4 GPU) |

---

## Training Approach

### Two-Phase Transfer Learning

```
Phase 1: Feature Extraction
  └─ Freeze pretrained base → Train classification head → 10 epochs, LR=1e-3

Phase 2: Fine-Tuning
  └─ Unfreeze top 30% of base → Train end-to-end → 20 epochs, LR=1e-5
```

### Handling Class Imbalance

- Weighted categorical crossentropy loss
- Targeted data augmentation (rotation, flip, zoom, brightness)
- Stratified train/validation splits

---

## Explainability

### Grad-CAM

Gradient-weighted Class Activation Mapping highlights which regions of the X-ray the model focuses on when making a prediction. This helps clinicians understand and trust the AI's decision.

### LIME

Local Interpretable Model-agnostic Explanations identify which superpixel regions contribute positively or negatively to each class prediction — providing a complementary view to Grad-CAM.

---

## Deployment Options

| Method | Use Case | Size |
|--------|----------|------|
| **Gradio Web App** | Browser-based demo, Hugging Face Spaces | Full model |
| **TFLite (float32)** | Mobile/edge deployment | ~8MB |
| **TFLite (int8 quantized)** | Ultra-lightweight deployment | ~3MB |

---

## Thesis Connection

This project is a direct extension of prior research on dragon fruit disease classification using CNNs. The same transfer learning methodology is applied here to medical imaging, demonstrating that:

1. **The pipeline is domain-agnostic** — works for agriculture and healthcare
2. **Transfer learning from ImageNet generalizes** across visual domains
3. **Explainability tools** (Grad-CAM) provide meaningful outputs for both domains
4. **The approach scales** from 4-class plant disease to 3-class medical diagnosis

---

## License

This project is for educational and research purposes. The chest X-ray dataset is publicly available on Kaggle under its original license.

---

## Acknowledgments

- **Dataset**: [Kermany et al., 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) — Chest X-Ray Images (Pneumonia)
- **CheXNet Reference**: [Rajpurkar et al., 2017](https://arxiv.org/abs/1711.05225) — DenseNet-121 for chest X-ray diagnosis
- **Grad-CAM**: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- **LIME**: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
