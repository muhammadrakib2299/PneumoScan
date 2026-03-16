# PneumoScan — Task Checklist

## Phase 0: Project Setup
- [ ] Initialize Git repository
- [ ] Create folder structure (data/, notebooks/, src/, models/, outputs/, app/)
- [ ] Create requirements.txt with all dependencies
- [ ] Create .gitignore (data/raw/, models/saved/, outputs/, __pycache__, .ipynb_checkpoints)
- [ ] Set up Google Colab notebook linked to Google Drive
- [ ] Download Kaggle Chest X-Ray dataset
- [ ] Write `src/config.py` — centralized hyperparameters and paths

## Phase 1: Data Preparation (Week 1)
- [ ] **Notebook 01**: EDA and Preprocessing
  - [ ] Write `src/preprocessing.py` — extract 3 classes from filenames (Normal/Bacteria/Virus)
  - [ ] Reorganize dataset into 3-class folder structure
  - [ ] Merge tiny val set (24 images) into training set
  - [ ] Count images per class per split — document imbalance
  - [ ] Visualize class distribution (bar chart)
  - [ ] Display sample images from each class (grid: 3×4)
  - [ ] Analyze image size distribution (histogram)
  - [ ] Compute pixel intensity distributions per class
  - [ ] Calculate class weights for weighted loss
  - [ ] Document findings in notebook markdown cells
- [ ] Write `src/data_loader.py`
  - [ ] Build tf.data pipeline with prefetch and caching
  - [ ] Implement augmentation layer (flip, rotate, zoom, brightness)
  - [ ] Add stratified train/val split function (for K-Fold)
  - [ ] Add class weight computation function
  - [ ] Test pipeline outputs shape and dtype
- [ ] Write `src/utils.py`
  - [ ] Plot training history function (loss + accuracy curves)
  - [ ] Plot confusion matrix function
  - [ ] Plot ROC curve function (One-vs-Rest)
  - [ ] Plot Precision-Recall curve function
  - [ ] Save figure helper (auto-creates directory)

## Phase 2: Baseline Model (Week 1)
- [ ] Write `src/models.py` — model builder functions
  - [ ] `build_custom_cnn()` — 4-block Conv2D baseline
  - [ ] `build_resnet50()` — pretrained + custom head
  - [ ] `build_efficientnet_b0()` — pretrained + custom head
  - [ ] `build_densenet121()` — pretrained + custom head
  - [ ] `build_mobilenetv2()` — pretrained + custom head
- [ ] Write `src/train.py`
  - [ ] Training function with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  - [ ] Two-phase training (frozen → fine-tune)
  - [ ] Checkpoint saving to Google Drive
  - [ ] Training history logging to CSV
- [ ] **Notebook 02**: Custom CNN Baseline
  - [ ] Build and compile custom CNN
  - [ ] Train with class weights
  - [ ] Plot training curves (loss + accuracy)
  - [ ] Evaluate on test set
  - [ ] Generate confusion matrix
  - [ ] Document baseline metrics

## Phase 3: Transfer Learning Models (Week 2)
- [ ] **Notebook 03**: ResNet-50
  - [ ] Phase 1: Feature extraction (frozen base, 10 epochs)
  - [ ] Phase 2: Fine-tuning (unfreeze top 30%, 20 epochs, LR=1e-5)
  - [ ] Plot training curves for both phases
  - [ ] Evaluate on test set
  - [ ] Save model to models/saved/
- [ ] **Notebook 04**: EfficientNet-B0
  - [ ] Phase 1: Feature extraction
  - [ ] Phase 2: Fine-tuning
  - [ ] Plot training curves
  - [ ] Evaluate on test set
  - [ ] Save model
- [ ] **Notebook 05**: DenseNet-121
  - [ ] Phase 1: Feature extraction
  - [ ] Phase 2: Fine-tuning
  - [ ] Plot training curves
  - [ ] Evaluate on test set
  - [ ] Save model
- [ ] **Notebook 06**: MobileNetV2
  - [ ] Phase 1: Feature extraction
  - [ ] Phase 2: Fine-tuning
  - [ ] Plot training curves
  - [ ] Evaluate on test set
  - [ ] Save model

## Phase 4: Ensemble Model (Week 2)
- [ ] Write `src/ensemble.py`
  - [ ] Load top-3 models by validation accuracy
  - [ ] Implement soft voting (average probabilities)
  - [ ] Implement weighted voting (weight by AUC)
  - [ ] Compare both approaches
  - [ ] Save ensemble config (model list + weights) to JSON
- [ ] **Notebook 07**: Ensemble Model
  - [ ] Load all 5 trained models
  - [ ] Rank by validation performance
  - [ ] Build soft voting ensemble
  - [ ] Build weighted voting ensemble
  - [ ] Evaluate both ensembles on test set
  - [ ] Compare ensemble vs individual models

## Phase 5: Evaluation & Comparison (Week 3)
- [ ] Write `src/evaluate.py`
  - [ ] Per-model evaluation pipeline
  - [ ] Multi-class confusion matrix
  - [ ] Classification report (precision, recall, F1 per class)
  - [ ] ROC curves (One-vs-Rest) with AUC scores
  - [ ] Precision-Recall curves with AP scores
  - [ ] Multi-threshold analysis (precision/recall/F1 at thresholds 0.3–0.9)
  - [ ] Cohen's Kappa score
  - [ ] Inference time measurement (ms per image)
- [ ] **Notebook 08**: Full Evaluation & Comparison
  - [ ] Load all models (5 individual + ensemble)
  - [ ] Run evaluation pipeline on each
  - [ ] Generate comparison table (Accuracy, F1-Macro, AUC-ROC, Params, Inference time)
  - [ ] Plot grouped bar chart comparing all models
  - [ ] Plot ROC curves for all models on same figure
  - [ ] Plot PR curves for all models on same figure
  - [ ] Multi-threshold analysis for best model
  - [ ] Save comparison CSV to outputs/reports/
  - [ ] Save all figures to outputs/figures/comparison/

## Phase 6: Explainability (Week 3)
- [ ] Write `src/gradcam.py`
  - [ ] Grad-CAM heatmap generation for any model + image
  - [ ] Overlay heatmap on original image
  - [ ] Multi-class Grad-CAM (heatmap per class)
  - [ ] Grid visualization: [Original | Normal-CAM | Bacteria-CAM | Virus-CAM]
- [ ] Write `src/lime_explain.py`
  - [ ] LIME image explainer setup
  - [ ] Generate superpixel explanations
  - [ ] Positive/negative contribution visualization
  - [ ] Side-by-side: Grad-CAM vs LIME for same image
- [ ] **Notebook 09**: Explainability (Grad-CAM + LIME)
  - [ ] Select 5 representative test images per class (15 total)
  - [ ] Generate Grad-CAM for all 5 models on selected images
  - [ ] Compare: Do different models attend to different regions?
  - [ ] Generate LIME explanations for best model
  - [ ] Side-by-side Grad-CAM vs LIME comparison
  - [ ] Cross-model attention comparison grid
  - [ ] Save all heatmaps to outputs/figures/gradcam/
  - [ ] Save all LIME outputs to outputs/figures/lime/
  - [ ] Interpret findings in notebook markdown

## Phase 7: Deployment (Week 3)
- [ ] **TFLite Export**
  - [ ] Convert best model to .tflite (float32)
  - [ ] Convert best model to quantized .tflite (int8)
  - [ ] Measure file size reduction
  - [ ] Test TFLite inference accuracy on test set
  - [ ] Report accuracy delta (full model vs TFLite vs quantized)
  - [ ] Save to outputs/tflite/
- [ ] Write `src/predict.py` — single-image inference
  - [ ] Load model from saved path
  - [ ] Preprocess input image
  - [ ] Run prediction → class + confidence
  - [ ] Generate Grad-CAM for the prediction
  - [ ] Print results + save heatmap
  - [ ] CLI: `python src/predict.py --image path/to/xray.jpg --model best`
- [ ] Write `app/demo.py` — Gradio web demo
  - [ ] Image upload interface
  - [ ] Model selection dropdown (5 models + ensemble)
  - [ ] Output: predicted class + confidence bar chart
  - [ ] Output: Grad-CAM heatmap overlay
  - [ ] Output: LIME explanation
  - [ ] Test locally
  - [ ] Add Hugging Face Spaces deployment instructions

## Phase 8: Documentation & Polish (Week 3)
- [ ] Finalize README.md with results, figures, and badges
- [ ] Add sample output images to README
- [ ] Verify all notebooks run end-to-end without errors
- [ ] Clean up code — remove dead code, unused imports
- [ ] Add docstrings to all src/ functions
- [ ] Verify .gitignore covers data/raw/, models/saved/, outputs/
- [ ] Final git commit with clean history
- [ ] Add requirements.txt with pinned versions
- [ ] Add sample images to data/sample/ for quick demos

## Stretch Goals (If Time Permits)
- [ ] Grad-CAM++ (improved Grad-CAM) comparison
- [ ] Cross-validation results (5-Fold) for all models
- [ ] Attention rollout visualization for EfficientNet
- [ ] Model card (standardized ML model documentation)
- [ ] Deploy Gradio demo to Hugging Face Spaces
- [ ] Record a 2-minute demo video
- [ ] Write a Medium/blog post about the project
