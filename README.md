# CVAE Plant Stress Detection

**Conditional Variational Autoencoder for Plant Anomaly Detection**

Uses reconstruction error as an anomaly score to classify plant images as Healthy or Stressed.

---

## Project Structure

```
cvae_plant_stress/
├── notebooks/
│   └── main.ipynb          ← Full training + evaluation notebook
├── webapp/
│   ├── app.py              ← Flask web demo
│   └── templates/
│       └── index.html      ← Demo UI
├── utils/
│   └── model.py            ← CVAE class + inference helpers (shared)
├── models/
│   └── cvae_weights.pth    ← Saved after training (auto-generated)
├── results/                ← All plots saved here (auto-generated)
│   ├── training_loss.png
│   ├── error_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── tsne_latent.png
│   ├── reconstruction_comparison.png
│   ├── synthetic_images.png
│   ├── metrics.json
│   └── classification_report.txt
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```
jupyter notebook notebooks/main.ipynb
```
Run all cells top to bottom. This will:
- Download and convert the AGMHS dataset
- Train the CVAE for 40 epochs
- Evaluate and save all result plots to `results/`
- Save model weights to `models/cvae_weights.pth`

### 3. Launch the web demo
```bash
cd webapp
python app.py
```
Open http://localhost:5000

---

## What the Notebook Covers

| Section | Content |
|---|---|
| Data loading | HuggingFace AGMHS dataset → PNG images |
| Train/test split | 80/20 with fixed seed |
| CVAE architecture | FC encoder + decoder, latent dim=50 |
| Training | 40 epochs, Adam, BCE+KLD loss |
| Loss curve | Training convergence plot |
| Anomaly scoring | MSE reconstruction error |
| Metrics | Accuracy, Precision, Recall, F1 |
| Confusion matrix | Seaborn heatmap |
| ROC + AUC | Full curve with AUC score |
| t-SNE | Latent space 2D visualization |
| Reconstruction | 5 sample original vs reconstructed |
| Generation | Synthetic plant image sampling |

---

## Key Results (expected after training)

- Healthy plants produce **low** reconstruction error
- Stressed plants produce **high** reconstruction error
- Threshold = midpoint of class means
- AUC typically > 0.80 after 40 epochs

---

## Web App Features

- Upload any plant image → get Healthy/Stressed prediction
- See original vs reconstructed side-by-side
- Visual error bar relative to threshold
- Generate synthetic plant images per class
- View all training result plots
- Live metrics dashboard (accuracy, F1, AUC)
