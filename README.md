# Food-101 Image Classifier (InceptionV3)

A beginner-friendly deep-learning project that classifies food images into the 101 Food-101 categories using transfer learning with TensorFlow/Keras and InceptionV3.

## What it does
1. Downloads the Food-101 dataset through `tensorflow_datasets` (first run requires internet).
2. Resizes and preprocesses images for InceptionV3.
3. Trains a classification head on top of a pretrained InceptionV3 model.
4. Saves the trained model and class names.
5. Evaluates the model and lets you predict the class of a local image.

> Training all 101 classes can take significant time and GPU memory. Start with a small run using `--epochs 1 --max-train 2000 --max-test 500` to verify the pipeline, then increase the limits. Pretrained ImageNet weights may need to download on first run.

## Requirements
- Python 3.10 or 3.11 recommended
- pip
- For practical training, an NVIDIA GPU is helpful but not mandatory. CPU training is much slower.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train a quick smoke test
```powershell
python train.py --epochs 1 --batch-size 16 --max-train 2000 --max-test 500
```

## Train with more data
```powershell
python train.py --epochs 5 --batch-size 32
```

The first run downloads Food-101 through TensorFlow Datasets. Full training can take hours depending on hardware. To improve accuracy, train the classification head first, then optionally fine-tune some top InceptionV3 layers with a low learning rate.

## Predict an image
After training:
```powershell
python predict.py --image path\to\food.jpg
```

## Output files
- `artifacts/food101_inceptionv3.keras` — trained model
- `artifacts/class_names.json` — class label order
- `artifacts/evaluation.json` — test evaluation metrics

## Project workflow
Food-101 images -> TensorFlow Datasets -> resize/preprocess -> InceptionV3 pretrained feature extractor -> 101-class softmax layer -> training/evaluation -> saved model -> prediction.

## Important notes
- The Food-101 dataset has 101 categories and is a large dataset. The quick test limits the number of examples to verify that code runs; it is not a final model-quality evaluation.
- Dataset labels come from TensorFlow Datasets, so class order is saved alongside the model to ensure predictions map to the right category.
- This is a portfolio starter. Report actual accuracy only after you run evaluation on your trained model; do not invent a score.
