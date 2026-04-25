# Eye Disease Classification (TensorFlow)

This notebook trains and evaluates an image classifier for eye disease categories using transfer learning with EfficientNetB3.

## Web App (UI + Backend)

This project now includes a complete local web application where you can:

- Upload an eye image from your computer
- Run prediction using your saved model (`final_eye_model.keras`)
- View predicted class, confidence, and probabilities for all classes

### Project Files for App

- `app.py` -> Flask backend API + model inference
- `templates/index.html` -> Modern responsive frontend UI
- `static/styles.css` -> UI design and animations
- `static/app.js` -> Frontend upload and prediction logic
- `requirements.txt` -> Python dependencies

### Run Locally (Windows)

1. Open terminal in this folder.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the app:

```powershell
python app.py
```

5. Open in browser:

```text
http://127.0.0.1:5000
```

### How Prediction Works

- Uploaded image is resized to `300 x 300`
- Image is preprocessed with `tf.keras.applications.efficientnet.preprocess_input`
- Model outputs softmax scores for:
  - Cataract
  - Diabetic Retinopathy
  - Glaucoma
  - Normal
- UI shows top class and confidence percentage

## Overview

- Loads images from a Kaggle dataset directory and splits into train/validation.
- Builds a transfer-learning model (EfficientNetB3 backbone + custom classifier head).
- Trains with data augmentation, early stopping, and learning-rate scheduling.
- Visualizes dataset balance, sample images, and training metrics.
- Evaluates with confusion matrix and classification report.
- Saves the trained model and runs a single-image prediction demo.

## Dataset

The notebook expects a folder-structured dataset (one folder per class) located at:

```
/kaggle/input/datasets/gunavenkatdoddi/eye-diseases-classification/dataset
```

Class names are inferred from the directory names. The demo prediction uses:

```
/kaggle/input/datasets/abhaypatil23/test-demo/_9_9966988.jpg
```

## Model

- Backbone: EfficientNetB3 (ImageNet weights, top removed)
- Input size: 300x300
- Head: GlobalAveragePooling2D -> BatchNorm -> Dense(256, relu) -> Dropout(0.6) -> Dense(NUM_CLASSES, softmax)

## Training

Key settings:

- Batch size: 32
- Epochs: 25
- Optimizer: Adam (1e-4)
- Loss: sparse categorical crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Outputs

The notebook:

- Plots class distribution, sample images, and learning curves
- Shows a confusion matrix and classification report
- Saves models as:
  - `best_model.keras` (checkpoint)
  - `final_eye_model.keras`
  - `final_eye_model.h5`

## How to Run (Kaggle)

1. Upload/open the notebook on Kaggle.
2. Ensure the dataset path matches the expected directory structure.
3. Run all cells to train and evaluate the model.
4. Update `IMAGE_PATH` to test a new image and re-run the prediction cell.

## Notes

- GPU is detected via `tf.config.list_physical_devices('GPU')`.
- Paths are Kaggle-specific; change them if running locally.
