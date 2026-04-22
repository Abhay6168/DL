# Eye Disease Classification (TensorFlow)

This notebook trains and evaluates an image classifier for eye disease categories using transfer learning with EfficientNetB3.

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
