# Eye Disease Classification

This project contains:

- A training notebook: `fa2-dl (2).ipynb`
- A saved TensorFlow model for inference: `final_eye_model.keras`
- A Flask web app for local prediction

The model classifies eye images into:

- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

## Project Structure

- `fa2-dl (2).ipynb`: training, evaluation, and visualization workflow
- `app.py`: Flask backend for loading model and serving predictions
- `templates/index.html`: frontend UI
- `static/styles.css`: styling
- `static/app.js`: upload and prediction requests
- `uploads/`: temporary uploaded images
- `requirements.txt`: Python dependencies

## Run the Web App (Windows)

1. Open a terminal in the project folder.
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

5. Open:

```text
http://127.0.0.1:5000
```

## Inference Pipeline

- Input image is resized to `300 x 300`
- Preprocessing uses `tf.keras.applications.efficientnet.preprocess_input`
- Model returns class probabilities (softmax)
- UI displays top prediction and confidence

## Notebook Summary

The notebook performs:

- Data loading from a class-folder dataset
- Train/validation split
- Transfer learning with EfficientNetB3
- Training with callbacks (early stopping, LR scheduling, checkpointing)
- Evaluation using confusion matrix and classification report
- Model export (`.keras` and `.h5`)

## Notes

- If dataset paths are Kaggle-specific, update them when running outside Kaggle.
- Ensure `final_eye_model.keras` is present in the project root before running the app.
