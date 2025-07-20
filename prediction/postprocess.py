# Created by Carlos Alvarez on 20-07-2025
import numpy as np

def postprocess_predictions(predictions:np.ndarray) -> np.ndarray:
    """Postprocess the model predictions."""

    # Round the predictions to the nearest integer
    predictions = predictions.round().astype(int)
    # Ensure predictions are non-negative
    predictions = predictions.clip(min=0)

    return predictions

def minutes_to_hhmm(x, pos):
    """Convert minutes to HH:MM format for plotting."""
    hours = int(x // 60)
    minutes = int(x % 60)
    return f'{hours:02}:{minutes:02}'