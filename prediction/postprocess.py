# Created by Carlos Alvarez on 20-07-2025
import numpy as np

def postprocess_predictions(predictions:np.ndarray) -> np.ndarray:
    """Postprocess the model predictions."""

    # Round the predictions to the nearest integer
    predictions = predictions.round().astype(int)
    # Ensure predictions are non-negative
    predictions = predictions.clip(min=0)

    return predictions