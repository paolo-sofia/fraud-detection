import logging
import os

import fastapi
import numpy as np
import pandas as pd
import uvicorn
from sklearnex import patch_sklearn

from src.fraud_detection.inference.loaders import load_columns, load_model
from src.fraud_detection.preprocessing.inference import prepare_data_for_inference

app = fastapi.FastAPI(title="fraud-detection model", description="Api that performs fraud detection", version="1.0.0")
patch_sklearn()
model = load_model()
columns = load_columns()


@app.get("/health")
async def predict() -> dict[str, str]:
    return {"message": "Healthy"}


@app.post("/predict")
async def predict(data: dict[str, str | int | bool | float]) -> dict[str, str | dict[str, int | float]]:
    try:
        data: pd.DataFrame = prepare_data_for_inference(data, columns).to_pandas()

        prediction_probability: np.ndarray = model.predict_proba(data)[0]

        prediction: bool = bool(prediction_probability[1] > float(os.getenv("THRESHOLD", "0.5")))
        probability: float = float(prediction_probability[1]) if prediction else float(prediction_probability[0])

        return {"message": "Prediction successfully", "data": {"class": prediction, "probability": probability}}

    except Exception as e:
        logging.error("Error inside the predict function")
        logging.error(e)
        return {"message": "Error when performing prediction", "error": e}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
