import json
import logging
import os

import numpy as np
import pandas as pd
from robyn import Request, Robyn
from sklearnex import patch_sklearn

from src.fraud_detection.inference.loaders import load_columns, load_model
from src.fraud_detection.preprocessing.inference import prepare_data_for_inference

app = Robyn(__file__)

patch_sklearn()
model = load_model()
columns = load_columns()


@app.get("/health")
async def predict() -> dict[str, str]:
    return {"message": "Healthy"}


@app.post("/predict")
def predict(request: Request) -> dict[str, str | dict[str, int | float]]:
    try:
        data: dict = json.loads(request.body)

        logging.error(f"data: {data}")

        data: pd.DataFrame = prepare_data_for_inference(data, columns).to_pandas()
        prediction_probability: np.ndarray = model.predict_proba(data)[0]

        logging.error(f"prediction_probability: {prediction_probability}")

        prediction: bool = bool(prediction_probability[1] > float(os.getenv("THRESHOLD", "0.5")))

        logging.error(f"prediction: {prediction}")
        probability: float = float(prediction_probability[1]) if prediction else float(prediction_probability[0])
        logging.error(f"results: {prediction}, {probability}")
        return {"message": "Prediction successfully", "data": {"class": prediction, "probability": probability}}

    except Exception as e:
        logging.error("Error inside the predict function")
        logging.error(e)
        return {"message": "Error when performing prediction", "error": e}


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8000)
