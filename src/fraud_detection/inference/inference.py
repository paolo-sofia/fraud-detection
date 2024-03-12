import fastapi
import uvicorn

app = fastapi.FastAPI(title="fraud-detection model", description="Api that performs fraud detection", version="1.0.0")


@app.get("/health")
async def predict() -> dict[str, str]:
    return {"message": "Healthy"}


@app.post("/predict")
async def predict(array: list) -> dict[str, str]:
    return {"message": "Healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
