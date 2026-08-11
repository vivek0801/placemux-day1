from fastapi import FastAPI, HTTPException

from src.schemas import (
    PredictionInput,
    BatchPredictionInput
)

from src.predict import (
    predict_single,
    predict_batch
)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="PlaceMux Model Scoring API",
    version="1.0.0",
    description="Validated scoring interface for the PlaceMux classifier."
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_version": "day12-v1"
    }


# --------------------------------------------------
# Single Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(input_data: PredictionInput):

    try:

        result = predict_single(
            input_data.features
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

@app.post("/predict/batch")
def predict_batch_endpoint(
    input_data: BatchPredictionInput
):

    try:

        results = predict_batch(
            [
                record.features
                for record in input_data.records
            ]
        )

        return {
            "model_version": "day12-v1",
            "results": results
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )