from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    """
    Input contract for a single prediction.
    Exactly 30 numerical features are required.
    """

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 numerical model features"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, value):

        for feature in value:

            if not isinstance(feature, (int, float)):
                raise ValueError(
                    "All features must be numerical"
                )

        return value


class BatchPredictionInput(BaseModel):
    """
    Input contract for batch predictions.
    """

    records: list[PredictionInput] = Field(
        ...,
        min_length=1
    )