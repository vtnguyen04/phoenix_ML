from pydantic import BaseModel


class PredictCommand(BaseModel):
    """
    Command object for prediction requests.
    """
    model_id: str
    model_version: str | None = None
    features: list[float] | None = None
    entity_id: str | None = None
