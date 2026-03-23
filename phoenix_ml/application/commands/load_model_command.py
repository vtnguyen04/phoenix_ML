from pydantic import BaseModel


class LoadModelCommand(BaseModel):
    """Input DTO for loading a model into the inference engine."""

    model_id: str
    model_version: str
    device: str = "cpu"
