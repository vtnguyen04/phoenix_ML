from pydantic import BaseModel


class LoadModelCommand(BaseModel):
    """
    Command to trigger loading of a model into inference engine.
    """
    model_id: str
    model_version: str
    device: str = "cpu"
