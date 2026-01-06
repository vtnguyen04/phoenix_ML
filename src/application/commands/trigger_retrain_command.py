from pydantic import BaseModel


class TriggerRetrainCommand(BaseModel):
    """
    Command to trigger a model retraining pipeline.
    """
    model_id: str
    reason: str
    dataset_date_range_start: str | None = None
    dataset_date_range_end: str | None = None
