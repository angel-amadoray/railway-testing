from pydantic import BaseModel

class CompositeResponse(BaseModel):
    url: str
    cached: bool
    parameters: dict