from pydantic import BaseModel


class InputRequest(BaseModel):
    question: str

class OutputResponse(BaseModel):
    response: str