from pydantic import BaseModel


class ReviewRequest(BaseModel):
    place_type: str
    text: str