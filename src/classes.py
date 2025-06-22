from langserve import CustomUserType
from pydantic import Field

class PDFUploadRequest(CustomUserType):
    file: str = Field(..., extra={"widget": {"type": "base64file"}})