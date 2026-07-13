from pydantic import BaseModel
from typing import Union, List, Dict


class RewriterSchema(BaseModel):
    revised_prompt: Union[str, List[Dict[str, str]]]
