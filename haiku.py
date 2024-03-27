import os
import base64

import anthropic

from dotenv import load_dotenv
from make_data import prompt_haiku


load_dotenv()

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 1024
MEDIA_TYPE = "image/jpeg"

client = anthropic.Anthropic(api_key=os.environ.get("HAIKU_API_KEY"))

with open("Part5.jpg", "rb") as fd:
    imgdata = base64.b64encode(fd.read()).decode()

message = client.messages.create(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    system="You are an AI materials expert who can understand cracks and surface defects in an object very well and describe the same",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": MEDIA_TYPE,
                        "data": imgdata,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe the surface defect in this image. You can mention the technical details, dimensions, causes and effects of this defect.",
                },
            ],
        }
    ],
)

print(message.content[0].text)

ans = prompt_haiku(imgdata, client)
print(f"ans = {ans}")
