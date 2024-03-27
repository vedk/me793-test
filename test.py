import os

import google.generativeai as genai

from PIL import Image
from dotenv import load_dotenv
from make_data import prompt_gemini

load_dotenv()

img = Image.open("Part5.jpg")

# create a file constants.py and put your Google API key there
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")
response = model.generate_content(
    [
        "Does the following material have any defect?",
        img,
        "Describe the defect in 50 words.",
        "Are you sure that the defect is not on the right side?",
    ],
    stream=True,
)

print(response.prompt_feedback)
for chunk in response:
    print(chunk.text)

ans = prompt_gemini(img, model)
print(f"ans = {ans}")
