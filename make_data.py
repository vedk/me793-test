import base64
import sqlite3
import time

import anthropic
import google.generativeai as genai

from os import environ, listdir
from os.path import isfile, join

from dotenv import load_dotenv
from PIL import Image


def prompt_gemini(img: Image, model: genai.GenerativeModel) -> str:
    response = model.generate_content(
        [
            "Does the following material have any defect?",
            img,
            "Describe the defect in 50 words.",
            # "Are you sure that the defect is not on the right side?",
        ],
        stream=True,
    )

    # print(response.prompt_feedback)

    complete_respose = str()
    for chunk in response:
        complete_respose += chunk.text

    return complete_respose


def prompt_haiku(b64img: str, format: str, client: anthropic.Anthropic) -> str:
    MODEL = "claude-3-haiku-20240307"
    MAX_TOKENS = 1024
    MEDIA_TYPE = f"image/{format}"

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
                            "data": b64img,
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

    return message.content[0].text


def main():
    load_dotenv()

    print("searching for dataset")
    DATASET_PATH = environ.get("DATASET_PATH")
    files = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]

    conn = sqlite3.connect("test.db")
    cur = conn.cursor()

    print("creating database")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS haiku(file_name STR PRIMARY KEY, output STR NOT NULL)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS gemini(file_name STR PRIMARY KEY, output STR NOT NULL)"
    )
    conn.commit()

    # configure Google Gemini
    genai.configure(api_key=environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro-vision")

    # configure Anthropic Claude Haiku
    client = anthropic.Anthropic(api_key=environ.get("HAIKU_API_KEY"))

    # fill the database
    for file in files:
        img = Image.open(join(DATASET_PATH, file))
        with open(join(DATASET_PATH, file), "rb") as fd:
            imgdata = base64.b64encode(fd.read()).decode()

        
        print(f"processing {file} on haiku")
        haiku_ans = prompt_haiku(imgdata, img.format.lower(), client)
        cur.execute("INSERT INTO haiku VALUES(?, ?)", (file, haiku_ans))

        print(f"processing {file} on gemini")
        gemini_ans = prompt_gemini(img, model)
        cur.execute("INSERT INTO gemini VALUES(?, ?)", (file, gemini_ans))

        conn.commit()

        print("sleeping for 12 s")
        time.sleep(12)

    conn.close()


if __name__ == "__main__":
    main()
