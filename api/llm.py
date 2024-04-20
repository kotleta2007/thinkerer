from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))


def complete_json(text, temperature=0.0,max_tokens=1024,json=False):
    res = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model="llama3-70b-8192", #llama3-8b-8192
        temperature=temperature,
        max_tokens=max_tokens,
        response_format= {"type": "json_object"}
    )
    return res.choices[0].message.content
