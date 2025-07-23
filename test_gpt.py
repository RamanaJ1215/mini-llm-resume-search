import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What are my strengths as a .NET developer?"
    answer = ask_gpt(question)
    print("GPT says:\n", answer)
