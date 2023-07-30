import openai
from Functions import *

openai.api_key = "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"

messages = []

text = open_file("Output/Tfidf")

messages.append({"role": "user", "content": "summarize this in 500 words : " + text})

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

chat_response = completion.choices[0].message.content
print(f"ChatGPT: {chat_response}")
# messages.append({"role": "assistant", "content": chat_response})
