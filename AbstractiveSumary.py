from transformers import pipeline

with open("text2.txt", "r") as file:
    text = file.read()

summarization = pipeline("summarization")
original_text = text[0:5000]
summary_text = summarization(original_text)
print("Summary:\n", summary_text)
