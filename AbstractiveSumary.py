from transformers import pipeline
from Functions import *

text = open_file("Output/Tfidf")

summarization = pipeline("summarization")
original_text = text
summary = summarization(original_text)[0]["summary_text"]
save_file("Output/AbstractiveSummary", summary)
