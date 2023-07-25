from transformers import pipeline
from Functions import *

text = open_file("CleanText")

summarization = pipeline("summarization")
original_text = text[0:5000]
save_file("AbstractiveSummary", str(summarization(original_text)))
