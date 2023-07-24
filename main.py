from Functions import *
import fitz
from transformers import pipeline

my_path = "test1.pdf"
pdf = fitz.open(my_path)

# print(extract_text(pdf))
# print(extract_dict(pdf))
# print(extract_spans(pdf))
# print(score_span(pdf))
#print(merge_text(pdf))
print(extract_keywords(merge_text(pdf)))

# summary
# using pipeline API for summarization task
summarization = pipeline("summarization")
original_text = merge_text(pdf)[0:5000]
#summary_text = summarization(original_text)
#print("Summary:\n", summary_text)
