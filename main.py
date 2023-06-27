from Functions import *
from transformers import pipeline

my_path = "test1.pdf"
pdf = fitz.open(my_path)

#print(Extract_text(pdf))
#print(Extract_dict(pdf))
#print(Extract_spans(pdf))
#print(Score_span(pdf))
print(Merge_text(pdf))

#summary
# using pipeline API for summarization task
summarization = pipeline("summarization")
original_text = Merge_text(pdf)[0:1000]
summary_text = summarization(original_text)
print("Summary:\n", summary_text)