from Functions import *

my_path = "test2.pdf"
pdf = fitz.open(my_path)

#print(Extract_text(pdf))
#print(Extract_dict(pdf))
#print(Extract_spans(pdf))
#print(Score_span(pdf))
print(Merge_text(pdf))