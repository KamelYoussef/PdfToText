from Functions import *
import fitz

my_path = "test2.pdf"
pdf = fitz.open(my_path)

# print(extract_text(pdf))
# print(extract_dict(pdf))
# print(extract_spans(pdf))
# print(score_span(pdf))
# print(merge_text(pdf))
# print(extract_keywords(merge_text(pdf)))

save_file("CleanText", merge_text(pdf))
