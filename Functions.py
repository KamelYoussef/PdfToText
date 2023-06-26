import fitz
from unidecode import unidecode
import re
import pandas as pd
import numpy as np

def Extract_text(doc):
    output = []
    raw = ""
    for page in doc:
        output += page.get_text("blocks")
    for block in output:
        if block[6] == 0: # We only take the text
               plain_text = unidecode(block[4]) # Encode in ASCII
               raw += plain_text
    return raw

def Extract_dict(doc):
    block_dict = {}
    page_num = 1
    for page in doc: # Iterate all pages in the document
        file_dict = page.get_text('dict') # Get the page dictionary
        block = file_dict['blocks'] # Get the block information
        block_dict[page_num] = block # Store in block dictionary
        page_num += 1 # Increase the page value by 1
    return block_dict

def Extract_spans(doc):
    spans = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'tag'])
    rows = []
    for page_num, blocks in Extract_dict(doc).items():
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    for span in line['spans']:
                        xmin, ymin, xmax, ymax = list(span['bbox'])
                        font_size = span['size']
                        text = unidecode(span['text'])
                        span_font = span['font']
                        is_upper = False
                        is_bold = False
                        if "bold" in span_font.lower():
                            is_bold = True
                        if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                            is_upper = True
                        if text.replace(" ","") !=  "":
                            rows.append((xmin, ymin, xmax, ymax, text, is_upper, is_bold, span_font, font_size))
                            span_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])
    return span_df

def Score_span(doc):
    span_scores = []
    span_num_occur = {}
    special = '[(_:/,#%\=@)&]'
    for index, span_row in Extract_spans(doc).iterrows():
        score = round(span_row.font_size)
        text = span_row.text
        if not re.search(special, text):
            if span_row.is_bold:
                score += 1
            if span_row.is_upper:
                score += 1
        span_scores.append(score)
    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = {}
    for value, count in zip(values, counts):
        style_dict[value] = count
    sorted(style_dict.items(), key=lambda x: x[1])
    p_size = max(style_dict, key=style_dict.get)
    idx = 0
    tag = {}
    for size in sorted(values, reverse=True):
        idx += 1
        if size == p_size:
            idx = 0
            tag[size] = 'p'
        if size > p_size:
            tag[size] = 'h{0}'.format(idx)
        if size < p_size:
            tag[size] = 's{0}'.format(idx)
    span_tags = [tag[score] for score in span_scores]
    span_df = Extract_spans(doc)
    span_df['tag'] = span_tags
    return span_df

def correct_end_line(line):
    "".join(line.rstrip().lstrip())
    if line[-1] == "-":
        return True
    else :
        return False

def Category_text(doc):
    headings_list = []
    text_list = []
    tmp = []
    heading = ''
    span_df = Score_span(doc)
    span_df = span_df.loc[span_df.tag.str.contains("h|p")]

    for index, span_row in span_df.iterrows():
        text = span_row.text
        tag = span_row.tag
        if 'h' in tag:
            headings_list.append(text)
            text_list.append(''.join(tmp))
            tmp = []
            heading = text
        else:
            if correct_end_line(text):
                tmp.append(text[:-1])
            else :
                tmp.append(text+" ")

    text_list.append(''.join(tmp))
    text_list = text_list[1:]
    text_df = pd.DataFrame(zip(headings_list, text_list),columns=['heading', 'content'])
    return text_df

def Merge_text(doc):
    s = ""
    for index, row in Category_text(doc).iterrows():
        s += ''.join((row['heading'],'\n',row['content']))
    #return ' '.join(s.split())
    return s