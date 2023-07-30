from unidecode import unidecode
import re
import pandas as pd
import numpy as np
import unicodedata
import contractions
import string
import yake
from typing import Tuple
from io import BytesIO
import os
import argparse
import fitz
from Tfidf import *


def extract_text(doc):
    output = []
    raw = ""
    for page in doc:
        output += page.get_text("blocks")
    for block in output:
        if block[6] == 0:  # We only take the text
            plain_text = unidecode(block[4])  # Encode in ASCII
            raw += plain_text
    return raw


def extract_dict(doc):
    block_dict = {}
    page_num = 1
    for page in doc:  # Iterate all pages in the document
        file_dict = page.get_text("dict")  # Get the page dictionary
        block = file_dict["blocks"]  # Get the block information
        block_dict[page_num] = block  # Store in block dictionary
        page_num += 1  # Increase the page value by 1
    return block_dict


def extract_spans(doc):
    spans = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "text", "tag"])
    rows = []
    for page_num, blocks in extract_dict(doc).items():
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        xmin, ymin, xmax, ymax = list(span["bbox"])
                        font_size = span["size"]
                        text = unidecode(span["text"])
                        span_font = span["font"]
                        is_upper = False
                        is_bold = False
                        if "bold" in span_font.lower():
                            is_bold = True
                        if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                            is_upper = True
                        if text.replace(" ", "") != "":
                            rows.append(
                                (
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                    text,
                                    is_upper,
                                    is_bold,
                                    span_font,
                                    font_size,
                                )
                            )
                            span_df = pd.DataFrame(
                                rows,
                                columns=[
                                    "xmin",
                                    "ymin",
                                    "xmax",
                                    "ymax",
                                    "text",
                                    "is_upper",
                                    "is_bold",
                                    "span_font",
                                    "font_size",
                                ],
                            )
    return span_df


def score_span(doc):
    span_scores = []
    span_num_occur = {}
    special = "[(_:/,#%\=@)&]"
    for index, span_row in extract_spans(doc).iterrows():
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
            tag[size] = "p"
        if size > p_size:
            tag[size] = "h{0}".format(idx)
        if size < p_size:
            tag[size] = "s{0}".format(idx)
    span_tags = [tag[score] for score in span_scores]
    span_df = extract_spans(doc)
    span_df["tag"] = span_tags
    return span_df


def correct_end_line(line):
    "".join(line.rstrip().lstrip())
    if line[-1] == "-":
        return True
    else:
        return False


def category_text(doc):
    headings_list = []
    text_list = []
    tmp = []
    heading = ""
    span_df = score_span(doc)
    span_df = span_df.loc[span_df.tag.str.contains("h|p")]

    for index, span_row in span_df.iterrows():
        text = span_row.text
        tag = span_row.tag
        if "h" in tag:
            headings_list.append(text)
            text_list.append("".join(tmp))
            tmp = []
            heading = text
        else:
            if correct_end_line(text):
                tmp.append(text[:-1])
            else:
                tmp.append(text + " ")

    text_list.append("".join(tmp))
    text_list = text_list[1:]
    text_df = pd.DataFrame(
        zip(headings_list, text_list), columns=["heading", "content"]
    )
    return text_df


def merge_text(doc):
    s = ""
    for index, row in category_text(doc).iterrows():
        s += "".join((row["heading"], "\n", row["content"]))
    return clean_text(" ".join(s.split()))
    # return s


def clean_text(text):
    out = to_lowercase(text)
    out = standardize_accented_chars(out)
    out = remove_url(out)
    out = expand_contractions(out)
    out = remove_mentions_and_tags(out)
    out = remove_special_characters(out)
    out = remove_spaces(out)
    return out


def to_lowercase(text):
    return text.lower()


def standardize_accented_chars(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_url(text):
    return re.sub(r"https?:\S*", "", text)


def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    return " ".join(expanded_words)


def remove_mentions_and_tags(text):
    text = re.sub(r"@\S*", "", text)
    return re.sub(r"#\S*", "", text)


def remove_special_characters(text):
    # define the pattern to keep
    pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
    return re.sub(pat, "", text)


def remove_spaces(text):
    return re.sub(" +", " ", text)


def remove_punctuation(text):
    return "".join([c for c in text if c not in string.punctuation])


def remove_stopwords(text, nlp):
    filtered_sentence = []
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False:
            filtered_sentence.append(token.text)
    return " ".join(filtered_sentence)


def lemmatize(text, nlp):
    doc = nlp(text)
    lemmatized_text = []
    for token in doc:
        lemmatized_text.append(token.lemma_)
    return " ".join(lemmatized_text)


def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, v in keywords]


def save_file(name, doc):
    with open(name + ".txt", "w") as file:
        file.write(doc)


def open_file(name):
    with open(name + ".txt", "r") as file:
        return file.read()


def extract_info(input_file: str):
    """
    Extracts file info
    """
    # Open the PDF
    pdfDoc = fitz.open(input_file)
    output = {
        "File": input_file,
        "Encrypted": ("True" if pdfDoc.isEncrypted else "False"),
    }
    # If PDF is encrypted the file metadata cannot be extracted
    if not pdfDoc.isEncrypted:
        for key, value in pdfDoc.metadata.items():
            output[key] = value

    # To Display File Info
    print("## File Information ##################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in output.items()))
    print("######################################################################")

    return True, output


def search_for_text(lines, search_str):
    """
    Search for the search string within the document lines
    """
    for line in lines:
        # Find all matches within one line
        # results = re.findall(search_str, line, re.IGNORECASE)
        results = closeMatches([line], search_str)
        # In case multiple matches within one line
        for result in results:
            yield result


def redact_matching_data(page, matched_values):
    """
    Redacts matching values
    """
    matches_found = 0
    # Loop throughout matching values
    for val in matched_values:
        matches_found += 1
        matching_val_area = page.searchFor(val)
        # Redact matching values
        [
            page.addRedactAnnot(area, text=" ", fill=(0, 0, 0))
            for area in matching_val_area
        ]
    # Apply the redaction
    page.apply_redactions()
    return matches_found


def frame_matching_data(page, matched_values):
    """
    frames matching values
    """
    matches_found = 0
    # Loop throughout matching values
    for val in matched_values:
        matches_found += 1
        matching_val_area = page.search_for(val)
        for area in matching_val_area:
            if isinstance(area, fitz.fitz.Rect):
                # Draw a rectangle around matched values
                annot = page.add_rect_annot(area)
                # , fill = fitz.utils.getColor('black')
                annot.set_colors(stroke=fitz.utils.getColor("red"))
                # If you want to remove matched data
                # page.addFreetextAnnot(area, ' ')
                annot.update()
    return matches_found


def highlight_matching_data(page, matched_values, type):
    """
    Highlight matching values
    """
    matches_found = 0
    # Loop throughout matching values
    for val in matched_values:
        matches_found += 1
        matching_val_area = page.search_for(val)
        # print("matching_val_area",matching_val_area)
        highlight = None
        if type == "Highlight":
            highlight = page.add_highlight_annot(matching_val_area)
        elif type == "Squiggly":
            highlight = page.add_squiggly_annot(matching_val_area)
        elif type == "Underline":
            highlight = page.add_underline_annot(matching_val_area)
        elif type == "Strikeout":
            highlight = page.addStrikeoutAnnot(matching_val_area)
        else:
            highlight = page.add_highlight_annot(matching_val_area)
        # To change the highlight colar
        # highlight.setColors({"stroke":(0,0,1),"fill":(0.75,0.8,0.95) })
        # highlight.setColors(stroke = fitz.utils.getColor('white'), fill = fitz.utils.getColor('red'))
        # highlight.setColors(colors= fitz.utils.getColor('red'))
        highlight.update()
    return matches_found


def process_data(
    input_file: str,
    output_file: str,
    search_str: str,
    pages: Tuple = None,
    action: str = "Highlight",
):
    """
    Process the pages of the PDF File
    """
    # Open the PDF
    pdfDoc = fitz.open(input_file)
    # Save the generated PDF to memory buffer
    output_buffer = BytesIO()
    total_matches = 0
    # Iterate through pages
    for pg in range(pdfDoc.page_count):
        # If required for specific pages
        if pages:
            if str(pg) not in pages:
                continue
        # Select the page
        page = pdfDoc[pg]
        # Get Matching Data
        # Split page by lines
        page_lines = page.get_text("text").split("\n")
        matched_values = search_for_text(page_lines, search_str)
        if matched_values:
            if action == "Redact":
                matches_found = redact_matching_data(page, matched_values)
            elif action == "Frame":
                matches_found = frame_matching_data(page, matched_values)
            elif action in ("Highlight", "Squiggly", "Underline", "Strikeout"):
                matches_found = highlight_matching_data(page, matched_values, action)
            else:
                matches_found = highlight_matching_data(
                    page, matched_values, "Highlight"
                )
            total_matches += matches_found
    print(
        f"{total_matches} Match(es) Found of Search String {search_str} In Input File: {input_file}"
    )
    # Save to output
    pdfDoc.save(output_buffer)
    pdfDoc.close()
    # Save the output buffer to the output file
    with open(output_file, mode="wb") as f:
        f.write(output_buffer.getbuffer())


def remove_highlght(input_file: str, output_file: str, pages: Tuple = None):
    # Open the PDF
    pdfDoc = fitz.open(input_file)
    # Save the generated PDF to memory buffer
    output_buffer = BytesIO()
    # Initialize a counter for annotations
    annot_found = 0
    # Iterate through pages
    for pg in range(pdfDoc.page_count):
        # If required for specific pages
        if pages:
            if str(pg) not in pages:
                continue
        # Select the page
        page = pdfDoc[pg]
        annot = page.first_annot
        while annot:
            annot_found += 1
            page.delete_annot(annot)
            annot = annot.next
    if annot_found >= 0:
        print(f"Annotation(s) Found In The Input File: {input_file}")
    # Save to output
    pdfDoc.save(output_buffer)
    pdfDoc.close()
    # Save the output buffer to the output file
    with open(output_file, mode="wb") as f:
        f.write(output_buffer.getbuffer())


def process_file(**kwargs):
    """
    To process one single file
    Redact, Frame, Highlight... one PDF File
    Remove Highlights from a single PDF File
    """
    input_file = kwargs.get("input_file")
    output_file = kwargs.get("output_file")
    if output_file is None:
        output_file = input_file
    search_str = kwargs.get("search_str")
    pages = kwargs.get("pages")
    # Redact, Frame, Highlight, Squiggly, Underline, Strikeout, Remove
    action = kwargs.get("action")
    if action == "Remove":
        # Remove the Highlights except Redactions
        remove_highlght(input_file=input_file, output_file=output_file, pages=pages)
    else:
        text = Functions.open_file("Output/CleanText")
        s = generate_summary(text, 50)
        for str in s:
            process_data(
                input_file=input_file,
                output_file=output_file,
                search_str=str,
                pages=pages,
                action=action,
            )


def process_folder(**kwargs):
    """
    Redact, Frame, Highlight... all PDF Files within a specified path
    Remove Highlights from all PDF Files within a specified path
    """
    input_folder = kwargs.get("input_folder")
    search_str = kwargs.get("search_str")
    # Run in recursive mode
    recursive = kwargs.get("recursive")
    # Redact, Frame, Highlight, Squiggly, Underline, Strikeout, Remove
    action = kwargs.get("action")
    pages = kwargs.get("pages")
    # Loop though the files within the input folder.
    for foldername, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            # Check if pdf file
            if not filename.endswith(".pdf"):
                continue
            # PDF File found
            inp_pdf_file = os.path.join(foldername, filename)
            print("Processing file =", inp_pdf_file)
            process_file(
                input_file=inp_pdf_file,
                output_file=None,
                search_str=search_str,
                action=action,
                pages=pages,
            )
        if not recursive:
            break


def is_valid_path(path):
    """
    Validates the path inputted and checks whether it is a file path or a folder path
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        type=is_valid_path,
        required=True,
        help="Enter the path of the file or the folder to process",
    )
    parser.add_argument(
        "-a",
        "--action",
        dest="action",
        choices=[
            "Redact",
            "Frame",
            "Highlight",
            "Squiggly",
            "Underline",
            "Strikeout",
            "Remove",
        ],
        type=str,
        default="Highlight",
        help="Choose whether to Redact or to Frame or to Highlight or to Squiggly or to Underline or to Strikeout or to Remove",
    )
    parser.add_argument(
        "-p",
        "--pages",
        dest="pages",
        type=tuple,
        help="Enter the pages to consider e.g.: [2,4]",
    )
    action = parser.parse_known_args()[0].action
    if action != "Remove":
        parser.add_argument(
            "-s",
            "--search_str",
            dest="search_str",  # lambda x: os.path.has_valid_dir_syntax(x)
            type=is_valid_path,
            required=True,
            help="Enter a valid search string",
        )
    path = parser.parse_known_args()[0].input_path
    if os.path.isfile(path):
        parser.add_argument(
            "-o",
            "--output_file",
            dest="output_file",
            type=str,  # lambda x: os.path.has_valid_dir_syntax(x)
            help="Enter a valid output file",
        )
    if os.path.isdir(path):
        parser.add_argument(
            "-r",
            "--recursive",
            dest="recursive",
            default=False,
            type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
            help="Process Recursively or Non-Recursively",
        )
    args = vars(parser.parse_args())
    # To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in args.items()))
    print("######################################################################")
    return args


from difflib import get_close_matches


def closeMatches(patterns, word):
    return get_close_matches(word, patterns, 1, 0.4)
