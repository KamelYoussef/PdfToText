from Functions import *
from Tfidf import *
import fitz

my_path = "Pdf_files/test1.pdf"
pdf = fitz.open(my_path)

# print(extract_text(pdf))
# print(extract_dict(pdf))
# print(extract_spans(pdf))
# print(score_span(pdf))
# print(merge_text(pdf))
# print(extract_keywords(merge_text(pdf)))

# save_file("Output/CleanText", merge_text(pdf))
# save_file("Output/Tfidf", generate_summary(merge_text(pdf), 50))
# print(generate_summary(merge_text(pdf), 50))


if __name__ == "__main__":
    # Parsing command line arguments entered by user
    args = parse_args()
    # If File Path
    if os.path.isfile(args["input_path"]):
        # Extracting File Info
        extract_info(input_file=args["input_path"])
        # Process a file
        process_file(
            input_file=args["input_path"],
            output_file=args["output_file"],
            search_str=args["search_str"] if "search_str" in (args.keys()) else None,
            pages=args["pages"],
            action=args["action"],
        )
    # If Folder Path
    elif os.path.isdir(args["input_path"]):
        # Process a folder
        process_folder(
            input_folder=args["input_path"],
            search_str=args["search_str"] if "search_str" in (args.keys()) else None,
            action=args["action"],
            pages=args["pages"],
            recursive=args["recursive"],
        )
