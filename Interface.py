import glob
import os
import sys
from Tfidf import *
from Functions import *
import gradio as gr
import fitz

sys.path.insert(0, os.path.abspath("./src"))


def summarize_text(text, number_sentences):
    return generate_summary(text, number_sentences)


input = gr.inputs.File(type="file", label="PDF File")
number_sentences = gr.inputs.Slider(1, 100, 1)
output_text = gr.outputs.Textbox()

gr.Interface(summarize_text, ["textbox", number_sentences], output_text).launch()
