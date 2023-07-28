from rouge_score import rouge_scorer
from Functions import *
from rouge import Rouge

text = open_file("Output/CleanText")

extractive_summary = open_file("Output/chatgpt")
abstractive_summary = open_file("Output/chatgpt2")


def evaluate_rouge(reference_text, summary_text):
    rouge = Rouge()
    scores = rouge.get_scores(reference_text, summary_text)
    return scores[0]["rouge-1"]["f"]


rouge_score = evaluate_rouge(extractive_summary, abstractive_summary)
print(f"ROUGE score: {rouge_score}")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(extractive_summary, abstractive_summary)
for key in scores:
    print(f"{key}: {scores[key]}")
