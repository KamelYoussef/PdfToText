# importing the required libraries
import Functions
from Functions import *

# importing TfidfVectorizer class to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer

# importing cosine_similarity function to compute the cosine similarity between two vectors.
from sklearn.metrics.pairwise import cosine_similarity

# importing nlargest to return the n largest elements from an iterable in descending order.
from heapq import nlargest

from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
)


def generate_summary(text, n):
    # Tokenize the text into individual sentences
    sentences = sent_tokenize(text)

    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute the cosine similarity between each sentence and the document
    sentence_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Select the top n sentences with the highest scores
    summary_sentences = nlargest(
        n, range(len(sentence_scores)), key=sentence_scores.__getitem__
    )

    summary_tfidf = " ".join([sentences[i] for i in sorted(summary_sentences)])

    return summary_tfidf


#text = Functions.open_file("Output/CleanText")

#save_file("Output/Tfidf", str(generate_summary(text, 50)))
