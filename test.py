from difflib import get_close_matches


def closeMatches(patterns, word):
    return get_close_matches(word, patterns)


line = ["Collaborative Filtering with Temporal Dynamics"]
search_str = "collaborative filtering with temporal dynamics yehuda koren yahoo!"
results = closeMatches(line, search_str)
print(results)
