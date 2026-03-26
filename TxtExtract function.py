# ------------------------------------------------------------
# text_extractor()
# Summary length is now one of: "short", "medium", "long"
#
# Extractive mode:
#     short  -> 2 sentences
#     medium -> 4 sentences
#     long   -> 6 sentences
#
# Abstractive mode:
#     short  -> ~50 tokens
#     medium -> ~100 tokens
#     long   -> ~200 tokens
#
# This matches the user story US2.4 exactly.
# ------------------------------------------------------------

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from transformers import pipeline

nltk.download('punkt', quiet=True)

def text_extractor(text, mode="extractive", length="medium"):
    """
    Generate extractive or abstractive summaries with 3-level length control.

    Parameters:
        text (str): Cleaned input text.
        mode (str): "extractive" or "abstractive".
        length (str): "short", "medium", or "long".

    Returns:
        str: Summary text.
    """

    # ------------------------------------------------------------
    # 1. Map length → number of sentences or tokens
    # ------------------------------------------------------------
    extractive_map = {
        "short": 2,
        "medium": 4,
        "long": 6
    }

    abstractive_map = {
        "short": 50,
        "medium": 100,
        "long": 200
    }

    # Validate length input
    if length not in extractive_map:
        return "Invalid length. Choose 'short', 'medium', or 'long'."

    # ------------------------------------------------------------
    # 2. Extractive Summary
    # ------------------------------------------------------------
    if mode == "extractive":
        sentences = sent_tokenize(text)

        # Build word frequency dictionary
        word_freq = defaultdict(int)
        for word in word_tokenize(text.lower()):
            if word.isalpha():
                word_freq[word] += 1

        # Score each sentence
        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_freq:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

        # Rank sentences by score
        ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

        # Select top N sentences based on length
        n = extractive_map[length]
        summary = " ".join(ranked_sentences[:n])
        return summary

    # ------------------------------------------------------------
    # 3. Abstractive Summary
    # ------------------------------------------------------------
    elif mode == "abstractive":
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        max_tokens = abstractive_map[length]

        result = summarizer(
            text,
            max_length=max_tokens,
            min_length=int(max_tokens * 0.5),
            do_sample=False
        )

        return result[0]["summary_text"]

    # ------------------------------------------------------------
    # 4. Invalid Mode
    # ------------------------------------------------------------
    else:
        return "Invalid mode. Choose 'extractive' or 'abstractive'."