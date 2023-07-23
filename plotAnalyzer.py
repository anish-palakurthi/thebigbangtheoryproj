# main runner file for topic modeling via tf-idf w/ scikit-learn in python

# dependencies
import re
import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# remove stop words and superflous expressions from text
def remove_stops(text, stops):

    # removes all stopwords
    words = text.split()
    cleaned = []
    for word in words:
        if word not in stops:
            cleaned.append(word)

    cleaned = " ".join(cleaned)

    # removes punctuation, numbers, and double spaces
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = "".join([i for i in cleaned if not i.isdigit()])
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    return cleaned


# customize stop words to be removed
def clean_text(plots):
    stops = stopwords.words("english")
    final = []

    # cleans each episode's plot and appends to final list
    for episode in plots:
        cleaned = remove_stops(episode, stops)
        final.append(cleaned)

    return final


episodes = load_data("plots.json")

names = [episodes["Title"] for episodes in episodes]
plots = [episodes["plot"] for episodes in episodes]


cleaned_plots = clean_text(plots)
print(plots[0])
print(cleaned_plots[0])
