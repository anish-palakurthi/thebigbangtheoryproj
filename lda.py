from gensim.models import TfidfModel
import numpy as np
import json
import glob


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# #spacy
import spacy
from nltk.corpus import stopwords

# visualization
import pyLDAvis
import pyLDAvis.gensim

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# prepping data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


stopwords = stopwords.words("english") + ["go", "re", "just", "get", "want", "know", "so", "need", "knock",
                                          "look", "guy", "work", "say", "let", "come", "here", "make", "see",  "tell", "thing", "talk", "ve", "really"]


episodeData = load_data("cleaned.json")

plots = [episodeData[episode] for episode in episodeData]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    nlp = spacy.load(
        "/Users/anishpalakurthi/opt/anaconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-3.6.0", disable=["parser", "ner"])
    texts_out = []
    for sent in texts:
        # contains metadata about the word
        doc = nlp(sent)
        # lemmatizes each word by appending allowed tokens from the doc object's metadata
        texts_out.append(
            " ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))

    for word in texts_out:
        if word in stopwords:
            texts_out.remove(word)
    return texts_out


print("path successfully hit")
lemmatized_texts = lemmatization(plots)
print(lemmatized_texts[0])


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)

    return final


# preprocess our lemmatized keywords
data_words = gen_words(lemmatized_texts)

print(data_words[0][0:10])

# bigrams and trigrams
bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50)
trigram_phrases = gensim.models.Phrases(
    bigram_phrases[data_words], threshold=50)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)


def make_bigrams(texts):
    return [bigram[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]


data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_words)


print(data_bigrams_trigrams[0])


# tfidf removal

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []

words_missing_in_tfidf = []

for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]

    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])

    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]

    new_bow = [b for b in bow if b[0]
               not in low_value_words and b[0] not in words_missing_in_tfidf]

    corpus[i] = new_bow


id2word = corpora.Dictionary(data_words)

corpus = [id2word.doc2bow(text) for text in data_words]

for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

# [index, frequency]
print(corpus[0][0:10])


# generate LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5,
                                            random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis
