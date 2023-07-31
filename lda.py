# %%
# Importing necessary libraries
from gensim.models import TfidfModel
import numpy as np
import json

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

# %%
# helper functions for data writing/reading


def load_data(file):
    # Function to load data from a JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_data(file, data):
    # Function to write data to a JSON file
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# %%
# helper functions for lda preprocessing


stopwords = stopwords.words("english") + ["go", "re", "just", "get", "want", "know", "so", "need", "knock",
                                          "look", "guy", "work", "say", "let", "come", "here", "make", "see",  "tell", "thing", "talk", "ve", "really"]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    # Function for lemmatizing the text and removing stopwords
    """https://spacy.io/api/annotation"""
    nlp = spacy.load(
        "/Users/anishpalakurthi/opt/anaconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-3.6.0", disable=["parser", "ner"])
    texts_out = []
    for sent in texts:
        # Lemmatizes each word by appending allowed tokens from the doc object's metadata
        doc = nlp(sent)
        texts_out.append(
            " ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))

    # Removing stopwords from the lemmatized texts
    for word in texts_out:
        if word in stopwords:
            texts_out.remove(word)
    return texts_out


def gen_words(texts):
    # Function to preprocess the lemmatized keywords
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final

# %%
# handle bigrams and trigrams


def make_bigrams(texts):
    # Applying bigram models to the data_words
    bigram_phrases = gensim.models.Phrases(texts, min_count=5, threshold=50)
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    return [bigram[doc] for doc in texts]


def make_trigrams(texts):
    # Applying trigram models to the data_words
    bigram_phrases = gensim.models.Phrases(texts, min_count=5, threshold=50)
    trigram_phrases = gensim.models.Phrases(
        bigram_phrases[texts], threshold=50)
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    return [trigram[bigram[doc]] for doc in texts]


# %%
# Load and preprocess data with helper calls
episodeData = load_data("data/cleaned.json")
plots = [episodeData[episode] for episode in episodeData]
lemmatized_texts = lemmatization(plots)
data_words = gen_words(lemmatized_texts)


# %%
# Create bigrams and trigrams
data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_words)

# %%
# Applying TF-IDF removal to the corpus
id2word = corpora.Dictionary(data_bigrams_trigrams)
texts = data_bigrams_trigrams
corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)

# %%
low_value = 0.03
words = []
words_missing_in_tfidf = []


# Identifying words to be removed based on their TF-IDF values
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

# %%
# Reconstructing the corpus after TF-IDF removal
id2word = corpora.Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]


# %%
# Generating the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5,
                                            random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# Enabling the notebook for visualization
pyLDAvis.enable_notebook()

# Preparing the data for visualization using pyLDAvis
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)

# Displaying the visualization
vis

# %%


def find_closest_episodes(user_input_episode, lda_model, episode_data, num_similar=5):
    # Check if the user input is a valid episode number
    if user_input_episode.isdigit() and int(user_input_episode) <= len(episode_data):
        episode_number = int(user_input_episode)
        target_episode = episode_data[episode_number - 1]
    else:
        # Assuming the user input is the title of an episode
        target_episode = user_input_episode

    # Get the topic distribution vector for the target episode
    target_corpus = id2word.doc2bow(
        simple_preprocess(target_episode, deacc=True))
    target_topics = lda_model.get_document_topics(
        target_corpus, minimum_probability=0)

    similarity_scores = []
    for episode_num, episode_corpus in enumerate(episode_data):
        doc_corpus = id2word.doc2bow(
            simple_preprocess(episode_corpus, deacc=True))
        doc_topics = lda_model.get_document_topics(
            doc_corpus, minimum_probability=0)
        similarity_score = gensim.matutils.cossim(target_topics, doc_topics)
        similarity_scores.append((episode_num + 1, similarity_score))

    # Sort episodes based on similarity scores (descending order)
    sorted_episodes = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar episodes
    top_similar_episodes = sorted_episodes[:num_similar]

    return top_similar_episodes


# User input: enter an episode title
user_input_episode = input("Enter the title of an episode: ")

# Find closest similarity episodes using LDA model
num_similar_episodes = 5
closest_episodes = find_closest_episodes(
    user_input_episode, lda_model, plots, num_similar_episodes)

# Display the closest episodes
print(
    f"\nTop {num_similar_episodes} similar episodes based on LDA topic distribution:")
for episode, similarity_score in closest_episodes:
    print(f"Episode: {episode}, Similarity Score: {similarity_score:.4f}")
