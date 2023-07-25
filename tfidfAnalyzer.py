# main runner file for topic modeling via tf-idf w/ scikit-learn in python
# dependencies
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


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
        if word.lower() in stops or word[-2:].lower() == "'s":
            continue
        else:
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
    stops += ["raj", "leonard", "howard", "penny", "bernadette", "sheldon", "stuart", "amy",
              "koothrappali", "wolowitz", "hofstadter", "cooper", "fowler", "rostenkowski", "bloom", "title", "reference"]
    final = []

    # cleans each episode's plot and appends to final list
    for episode in plots:
        cleaned = remove_stops(episode, stops)
        final.append(cleaned)

    return final


# loads descriptions of episodes
episodes = load_data("plots.json")

# episode titles in the form of "Season;Episode"
names = [str(episode["Season"]) + ";" + str(episode["No. inseason"])
         for episode in episodes]

# maps episode titles to their index in the names array
names_map = {}
for index in range(len(names)):
    names_map[names[index]] = index

# episode summaries
plots = [episode["plot"] for episode in episodes]

# episode summaries in the form of "Season;Episode\nSummary"
summaries = [str(episode["Season"]) + ";" + str(episode["No. inseason"]
                                                ) + "\n" + str(episode["plot"]) for episode in episodes]

# loads episode scripts
episodes_data = {}
data = load_data("scripts.json")

for line in data:
    episode_title = line["episode_name"]
    script_line = line["dialogue"]

    if episode_title not in episodes_data:
        episodes_data[episode_title] = ""

    episodes_data[episode_title] += script_line + " "

episodes_array = []
# Iterate through the episodes_data dictionary and extract the script lines
for lines in episodes_data.values():
    # Combine all script lines into a single string for each episode
    episode_string = "".join(lines)
    episodes_array.append(episode_string)

# clean up the episode scripts and combine them with the episode summaries
for i in range(len(episodes_array)):
    episodes_array[i] = episodes_array[i].replace("\\n", " ")
    episodes_array[i] = episodes_array[i].replace("\\", "")
    episodes_array[i] = episodes_array[i].replace("  ", " ")
    plots[i] = plots[i] + " " + episodes_array[i]

cleaned_plots = clean_text(plots)

# TF-IDF Vectorizer parameters
vectorizer = TfidfVectorizer(lowercase=True,
                             max_features=100,  # max number of words to keep
                             max_df=0.8,  # ignore words that appear in more than 80% of the documents
                             min_df=5,  # ignore words that appear in less than 5 documents
                             # unigrams, bigrams, and trigrams
                             ngram_range=(1, 3),
                             stop_words='english'  # remove stop words again just in case
                             )

# training the vectorizer
vectors = vectorizer.fit_transform(cleaned_plots)
feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
dense_list = dense.tolist()

# stores keywords for each episode's plot
allKeywords = []

# out of the main feature words that vectorizer found, which ones are in each episode's plot?
# stores them per and for each episode
for desc in dense_list:
    x = 0
    keywords = []
    for word in desc:
        if word > 0:
            keywords.append(feature_names[x])
        x += 1

    allKeywords.append(keywords)

# K-Means Clustering ### -- limits each desc to one cluster/topic -- LDA will improve this
cluster_num = 10

# clusters descriptions into 20 clusters
model = KMeans(n_clusters=cluster_num,
               init='k-means++', max_iter=100, n_init=1)
model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

### PCA ###
# preps for 2d visualization of clusters
kmean_indices = model.fit_predict(vectors)
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r", "b", "g", "c", "m", "y", "k", "w", "orange", "purple", "pink",
          "brown", "gray", "olive", "cyan", "lime", "teal", "navy", "maroon", "gold"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig, ax = plt.subplots(figsize=(50, 50))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

for i, txt in enumerate(names):
    ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))

plt.savefig("clusters.png")

# takes user input for episode and number of recommendations
user_episode = input("Enter the name of an episode: ")
user_ep_index = names_map[user_episode]
num_recs = int(input("Enter how many recommendations you want: "))
num_recs_adj = num_recs * -1 - 1

target_vector = vectors[user_ep_index]
sims = cosine_similarity(target_vector, vectors)
closest = np.argsort(sims)[0][num_recs:-1]

print("Requested Episode:")
print(summaries[user_ep_index])

# prints recommendations
for i in range(num_recs):
    print("\n")
    print(summaries[closest[i]])
