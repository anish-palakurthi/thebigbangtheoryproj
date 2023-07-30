# %%
from top2vec import Top2Vec
import json


# %%
# prepping data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# %%
# loads descriptions of episodes
episodes = load_data("data/plots.json")

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
data = load_data("data/scripts.json")

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


# %%
print((plots[0]))

# %%

model = Top2Vec(plots)


# %%
topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_nums)

topic_words, word_scores, topic_nums = model.get_topics(2)

# %%
for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(f"Topic #{num}: words")
