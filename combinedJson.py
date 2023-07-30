import json
from nltk.corpus import stopwords


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


desc = load_data("data/plots.json")
desc = [episode["plot"] for episode in desc]


scripts = load_data("data/scripts.json")
epScripts = {}
stops = stopwords.words("english") + ["Raj", "Leonard", "Howard", "Penny", "Bernadette", "Sheldon", "Stuart", "Amy", "Koothrappali", "Wolowitz", "Hofstadter", "Cooper", "Fowler", "Rostenkowski", "Bloom",
                                      "Title", "Reference", "go", "re", "just", "get", "want", "know", "so", "need", "knock", "look", "guy", "work", "say", "let", "come", "here", "make", "see",  "tell", "thing", "talk", "ve", "really"]

for episode in scripts:

    dialogueArr = episode["dialogue"].split(" ")

    for word in dialogueArr:
        if word in stops:
            dialogueArr.remove(word)

    epScripts[episode["episode_name"]] = epScripts.get(
        episode["episode_name"], "") + (" ".join(dialogueArr))


write_data("cleaned.json", epScripts)
