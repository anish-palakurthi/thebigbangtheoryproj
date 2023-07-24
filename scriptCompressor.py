import json


with open("scripts.json", "r", encoding="utf-8") as f:
    data = json.load(f)


episodes_data = {}


print(data)


for line in data:
    episode_title = line["episode_name"]
    script_line = line["dialogue"]

    if episode_title not in episodes_data:
        episodes_data[episode_title] = ""

    episodes_data[episode_title] += script_line + " "


with open("scripts_combined.json", "w", encoding="utf-8") as f:
    json.dump(episodes_data, f, indent=4, ensure_ascii=False)
