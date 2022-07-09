from genre.hf_model import GENRE
import json
import re
model = GENRE.from_pretrained(
    "./hf_e2e_entity_linking_wiki_abs").eval()

with open('./datasets/emb_train.jsonl', 'r') as json_file:
    json_list = list(json_file)

#claim = {}
for json_str in json_list:
    result = json.loads(json_str)
    predictions = model.sample([" " + result["input"]])
    #print(predictions[0][0]["text"])
    text = predictions[0][0]["text"]
    wikipedia_pages = []
    f = 0
    for character in text:
        if character == "[":
            f = 1
            page = ""
            continue
        if f == 1:
            if character != "]":
                page += character
            else:
                wikipedia_pages.append(page)
                f = 0
    #print(wikipedia_pages)
    #claim["wikipedia_pages"] = wikipedia_pages
json_file.close()
