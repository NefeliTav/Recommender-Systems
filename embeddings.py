import numpy as np
import json
from sentence_transformers import SentenceTransformer, models
model = SentenceTransformer('paraphrase-distilroberta-base-v1')


with open('./datasets/train_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./datasets/emb_train.jsonl', 'w')
sentences = []
i = 0
for json_str in json_list:
    result = json.loads(json_str)
    claims_train = {}
    claims_train["id"] = result["id"]
    claims_train["input"] = result["input"]
    output = {}
    output["answer"] = result["output"][0]["answer"]
    claims_train["output"] = [output]
    sentences.append(result["input"])

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        claims_train["claim_embedding"] = embedding
        claims_train["claim_embedding"] = claims_train["claim_embedding"].tolist()
    json.dump(claims_train, outfile)
    outfile.write('\n')
    i += 1
json_file.close()
outfile.close()


with open('./datasets/dev_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./datasets/emb_dev.jsonl', 'w')
sentences = []
i = 0
for json_str in json_list:
    result = json.loads(json_str)
    claims_dev = {}
    claims_dev["id"] = result["id"]
    claims_dev["input"] = result["input"]
    output = {}
    output["answer"] = result["output"][0]["answer"]
    claims_dev["output"] = [output]
    sentences.append(result["input"])
    embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, embeddings):
        claims_dev["claim_embedding"] = embedding
        claims_dev["claim_embedding"] = claims_dev["claim_embedding"].tolist()
    json.dump(claims_dev, outfile)
    outfile.write('\n')
    i += 1
json_file.close()
outfile.close()


with open('./datasets/test_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./emb_test.jsonl', 'w')
sentences = []
i = 0
for json_str in json_list:
    result = json.loads(json_str)
    claims_test = {}
    claims_test["id"] = result["id"]
    claims_test["input"] = result["input"]
    sentences.append(result["input"])
    embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, embeddings):
        claims_test["claim_embedding"] = embedding
        claims_test["claim_embedding"] = claims_test["claim_embedding"].tolist()
    json.dump(claims_test, outfile)
    outfile.write('\n')
    i += 1
json_file.close()
outfile.close()
