from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm
import json
from sentence_transformers import SentenceTransformer, models
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

'''
with open('./datasets/train_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./datasets/emb_train.jsonl', 'w')

for json_str in json_list:
    result = json.loads(json_str)
    claims_train = {}

    claims_train["id"] = result["id"]
    claims_train["input"] = result["input"]

    output = {}
    output["answer"] = result["output"][0]["answer"]
    claims_train["output"] = [output]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(result["input"])
    # Print the embeddings
    claims_train["claim_embedding"] = embeddings.tolist()

    outfile.write('\n')

json_file.close()
outfile.close()


with open('./datasets/dev_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./datasets/emb_dev.jsonl', 'w')
sentences = []
for json_str in json_list:
    result = json.loads(json_str)
    claims_dev = {}

    claims_dev["id"] = result["id"]
    claims_dev["input"] = result["input"]

    output = {}
    output["answer"] = result["output"][0]["answer"]
    claims_dev["output"] = [output]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(result["input"])
    # Print the embeddings
    claims_dev["claim_embedding"] = embeddings.tolist()

    json.dump(claims_dev, outfile)
    outfile.write('\n')
json_file.close()
outfile.close()


with open('./datasets/test_set.jsonl', 'r') as json_file:
    json_list = list(json_file)

outfile = open('./datasets/emb_test.jsonl', 'w')
sentences = []
for json_str in json_list:
    result = json.loads(json_str)
    claims_test = {}
    claims_test["id"] = result["id"]
    claims_test["input"] = result["input"]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(result["input"])
    # Print the embeddings
    claims_test["claim_embedding"] = embeddings.tolist()

    json.dump(claims_test, outfile)
    outfile.write('\n')
json_file.close()
outfile.close()

'''
# Train
with open('./datasets/emb_train.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = {}
data["Embeddings"] = []
data["Label"] = []
i = 0
for json_str in json_list:
    result = json.loads(json_str)

    data["Embeddings"].append(np.array(result["claim_embedding"]))
    if result["output"][0]["answer"] == "SUPPORTS":
        data["Label"].append(1)
    elif result["output"][0]["answer"] == "REFUTES":
        data["Label"].append(0)
    i += 1
x_train = np.array(data["Embeddings"])
y_train = np.array(data["Label"])


param_grid_rf = {'max_features': ["auto", "sqrt", "log2"], 'criterion': [
    "gini", "entropy"], 'max_depth': [None, 5, 10, 12], 'min_samples_split': [2, 5, 7]}
rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, param_grid_rf)

grid_rf = grid_rf.fit(x_train, y_train)
print(grid_rf.score(x_train, y_train))  # = 0.9992569022350094
# print best parameter after tuning
print(grid_rf.best_params_)
print()

param_grid_knn = {'n_neighbors': [3, 5, 10], 'weights': ["uniform", "distance"], 'algorithm': [
    "auto", "ball_tree", "brute"], 'leaf_size': [30, 40, 50], 'p': [1, 2, 5]}


knn_model = KNeighborsClassifier()
grid_knn = GridSearchCV(knn_model, param_grid_knn)

grid_knn = grid_knn.fit(x_train, y_train)
print(grid_knn.score(x_train, y_train))  # = 0.8259531657870167
print(grid_knn.best_params_)

json_file.close()


# Test
with open('./datasets/emb_dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
data = {}
data["Embeddings"] = []
data["Label"] = []
i = 0
for json_str in json_list:
    result = json.loads(json_str)

    data["Embeddings"].append(np.array(result["claim_embedding"]))
    if result["output"][0]["answer"] == "SUPPORTS":
        data["Label"].append(1)
    elif result["output"][0]["answer"] == "REFUTES":
        data["Label"].append(0)
    i += 1

x_test = np.array(data["Embeddings"])
y_test = np.array(data["Label"])

predicted = pd.DataFrame(grid_rf.predict(x_test))
probs = pd.DataFrame(grid_rf.predict_proba(x_test))

print("Random Forest:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

# Evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(
    RandomForestClassifier(), x_test, y_test, scoring='precision', cv=10)
print(np.mean(rf_cv_scores))


predicted = pd.DataFrame(grid_knn.predict(x_test))
probs = pd.DataFrame(grid_knn.predict_proba(x_test))

# Store metrics
print("KNN:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

# Evaluate the model using 10-fold cross-validation
knn_cv_scores = cross_val_score(
    KNeighborsClassifier(), x_test, y_test, scoring='precision', cv=10)
print(np.mean(knn_cv_scores))

json_file.close()
