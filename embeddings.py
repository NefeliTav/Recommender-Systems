from sklearn.preprocessing import StandardScaler
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

# Get embeddings

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
    json.dump(claims_train, outfile)
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
    # encode
    if result["output"][0]["answer"] == "SUPPORTS":
        data["Label"].append(1)
    elif result["output"][0]["answer"] == "REFUTES":
        data["Label"].append(0)
    i += 1
x_train = np.array(data["Embeddings"])
y_train = np.array(data["Label"])


# Random Forest
param_grid_rf = {'max_features': ["auto", "sqrt", "log2"], 'criterion': [
    "gini", "entropy"], 'max_depth': [None, 5, 10, 12], 'min_samples_split': [2, 5, 7]}
rf = RandomForestClassifier()

# cross validation
grid_rf = GridSearchCV(rf, param_grid_rf)

grid_rf = grid_rf.fit(x_train, y_train)
print(grid_rf.score(x_train, y_train))
# print best parameter after tuning
print(grid_rf.best_params_)
print(grid_rf.best_score_)
print(grid_rf.best_estimator_)
print()


# KNN
param_grid_knn = {'n_neighbors': [3, 5, 7], 'weights': [
    'uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

#ss = StandardScaler()
knn_model = KNeighborsClassifier()

# cross validation
grid_knn = GridSearchCV(knn_model, param_grid_knn)

grid_knn = grid_knn.fit(x_train, y_train)
print(grid_knn.score(x_train, y_train))
print(grid_knn.best_params_)
print(grid_knn.best_score_)
print(grid_knn.best_estimator_)


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


# Random Forest
predicted = pd.DataFrame(grid_rf.predict(x_test))
probs = pd.DataFrame(grid_rf.predict_proba(x_test))

print("Random Forest:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# KNN
predicted = pd.DataFrame(grid_knn.predict(x_test))
probs = pd.DataFrame(grid_knn.predict_proba(x_test))

# Store metrics
print("KNN:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


json_file.close()

# Train
with open('./datasets/emb_train.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = {}
data["Embeddings"] = []
data["Label"] = []
for json_str in json_list:
    result = json.loads(json_str)

    data["Embeddings"].append(np.array(result["claim_embedding"]))
    if result["output"][0]["answer"] == "SUPPORTS":
        data["Label"].append(1)
    elif result["output"][0]["answer"] == "REFUTES":
        data["Label"].append(0)
x_train = np.array(data["Embeddings"])
y_train = np.array(data["Label"])


# Random Forest
rf = RandomForestClassifier(
    max_features='sqrt', criterion='entropy')

rf = rf.fit(x_train, y_train)
print(rf.score(x_train, y_train))


# KNN
knn_model = KNeighborsClassifier(
    algorithm='auto', n_neighbors=7, weights='distance')

knn_model = knn_model.fit(x_train, y_train)
print(knn_model.score(x_train, y_train))

json_file.close()


# Test
with open('./datasets/emb_dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
data = {}
data["Embeddings"] = []
data["Label"] = []
for json_str in json_list:
    result = json.loads(json_str)

    data["Embeddings"].append(np.array(result["claim_embedding"]))
    if result["output"][0]["answer"] == "SUPPORTS":
        data["Label"].append(1)
    elif result["output"][0]["answer"] == "REFUTES":
        data["Label"].append(0)

x_test = np.array(data["Embeddings"])
y_test = np.array(data["Label"])


# Random Forest
predicted = pd.DataFrame(rf.predict(x_test))

print("Random Forest:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# KNN
predicted = pd.DataFrame(knn_model.predict(x_test))

print("KNN:")
print(metrics.accuracy_score(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


json_file.close()

# Apply model - Classify
with open('./datasets/emb_test.jsonl', 'r') as json_file:
    json_list = list(json_file)
data = {}
data["Embeddings"] = []
data["Label"] = []
data["id"] = []  # i need id for the output file
for json_str in json_list:
    result = json.loads(json_str)

    data["id"].append(result["id"])
    data["Embeddings"].append(np.array(result["claim_embedding"]))
    # i dont care about this column ,so i put 0 everywhere
    data["Label"].append(0)

x_test = np.array(data["Embeddings"])
y_test = np.array(data["Label"])

# Random Forest
outfile = open('./datasets/test_set_pred_1.jsonl', 'w')
predicted = pd.DataFrame(rf.predict(x_test))
arr = predicted.values.tolist()

# clean list and convert 0,1 to Refutes,Supports
clean_arr = []
for i in arr:
    for j in i:
        if j == 1:
            j = "SUPPORTS"
        elif j == 0:
            j = "REFUTES"
        clean_arr.append(j)
results = {}
for id, output in zip(data["id"], clean_arr):
    results["id"] = id
    res = {}
    res["answer"] = output
    results["output"] = [res]
    json.dump(results, outfile)
    # write results in output file
    outfile.write('\n')
outfile.close()

# KNN
outfile = open('./datasets/test_set_pred_2.jsonl', 'w')
predicted = pd.DataFrame(knn_model.predict(x_test))
arr = predicted.values.tolist()

clean_arr = []
for i in arr:
    for j in i:
        if j == 1:
            j = "SUPPORTS"
        elif j == 0:
            j = "REFUTES"
        clean_arr.append(j)
results = {}
for id, output in zip(data["id"], clean_arr):
    results["id"] = id
    res = {}
    res["answer"] = output
    results["output"] = [res]
    json.dump(results, outfile)
    outfile.write('\n')
outfile.close()

json_file.close()
