import networkx as nx
import csv
import pprint as pp
import matplotlib.pyplot as plt


def compute_top_k(map__node_id__score, k=20):
    list__node_id__score = [(node_id, score)
                            for node_id, score in map__node_id__score.items()]
    list__node_id__score.sort(key=lambda x: (-x[1], x[0]))
    return list__node_id__score[:k]


input_graph = "./dataset/pkmn_graph_data.tsv"
k = 6
# Graph creation by reading the list of unweighted edges from file
file_handler = open(input_graph, 'r', encoding="utf-8")
csv_reader = csv.reader(file_handler, delimiter='\t',
                        quotechar='"', quoting=csv.QUOTE_NONE)
u_v = []
graph = nx.karate_club_graph()
for record in csv_reader:
    u = record[0]
    v = record[1]
    graph.add_edge(u, v)

file_handler.close()

#nx.draw(graph, with_labels=True, font_weight='bold')
# plt.show()

probability = {}
for node_id in graph:
    for node_id in graph:
        probability[node_id] = 0
    probability[node_id] = 1
    for damping_factor in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        pr = nx.pagerank(graph, alpha=damping_factor,
                         personalization=probability,
                         weight='weight')

        top_k__node_id__node_pagerank_value = compute_top_k(
            pr, k)
        print("SORTED Top-K nodes according to the Personalized-PageRank value.")
        pp.pprint(top_k__node_id__node_pagerank_value)
        print()
