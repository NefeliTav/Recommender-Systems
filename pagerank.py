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
graph = nx.Graph()
for record in csv_reader:
    u = record[0]
    v = record[1]
    graph.add_edge(u, v)

file_handler.close()

#nx.draw(graph, with_labels=True, font_weight='bold')
# plt.show()
print("Number of edges in the graph", graph.number_of_edges())
print("Number of nodes in the graph", graph.number_of_nodes())
print()
