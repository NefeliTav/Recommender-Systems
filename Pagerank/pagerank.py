import networkx as nx
import csv
import pprint as pp
import matplotlib.pyplot as plt


def compute_top_k(map__node_id__score, k=20):
    list__node_id__score = [(node_id, score)
                            for node_id, score in map__node_id__score.items()]
    list__node_id__score.sort(key=lambda x: (-x[1], x[0]))
    return list__node_id__score[:k]


input_graph = "../dataset/pkmn_graph_data.tsv"
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

# nx.draw(graph, with_labels=True, font_weight='bold')
# plt.show()
# print("Number of edges in the graph", graph.number_of_edges())
# print("Number of nodes in the graph", graph.number_of_nodes())
# print()

# pokemon sets
set_A = set(["Pikachu"])
set_B = set(["Venusaur", "Charizard", "Blastoise"])
set_C = set(["Excadrill", "Dracovish", "Whimsicott", "Milotic"])
set_NOT_A = set()
set_NOT_B = set()
set_NOT_C = set()


set_Charizard = (["Charizard"])
set_Venusaur = (["Venusaur"])
set_Kingdra = (["Kingdra"])
set_NOT_Charizard = set()
set_NOT_Venusaur = set()
set_NOT_Kingdra = set()

set_Char_Ven = set(["Charizard", "Venusaur"])
set_Char_King = set(["Charizard", "Kingdra"])
set_Ven_King = set(["Venusaur", "Kingdra"])
set_NOT_Char_Ven = set()
set_NOT_Char_King = set()
set_NOT_Ven_King = set()


for node_id in graph:
    if "Pikachu" not in node_id:
        set_NOT_A.add(node_id)

    if ("Venusaur" not in node_id) and ("Charizard" not in node_id) and ("Blasmax_conductancetoise" not in node_id):
        set_NOT_B.add(node_id)

    if ("Excadrill" not in node_id) and ("Dracovish" not in node_id) and ("Whimsicott" not in node_id) and ("Milotic" not in node_id):
        set_NOT_C.add(node_id)

    if "Charizard" not in node_id:
        set_NOT_Charizard.add(node_id)

    if "Venusaur" not in node_id:
        set_NOT_Venusaur.add(node_id)

    if "Kingdra" not in node_id:
        set_NOT_Kingdra.add(node_id)

    if ("Venusaur" not in node_id) and ("Charizard" not in node_id):
        set_NOT_Char_Ven.add(node_id)

    if ("Kingdra" not in node_id) and ("Charizard" not in node_id):
        set_NOT_Char_King.add(node_id)

    if ("Venusaur" not in node_id) and ("Kingdra" not in node_id):
        set_NOT_Ven_King.add(node_id)


for pokemon_set in [set_A, set_B, set_C, set_Charizard, set_Venusaur, set_Kingdra, set_Char_Ven, set_Char_King, set_Ven_King]:
    probability = {node_id: 1. / len(pokemon_set) for node_id in pokemon_set}

    if (pokemon_set == set_A):
        for node_id in set_NOT_A:
            probability[node_id] = 0
    if (pokemon_set == set_B):
        for node_id in set_NOT_B:
            probability[node_id] = 0
    if (pokemon_set == set_C):
        for node_id in set_NOT_C:
            probability[node_id] = 0
    if (pokemon_set == set_Charizard):
        for node_id in set_NOT_Charizard:
            probability[node_id] = 0
    if (pokemon_set == set_Venusaur):
        for node_id in set_NOT_Venusaur:
            probability[node_id] = 0
    if (pokemon_set == set_Kingdra):
        for node_id in set_NOT_Kingdra:
            probability[node_id] = 0
    if (pokemon_set == set_Char_Ven):
        for node_id in set_NOT_Char_Ven:
            probability[node_id] = 0
    if (pokemon_set == set_Char_King):
        for node_id in set_NOT_Char_King:
            probability[node_id] = 0
    if (pokemon_set == set_Ven_King):
        for node_id in set_NOT_Ven_King:
            probability[node_id] = 0

    # Computation of the PageRank vector.
    pr = nx.pagerank(graph, alpha=0.33,
                     personalization=probability, weight='weight')
    # Extract and print the Top-K node identifiers according to the PageRank score.
    top_k__node_id__node_pagerank_value = compute_top_k(pr, k)
    print("SORTED Top-6 nodes according to the Topic-Specific-PageRank value.")
    pp.pprint(top_k__node_id__node_pagerank_value)
    print()
