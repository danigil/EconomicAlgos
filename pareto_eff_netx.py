import networkx as nx
import itertools
from typing import List, Union
import numpy as np

# def get_value(valuations: list[list[float]], agent, item):
#     return valuations[agent][item]

def is_pareto_efficient(valuations: List[List[float]], allocation: List[List[float]]) -> Union[bool, List[List[float]]]:
    n_participants = len(valuations)
    n_items = len(valuations[0])

    def construct_switching_graph(allocation: List[List[float]]) -> nx.DiGraph:
        G = nx.DiGraph()
        for i, j in itertools.permutations(range(n_participants), 2):
            i_items = set([k for k in range(n_items) if allocation[i][k] > 0])
            ratios = [valuations[i][item]/valuations[j][item] for item in i_items]
            min_ratio = min(ratios)

            G.add_edge(i, j, weight=min_ratio)

        return G

    G = construct_switching_graph(allocation)
    # for edge in G.edges:
    #     print(edge, G[edge[0]][edge[1]]['weight'])
    cycles = list(nx.simple_cycles(G))
    for cycle in cycles:
        # print(f'cycle: {cycle}')
        # print([(i,j) for i,j in zip(cycle, cycle[1:]+[cycle[0]])])
        weights = [G[i][j]['weight'] for i,j in zip(cycle, cycle[1:]+[cycle[0]])]
        if np.prod(weights) < 1:
            return False
    return True
   


def draw_digraph(G: nx.DiGraph):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=6)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.axis('off')
    plt.show()

def test():
    def test1():
        valuations = [[10, 20, 30, 40], [40, 30, 20, 10]]
        allocation = [[0, 0.7, 1, 1], [1, 0.3, 0, 0]]

        assert is_pareto_efficient(valuations, allocation) == True
    test1()

if __name__ == '__main__':
    valuations = [[3, 1, 6], [6, 3, 1], [1, 6, 3]]
    allocation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    is_pareto_efficient(valuations, allocation)
    test()