import networkx as nx
import itertools
from typing import List, Union
import numpy as np
from copy import deepcopy
import math

# def get_value(valuations: list[list[float]], agent, item):
#     return valuations[agent][item]

# def return_pareto_enhancement(valuations: List[List[float]], allocation: List[List[float]]) -> Union[bool, List[List[float]]]:
#     n_participants = len(valuations)
#     n_items = len(valuations[0])

#     def is_pareto_efficient(valuations: List[List[float]], allocation: List[List[float]]) -> bool:
#         def construct_switching_graph(allocation: List[List[float]]) -> nx.DiGraph:
#             G = nx.DiGraph()
#             for i, j in itertools.permutations(range(n_participants), 2):
#                 i_items = set([k for k in range(n_items) if allocation[i][k] > 0])
#                 ratios = [valuations[i][item]/valuations[j][item] for item in i_items]
#                 min_ratio = min(ratios)

#                 G.add_edge(i, j, weight=min_ratio)

#             return G

#         G = construct_switching_graph(allocation)
#         # for edge in G.edges:
#         #     print(edge, G[edge[0]][edge[1]]['weight'])
#         cycles = list(nx.simple_cycles(G))
#         for cycle in cycles:
#             # print(f'cycle: {cycle}')
#             # print([(i,j) for i,j in zip(cycle, cycle[1:]+[cycle[0]])])
#             weights = [G[i][j]['weight'] for i,j in zip(cycle, cycle[1:]+[cycle[0]])]
#             if np.prod(weights) < 1:
#                 return False
#         return True

def is_allocation_valid(allocation: List[List[float]]):
    n_participants = len(allocation)
    assert n_participants > 0
    n_items = len(allocation[0])
    assert n_items > 0
    assert all(len(allocation[i]) == n_items for i in range(n_participants))
    assert all(all(1 >= allocation[i][j] >= 0 for j in range(n_items)) for i in range(n_participants)), f'invalid allocation: {allocation}'
    assert all(sum(allocation[i][j] for i in range(n_participants)) == 1 for j in range(n_items))

    return True

def is_alloc_a_pareto_enhancement(valuations: List[List[float]], alloc_a: List[List[float]], alloc_b: List[List[float]]) -> bool:
    n_participants = len(valuations)
    n_items = len(valuations[0])

    values_a = [sum(valuations[i][j]*alloc_a[i][j] for j in range(n_items)) for i in range(n_participants)]
    values_b = [sum(valuations[i][j]*alloc_b[i][j] for j in range(n_items)) for i in range(n_participants)]

    all_ge = all(values_a[i] >= values_b[i] for i in range(n_participants))
    any_gt = any(values_a[i] > values_b[i] for i in range(n_participants))

    assert all_ge, f'values_a: {values_a}, values_b: {values_b}'
    assert any_gt, f'values_a: {values_a}, values_b: {values_b}'

def is_pareto_efficient(valuations: List[List[float]], allocation: List[List[float]], fraction=0.00001) -> bool:
    n_participants = len(valuations)
    n_items = len(valuations[0])

    def construct_switching_graph(allocation: List[List[float]]) -> nx.DiGraph:
        G = nx.DiGraph()
        items = {}
        for i, j in itertools.permutations(range(n_participants), 2):
            i_items = set([k for k in range(n_items) if allocation[i][k] > 0])
            ratios = [(valuations[i][item]/valuations[j][item], item) for item in i_items]
            min_ratio = min(ratios, key=lambda x: x[0])

            # G.add_edge(i, j, weight=round(np.log2(min_ratio[0]), 6))
            # G.add_edge(i, j, weight=np.log2(min_ratio[0]))
            G.add_edge(i, j, weight=min_ratio[0])
            items[(i, j)] = min_ratio[1]

        return G, items

    G, items = construct_switching_graph(allocation)
    # for edge in G.edges:
        # print(edge, G[edge[0]][edge[1]]['weight'])
    found=False
    cycles = list(nx.simple_cycles(G))
    for cycle in cycles:
        # print(f'cycle: {cycle}')
        # print([(i,j) for i,j in zip(cycle, cycle[1:]+[cycle[0]])])
        weights = [G[i][j]['weight'] for i,j in zip(cycle, cycle[1:]+[cycle[0]])]
        if np.prod(weights) < 1:
            # print(f'cycle: {cycle}')
            # print(f'weights: {weights}')
            # print([(i,j) for i,j in zip(cycle, cycle[1:]+[cycle[0]])])
            found=True
            break
    
    if found:
        cycle += [cycle[0]]
    # if nx.algorithms.shortest_paths.weighted.negative_edge_cycle(G):
    #     cycle = nx.algorithms.shortest_paths.weighted.find_negative_cycle(G, 0)
        # print(f'cycle: {cycle}')
        curr_fraction = fraction
        prev_val=None

        changes = []
        for i,j in zip(cycle, cycle[1:]):
            item = items[(i, j)]
            if prev_val is not None:
                curr_fraction = prev_val/valuations[i][item]
            changes.append((i, item, -curr_fraction))
            changes.append((j, item, curr_fraction))    
            prev_val = curr_fraction*valuations[j][item]

        new_allocation = deepcopy(allocation)
        for i, item, change in changes:
            new_allocation[i][item] += change

        is_allocation_valid(new_allocation)
        is_alloc_a_pareto_enhancement(valuations, new_allocation, allocation)

        return new_allocation
            # print(f'item: {item}')
            # print(f'edge: {i, j}, weight: {G[i][j]["weight"]}')
    else:
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

        valuations = [[80, 19, 1], [79, 1, 20]]
        allocation = [[5/8, 0, 0], [3/8, 1, 1]]

        better_allocation = is_pareto_efficient(valuations, allocation)
        # print(better_allocation)

        # assert is_pareto_efficient(valuations, allocation) != True

    def test2():
        valuations = [[3, 1, 6], [6, 3, 1], [1, 6, 3]]
        allocation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        assert is_pareto_efficient(valuations, allocation) != True

    test1()
    test2()

if __name__ == '__main__':
    valuations = [[3, 1, 6], [6, 3, 1], [1, 6, 3]]
    allocation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    ret = is_pareto_efficient(valuations, allocation)
    # print(ret)

    allocation = [[1, 0.2, 0], [0, 0.8, 0], [0, 0, 1]]

    ret = is_pareto_efficient(valuations, allocation)
    # print(ret)

    test()