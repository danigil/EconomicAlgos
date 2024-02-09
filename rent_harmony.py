from typing import List
import networkx as nx
import cvxpy as cp


def find_rent_with_nonnegative_prices(values: List[List[float]], rent_price: float):
    n_participants = len(values)
    assert n_participants > 0
    n_rooms = len(values[0])
    assert n_rooms == n_participants
    
    B = nx.Graph()
    B.add_nodes_from([f"person_{i}" for i in range(n_participants)], bipartite=0)
    B.add_nodes_from([f"room_{i}" for i in range(n_rooms)], bipartite=1) 
    B.add_weighted_edges_from([(f"person_{i}", f"room_{j}", values[i][j]) for i in range(n_participants) for j in range(n_rooms)])
    
    M = nx.max_weight_matching(B)
    M_0 = {k:v for k,v in M}
    M_1 = {v:k for k,v in M}
    M = {**M_0, **M_1}
    
    prices = cp.Variable(n_rooms, nonneg=True)
    
    prices_sum = cp.sum(prices)
    prices_sum_constr = cp.sum(prices) == rent_price
    
    val_const = [values[i][M[f"person_{i}"]] - prices[i] >= values[i][M[f"person_{j}"]] - prices[j] for i in range(n_participants) for j in range(n_participants)]
    
    
    
    
    