from typing import List, Union, Tuple
import networkx as nx
import cvxpy as cp
import math

def find_rent_alloc(values: List[List[float]], rent_price: float, positive_prices=False, is_print=False) -> Union[Tuple[List[int], List[float]], None]:
    """
    find rent allocation with non-negative/positive prices if it exists

    If exists:
        returns a tuple of two lists:
            - the first list contains the room indices allocated to each participant
            - the second list contains the prices of the rooms
    If no allocation exists:
        returns None
    """

    def extract_room_idx(s: str) -> int:
        return int(s.split('_')[-1])

    n_participants = len(values)
    assert n_participants > 0
    n_rooms = len(values[0])
    assert n_rooms == n_participants
    
    B = nx.Graph()
    B.add_nodes_from([f"person_{i}" for i in range(n_participants)], bipartite=0)
    B.add_nodes_from([f"room_{i}" for i in range(n_rooms)], bipartite=1) 
    B.add_weighted_edges_from([(f"person_{i}", f"room_{j}", values[i][j]) for i in range(n_participants) for j in range(n_rooms)])
    
    M = nx.max_weight_matching(B, maxcardinality=True)
    M_0 = {k:v for k,v in M}
    M_1 = {v:k for k,v in M}
    M = {**M_0, **M_1}

    z = cp.Variable(1)
    prices = cp.Variable(n_rooms)

    prices_constr = [prices[i] >= z for i in range(n_rooms)]
    
    prices_sum = cp.sum(prices)
    prices_sum_constr = cp.sum(prices) == rent_price
    
    val_const = [values[i][extract_room_idx(M[f"person_{i}"])] - prices[i] >= values[i][extract_room_idx(M[f"person_{j}"])] - prices[j] for i in range(n_participants) for j in range(n_participants)]

    prob = cp.Problem(cp.Maximize(z), [prices_sum_constr] + val_const + prices_constr)
    prob.solve(solver=cp.CLARABEL)

    z_req = z.value > 0 if positive_prices else z.value >= 0

    if prob.status == "infeasible" or not z_req:
        return None
    else:
        room_alloc =  [extract_room_idx(M[f"person_{i}"]) for i in range(n_participants)]
        room_prices = prices.value
        assert math.isclose(sum(room_prices), rent_price)

        if is_print:
            print(f"rent price: {rent_price}")
            print(f"room values: {values}")
            for i in range(n_participants):
                print(f"\tperson_{i} gets room_{room_alloc[i]} at price {room_prices[room_alloc[i]]}")
        return room_alloc, room_prices

def test():
    def test1_nonneg():
        values = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        rent_price = 2
        assert find_rent_alloc(values, rent_price) is not None

        values = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
        rent_price = 90
        assert find_rent_alloc(values, rent_price) is not None

        values = [[150, 0], [140, 10]]
        rent_price = 100
        assert find_rent_alloc(values, rent_price) is None

        values = [[36, 34, 30, 0], [31, 36, 33, 0], [34, 30, 36, 0], [32, 33, 35, 0]]
        rent_price = 100
        assert find_rent_alloc(values, rent_price) is None
    
    def test2_pos():
        values = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        rent_price = 2
        assert find_rent_alloc(values, rent_price, positive_prices=True) is not None

        values = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
        rent_price = 90
        assert find_rent_alloc(values, rent_price, positive_prices=True) is not None

        values = [[150, 0], [140, 10]]
        rent_price = 100
        assert find_rent_alloc(values, rent_price, positive_prices=True) is None

        values = [[36, 34, 30, 0], [31, 36, 33, 0], [34, 30, 36, 0], [32, 33, 35, 0]]
        rent_price = 100
        assert find_rent_alloc(values, rent_price, positive_prices=True) is None

    test1_nonneg()
    test2_pos()

if __name__ == '__main__':
    test()
    
    values = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
    rent_price = 90

    ret = find_rent_alloc(values, rent_price, is_print=True)

    