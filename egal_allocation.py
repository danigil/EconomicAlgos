from typing import List
import cvxpy as cp
import numpy as np

def egal(values: List[List[float]]):

    n_participants = len(values)
    assert n_participants > 0
    n_items = len(values[0])
    assert n_items > 0
    assert all(len(values[i]) == n_items for i in range(n_participants))
    
    values = np.array(values)
    
    X = cp.Variable((n_participants, n_items), nonneg=True)
    alloc_constraint = cp.sum(X, axis=0) == 1.0
    
    values_mat = cp.multiply(X, values)
    
    # cp.Problem(cp.Maximize(cp.sum(cp.min(values_mat, axis=0))), [alloc_constraint]).solve()
    cp.Problem(cp.Maximize(cp.min(cp.sum(values_mat, axis=0))), [alloc_constraint]).solve()
    
    
    print(X.value)
    
    
if __name__ == '__main__':
    # values = [[1,2], [2,1]]
    values = [[80, 19, 1], [79, 1, 20]]
    
    egal(values)
