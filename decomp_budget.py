from typing import List, Set, Literal, Union

import numpy as np
import cvxpy as cp


def find_decompostion(
    budget: List[float],
    preferences: List[Set[int]],
    is_print=False,
) -> Union[None, np.ndarray]:
    """
    
    >>> budget = [400, 50, 50, 0]
    >>> preferences = [{0,1}, {0,2}, {0,3}, {1,2}, {0}]
    >>> find_decompostion(budget, preferences)
    array([[100.,   0.,   0.,   0.],
           [100.,   0.,   0.,   0.],
           [100.,   0.,   0.,   0.],
           [  0.,  50.,  50.,   0.],
           [100.,   0.,   0.,   0.]])

    >>> budget = [1, 99]
    >>> preferences = [{0}]*99 + [{1}]
    >>> find_decompostion(budget, preferences) is None
    True

    >>> budget = [99,1]
    >>> np.array_equal(find_decompostion(budget, preferences), np.array([1,0]*99 + [0,1]).reshape((100,2)))
    True

    """
    n_subjects = len(budget)
    n_civillians = len(preferences)
    
    C = sum(budget)
    proportional_price = C / n_civillians

    budget = np.array(budget, dtype=float)
    
    D = cp.Variable((n_civillians, n_subjects), nonneg=True)
    subj_constraints = cp.sum(D, axis=0) == budget
    civ_constraints = cp.sum(D, axis=1) == proportional_price
    
    supp_constraints = [D[i,j]==0 for i in range(n_civillians) for j in set(range(n_subjects)) - preferences[i]]
    
    prob = cp.Problem(cp.Maximize(cp.sum(D)), [subj_constraints, civ_constraints] + supp_constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ["infeasible", "unbounded"]:
        ret = np.array(D.value)
        ret = np.round(ret, 3)
    else:
        ret = None    
    if is_print:
        print(ret)

    return ret

    
    
if __name__ == "__main__":
    budget = [400, 50, 50, 0]
    preferences = [{0,1}, {0,2}, {0,3}, {1,2}, {0}]
    find_decompostion(budget, preferences, is_print=True)

    budget = [99, 1]
    preferences = [{0}]*99 + [{1}]
    find_decompostion(budget, preferences, is_print=True)

    budget = [1, 99]
    find_decompostion(budget, preferences)

            
    