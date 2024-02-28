from typing import List, Set

import numpy as np
import cvxpy as cp


def find_decompostion(
    budget: List[float],
    preferences: List[Set[int]],
):
    n_subjects = len(budget)
    n_civillians = len(preferences)
    # D = np.zeros((n_civillians, n_subjects), dtype=float)
    
    # for pref in preferences:
    #     for i, p in enumerate(pref):
    #         D[i, p] = 1
    
    C = sum(budget)
    proportional_price = C / n_civillians
    
    D = cp.Variable((n_civillians, n_subjects), nonneg=True)
    subj_constraints = cp.sum(D, axis=0) == budget
    civ_constraints = cp.sum(D, axis=1) == proportional_price
    
    supp_constraints = [D[i,j]==0 for i in range(n_civillians) for j in set(range(n_subjects) - preferences[i])]
    
    cp.Problem(cp.Maximize(D), [subj_constraints, civ_constraints] + supp_constraints).solve(solver=cp.CLARABEL)
    
    
    
if __name__ == "__main__":
    budget = [400, 50, 50, 0]
    preferences = [{0,1}, {0,2}, {0,3}, {1,2}, {0}]
    
    find_decompostion(budget, preferences)
            
    