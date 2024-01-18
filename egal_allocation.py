from typing import List, Union
import cvxpy as cp
import numpy as np
import logging

def egal(values: List[List[float]], print_results: bool = False):

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
    cp.Problem(cp.Maximize(cp.min(cp.sum(values_mat, axis=0))), [alloc_constraint]).solve(solver=cp.CLARABEL)
    
    allocation = X.value
    if print_results:
        for i_participant in range(n_participants):
            print(f'Participant {i_participant} gets:')
            for i_item in range(n_items):
                print(f'\t {allocation[i_participant, i_item]*100: .3f}% of item {i_item}')

    return np.array(allocation)

def check_valid_allocation(X):
    X = np.array(X)
    
    all_nonneg = np.all(X >= 0)
    all_sum_to_one = np.allclose(np.sum(X, axis=0), 1.0)

    return all_nonneg and all_sum_to_one

def check_is_X_more_egalitarian_than_Y(X, Y, values):
    logging.info(f'check_is_X_more_egalitarian_than_Y')

    X = np.array(X)
    Y = np.array(Y)
    values = np.array(values)

    logging.info(f'X: {X}')
    logging.info(f'Y: {Y}')
    logging.info(f'values: {values}')
    
    X_values_mat = np.multiply(X, values)
    Y_values_mat = np.multiply(Y, values)

    logging.info(f'X_values_mat: {X_values_mat}')
    logging.info(f'Y_values_mat: {Y_values_mat}')

    X_egalitarian_utility = np.min(np.sum(X_values_mat, axis=0))
    Y_egalitarian_utility = np.min(np.sum(Y_values_mat, axis=0))

    logging.info(f'X_egalitarian_utility: {X_egalitarian_utility}')
    logging.info(f'Y_egalitarian_utility: {Y_egalitarian_utility}')

    return X_egalitarian_utility >= Y_egalitarian_utility

def test():
    def test1():
        # Example from lecture

        values = [[80, 19, 1], [79, 1, 20]]
        X = egal(values)
        assert check_valid_allocation(X), 'Allocation X is not valid'
        assert check_is_X_more_egalitarian_than_Y(X, X, values), 'Allocation X is not more egalitarian than itself'

        Y = [[0.5, 1, 0], [0.5, 0, 1]]
        assert check_valid_allocation(Y), 'Allocation Y is not valid'

        assert check_is_X_more_egalitarian_than_Y(X, Y, values), 'Allocation X is not more egalitarian than Y'

    def test2():
        # Dictatorship

        values = [[1, 1], [1, 1]]
        X = egal(values)
        assert check_valid_allocation(X), 'Allocation X is not valid'

        Y = [[1, 0], [1, 0]]
        assert check_is_X_more_egalitarian_than_Y(X, Y, values), 'Allocation X is not more egalitarian than Y'

    test1()
    test2()
      
    
if __name__ == '__main__':
    values = [[80, 19, 1], [79, 1, 20]]
    
    X = egal(values, print_results=True)
    # print(f'Returned Allocation X is valid?: {check_valid_allocation(X)}')
    # print(X)

    test()

