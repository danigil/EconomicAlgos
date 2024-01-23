from typing import List, Literal
import numpy as np
from copy import deepcopy

import time
from itertools import product
import matplotlib.pyplot as plt
import random

def state_space_search_discrete_allocation(values: List[List[float]],
                                            is_prune_a=False,
                                            is_prune_b=False,
                                            rule: Literal['egalitarian', 'max_product'] = 'egalitarian',
                                            is_print=True
                                            ):
    #state space search

    def allocate_item(state, item):
        def add_ith_value(tup, i):
            return tup[:i]+(tup[i]+values[i][item], *tup[i+1:])

        def add_ith_item(tup, i):
            curr_set = tup[i]
            curr_set.add(item)
            return tup[:i]+(curr_set, *tup[i+1:])

        values_so_far, n_items_allocated, allocated_items_so_far = state
        # new_allocations = [print(f'values_so_far: {values_so_far}, n_items_allocated: {n_items_allocated}, allocated_items_so_far: {allocated_items_so_far}')]
        # exit()
        new_allocations = [(add_ith_value(values_so_far, i), n_items_allocated + 1, add_ith_item(deepcopy(allocated_items_so_far), i)) for i in range(n_participants)]
        # for i_participant in range(n_participants):
            
        #     new_allocations[i_participant][0][i_participant] += values[i_participant][item]
        #     new_allocations[i_participant][2][i_participant].add(item)
        
        return new_allocations

    def prune_a(state_space):
        unique_states = set([state[0] for state in state_space])
        new_unique_states = []
        for state in state_space:
            if state[0] in unique_states:
                unique_states.remove(state[0])
                new_unique_states.append(state)
        return new_unique_states

    def prune_b(state_space):
        def pessimistic_bound(state):
            curr_values, curr_n_items_allocated, curr_items = state
            # n_remaining_items = n_items - curr_n_items_allocated
            remaining_items = set(range(n_items)) - set(range(curr_n_items_allocated))
            items_each_gets = tuple(set() for _ in range(n_participants))
            for i in range(n_participants-1):
                n_items_i_gets = random.randint(0, len(remaining_items))
                items_i_gets = random.sample(remaining_items, n_items_i_gets)

                remaining_items -= set(items_i_gets)
                items_each_gets[i].update(items_i_gets)
                
            items_each_gets[-1].update(remaining_items)

            final_items = tuple(curr_items[i].union(items_each_gets[i]) for i in range(n_participants))
            sums = tuple(sum(values[i][j] for j in final_items[i]) for i in range(n_participants))
            if rule == 'egalitarian':
                return min(sums)
            elif rule == 'max_product':
                return np.prod(sums)
        
        def optimistic_bound(state):
            curr_values, curr_n_items_allocated, curr_items = state
            remaining_items = set(range(n_items)) - set(range(curr_n_items_allocated))

            final_items = tuple(curr_items[i].union(remaining_items) for i in range(n_participants))
            # print(f'opt final_items: {final_items}')
            sums = tuple(sum(values[i][j] for j in final_items[i]) for i in range(n_participants))
            if rule == 'egalitarian':
                return min(sums)
            elif rule == 'max_product':
                return np.prod(sums)

        new_state_space = list(filter(lambda state: state[1]==n_items or pessimistic_bound(state) < optimistic_bound(state), state_space))

        return new_state_space

    n_participants = len(values)
    assert n_participants > 0
    n_items = len(values[0])
    assert n_items > 0
    assert all(len(values[i]) == n_items for i in range(n_participants))

    empty_allocation = tuple([0]*n_participants), 0, tuple(set() for _ in range(n_participants))
    state_space = [empty_allocation]
    curr_round = 0
    while curr_round < n_items:
        if state_space[0][1] != curr_round:
            curr_round+=1
            if curr_round == n_items:
                break
            
            if is_prune_a:
                state_space=prune_a(state_space)
            if is_prune_b:
                state_space=prune_b(state_space)

        state = state_space.pop(0)
        state_space.extend(allocate_item(state, state[1]))

    if rule == 'egalitarian':
        rule_fn = lambda state: min(state[0])
    elif rule == 'max_product':
        rule_fn = lambda state: np.prod(state[0])

    best_allocation = max(state_space, key=rule_fn)
    values, _, items = best_allocation
    if is_print:
        for i in range(n_participants):
            print(f'Participant {i} gets items {items[i]} with value {values[i]}')

    return values, items

def plot_runtime(n_items=10):
    times = tuple([] for _ in range(4))
    x = range(1, n_items+1)
    for i in x:
        values = [[1]*i for _ in range(2)]
        for j, (is_prune_a, is_prune_b) in enumerate(product([False, True], repeat=2)):
            start = time.time()
            ret = state_space_search_discrete_allocation(values, is_prune_a=is_prune_a, is_prune_b=False, is_print=False)
            if j==0:
                optimal_result = ret
            else:
                assert all(np.allclose(optimal_result[0][i], ret[0][i]) for i in range(2)), f'Results should be the same: ret:{ret[0]} != opt:{optimal_result[0]}'

            end = time.time()
            times[j].append(end-start)

    
    for i, (is_prune_a, is_prune_b) in enumerate(product([False, True], repeat=2)):
        plt.plot(x, times[i], label=f'Prune A: {is_prune_a}, Prune B:{is_prune_b}')

    plt.xlabel('Number of items')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(x, x)
    plt.grid()
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    values = [[1, 2, 3], [4, 5, 6]]
    state_space_search_discrete_allocation(values, rule='egalitarian', is_prune_b=True)
    print()

    values = [[4,5,6,7,8], [8,7,6,5,4]]
    state_space_search_discrete_allocation(values)

    plot_runtime(n_items=15)