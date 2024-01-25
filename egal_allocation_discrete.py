from typing import List, Literal
import numpy as np
from copy import deepcopy

import time
from itertools import product, combinations
import matplotlib.pyplot as plt
import random

def assert_values_valid(values: List[List[float]]):
    n_participants = len(values)
    assert n_participants > 0
    n_items = len(values[0])
    assert n_items > 0
    assert all(len(values[i]) == n_items for i in range(n_participants))

    return n_participants, n_items

def calc_rule(items, rule='egalitarian'):
    if rule == 'egalitarian':
        return min(items)
    elif rule == 'max_product':
        return np.prod(items)

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
            # items_each_gets = tuple(set() for _ in range(n_participants))
            # for i in range(n_participants-1):
            #     n_items_i_gets = random.randint(0, len(remaining_items))
            #     items_i_gets = random.sample(remaining_items, n_items_i_gets)

            #     remaining_items -= set(items_i_gets)
            #     items_each_gets[i].update(items_i_gets)
                
            # items_each_gets[-1].update(remaining_items)

            # final_items = tuple(curr_items[i].union(items_each_gets[i]) for i in range(n_participants))
            final_items = tuple(curr_items[i].union(remaining_items) if i==0 else curr_items[i] for i in range(n_participants)) # give everything to 0
            assert set().union(*final_items) == set(range(n_items)), f'{set().union(*final_items)}'
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
            
        def check_state(state):
            pes = pessimistic_bound(state)
            opt = optimistic_bound(state)
            
            # print(f'state: {state}, pes: {pes}, opt: {opt}')
            # if pes >= opt:
            #     print(f'pruned_b state: {state}, pes: {pes}, opt: {opt}')
            return state[1]==n_items or pes < opt
            

        new_state_space = list(filter(check_state, state_space))

        return new_state_space

    n_participants, n_items = assert_values_valid(values)

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

def brute_force_discrete_allocation(values: List[List[float]], rule: Literal['egalitarian', 'max_product'] = 'egalitarian', is_print=True):
    n_participants, n_items = assert_values_valid(values)


    def generate_options(n_participants, items={1,3,5}):

        if n_participants == 1:
            # print(f'items: {items}')
            yield (items,)
        else:
            for k in range(0, len(items)+1):
                for items0 in combinations(items, k):
                    # print(f'items0: {items0}')
                    remaining_items = items - set(items0)
                    for option in generate_options(n_participants-1, remaining_items):
                        yield (set(items0), *option)

    rets = []

    for option in generate_options(n_participants, set(range(n_items))):
        final_value = tuple(sum(values[i][j] for j in option[i]) for i in range(n_participants))
        ret = calc_rule(final_value, rule=rule)
        rets.append((ret, option))

    return max(rets, key=lambda x: x[0])



def plot_runtime(n_items=10):
    times = tuple([] for _ in range(4))
    x = range(1, n_items+1)
    
    for i in x:
        values = [[1]*i for _ in range(2)]
        for j, (is_prune_a, is_prune_b) in enumerate(product([False, True], repeat=2)):

            start = time.time()
            ret = state_space_search_discrete_allocation(values, is_prune_a=is_prune_a, is_prune_b=is_prune_b, is_print=False)
            end = time.time()
            times[j].append(end-start)

            if j==0:
                optimal_result = ret
            else:
                assert all(np.allclose(optimal_result[0][i], ret[0][i]) for i in range(2)), f'Results should be the same: ret:{ret[0]} != opt:{optimal_result[0]}'

            

    
    for i, (is_prune_a, is_prune_b) in enumerate(product([False, True], repeat=2)):
        plt.plot(x, times[i], label=f'Prune A: {is_prune_a}, Prune B:{is_prune_b}')

    plt.xlabel('Number of items')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(x, x)
    plt.grid()
    plt.legend()
    plt.show()

def test():
    def test_case(values, rule='egalitarian'):
        ret00 = state_space_search_discrete_allocation(values, rule=rule, is_print=False)
        opt = brute_force_discrete_allocation(values, rule=rule)[0]
        assert calc_rule(ret00[0], rule=rule) == opt, f'{min(ret00[0])} != {opt}'

        ret01 = state_space_search_discrete_allocation(values, rule=rule, is_print=False, is_prune_a=False, is_prune_b=True)
        assert calc_rule(ret01[0], rule=rule) == opt

        ret10 = state_space_search_discrete_allocation(values, rule=rule, is_print=False, is_prune_a=True, is_prune_b=False)
        assert calc_rule(ret10[0], rule=rule) == opt

        ret11 = state_space_search_discrete_allocation(values, rule=rule, is_print=False, is_prune_a=True, is_prune_b=True)
        assert calc_rule(ret11[0], rule=rule) == opt

    def test1():
        # example from lecture
        values = [[11, 11, 55], [22, 22, 33], [33, 44, 0]]
        test_case(values)
        test_case(values, rule='max_product')

    def test2(n=5, m=5):

        for i in range(2, n):
            for j in range(1, m):
                values = [[1]*j]*i
                try:
                    test_case(values)
                except:
                    print(f'values: {values}')
                    print(f'i: {i}, j: {j}')

                
                test_case(values, rule='max_product')

    test1()
    test2()

if __name__ == "__main__":
    values = [[1, 2, 3], [4, 5, 6]]
    state_space_search_discrete_allocation(values, rule='egalitarian', is_print=True)

    values = [[1]*10 for _ in range(2)]
    state_space_search_discrete_allocation(values, rule='egalitarian', is_print=True)

    test()

    plot_runtime(n_items=10)