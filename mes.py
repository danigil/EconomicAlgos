from typing import Dict, List, Set, Iterable

CANDIDATE_ALREADY_ELECTED = -1

def elect_next_budget_item(
    votes: List[Set[str]],
    balances: List[float],
    costs: Dict[str,float],
    is_print: bool = False,
    require_unique_candidate: bool = False,
):
    items = costs.keys()
    voters_per_item = {item:set() for item in items} # group the voters of each project
    for ivoter, vote in enumerate(votes):
        for item in vote:
            voters_per_item[item].add(ivoter)

    # print(voters_per_item)
    # print([sum(balances[voter] for voter in voters_per_item[item]) for item in items])
    # print([costs[item] for item in items])
            
    def is_affordable(item: str):
        return costs[item]!=CANDIDATE_ALREADY_ELECTED and sum(balances[voter] for voter in voters_per_item[item]) >= costs[item]         
    
    affordable_itmes = set(filter(is_affordable, items))
    if len(affordable_itmes) == 0:
        raise Exception('No affordable item')
    
    if require_unique_candidate:
        prices_per_voter = [costs[item]/len(voters_per_item[item]) for item in affordable_itmes]
        if prices_per_voter.count(min(prices_per_voter)) > 1:
            raise Exception('There is no unique candidate with the minimum price per voter')
    
    min_price_per_voter_item = min(affordable_itmes, key=lambda item: costs[item]/len(voters_per_item[item])) # selected item
    ivoters_by_ascending_balance = sorted(voters_per_item[min_price_per_voter_item], key= lambda ivoter: balances[ivoter]) # sort the voters by balance
    
    remaining_cost = costs[min_price_per_voter_item]
    proportional_cost = remaining_cost/len(voters_per_item[min_price_per_voter_item])
    
    # People with less balance pay what they have, people with more compensate for them.
    payments = {ivoter: 0 for ivoter in voters_per_item[min_price_per_voter_item]}
    
    # Pay the proportional cost to each voter, recalc the proportional cost and repeat until the cost is covered
    for i, ivoter in enumerate(ivoters_by_ascending_balance):
        if balances[ivoter] < proportional_cost:
            payments[ivoter] = balances[ivoter]
            remaining_cost -= balances[ivoter]
            
            proportional_cost = remaining_cost/(len(voters_per_item[min_price_per_voter_item])-i-1)
        else:
            payments[ivoter] = proportional_cost
            remaining_cost -= proportional_cost
    if is_print:
        print(f'"{min_price_per_voter_item}" is elected')
        for ivoter in sorted(payments.keys()):
            print(f'Citizen {ivoter} pays {payments[ivoter]} and has {balances[ivoter]-payments[ivoter]} left.')

    return min_price_per_voter_item, payments


def mes(votes: Iterable[Iterable[str]], k: int):
    votes = list(set(vote) for vote in votes)

    candidates_map = {candidate:i for i,candidate in enumerate(set().union(*votes))}
    reverse_candidates_map = {i:candidate for candidate,i in candidates_map.items()}

    votes = [set(candidates_map[candidate] for candidate in vote) for vote in votes]
    
    n_candidates = max(max(vote) for vote in votes) + 1
    n_voters = len(votes)
    assert set().union(*votes) == set(range(n_candidates))
    assert 1 <= k <= n_candidates
    
    voters_per_candidate = [set()] * n_candidates
    for i, vote in enumerate(votes):
        for candidate in vote:
            voters_per_candidate[candidate].add(i)
    budgets = [k/n_voters] * n_voters
    candidate_prices = {icandidate:1 for icandidate in range(n_candidates)}

    elected = set()
    # try:
    for i in range(k):
        elected_candidate, payments = elect_next_budget_item(votes, budgets, candidate_prices, require_unique_candidate=True)
        for ivoter, payment in payments.items():
            budgets[ivoter] -= payment
        candidate_prices[elected_candidate] = CANDIDATE_ALREADY_ELECTED
        elected.add(elected_candidate)
    # except Exception as e:
    #     # print(e)
    #     return None
    return elected
    
    
    
    
if __name__ == "__main__":
    votes = [{'a'}, {'b'}, {'a', 'c'}]
    balances = [0.5, 1, 1]
    costs = {'a': 1.5, 'b':1, 'c':1}
    
    # elect_next_budget_item(votes, balances, costs, is_print=True, require_unique_candidate=True)
    # print(mes(votes, 2))
    