from typing import Dict, List, Set


def elect_next_budget_item(
    votes: List[Set[str]],
    balances: List[float],
    costs: Dict[str,float]
):
    items = costs.keys()
    voters_per_item = {item:set() for item in items} # group the voters of each project
    for ivoter, vote in enumerate(votes):
        for item in vote:
            voters_per_item[item].add(ivoter)
            
    def is_affordable(item: str):
        return sum(balances[voter] for voter in voters_per_item[item]) >= costs[item]         
    
    affordable_itmes = set(filter(is_affordable, items))
    min_price_per_voter_item = min(affordable_itmes, key=lambda item: costs[item]/len(voters_per_item[item]), ) # selected item
    ivoters_by_ascending_balance = sorted(voters_per_item[min_price_per_voter_item], key= lambda ivoter: balances[ivoter]) # sort the voters by balance
    
    remaining_cost = costs[min_price_per_voter_item]
    proportional_cost = remaining_cost/len(voters_per_item[min_price_per_voter_item])
    
    # People with less balance pay what they have, people with more compensate for them.
    payments = {ivoter: 0 for ivoter in voters_per_item[min_price_per_voter_item]}
    
    # Pay the proportional cost to each voter, recalc the proportional cost and repeat until the cost is covered
    for ivoter in ivoters_by_ascending_balance:
        if balances[ivoter] < proportional_cost:
            payments[ivoter] = balances[ivoter]
            remaining_cost -= balances[ivoter]
            
            proportional_cost = remaining_cost/len(voters_per_item[min_price_per_voter_item])
        else:
            payments[ivoter] = proportional_cost
    
    print(f'"{min_price_per_voter_item}" is elected')
    for ivoter in sorted(payments.keys()):
        print(f'Citizen {ivoter} pays {payments[ivoter]} and has {balances[ivoter]-payments[ivoter]} left.')

            
    
    
    
    
if __name__ == "__main__":
    votes = [{'a'}, {'b'}, {'a', 'c'}]
    balances = [0.5, 1, 1]
    costs = {'a': 1.5, 'b':1, 'c':1}
    
    elect_next_budget_item(votes, balances, costs)
    