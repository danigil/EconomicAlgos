from itertools import combinations, product
import sys
import traceback
from mes import mes
counter = 0
error_counter = {}
found_counter = 0
def check_monotonicty_all_ks(votes, c):
    global counter
    def check_monotonicty(committees_k1, committees_k2):
        return committees_k1 <= committees_k2
    rets = []
    for k1 in range(1, c):
        counter += 1

        k2 = k1 + 1
        committees_k1 = None
        committees_k2 = None
        try:
            committees_k1 =mes(votes, k1)
            committees_k2 =mes(votes, k2)
            
        except Exception as e:
            if str(e) == '':
                # print(e)
                ex_type, ex_value, ex_traceback = sys.exc_info()
                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(ex_traceback)

                # Format stacktrace
                stack_trace = list()
                print(votes)
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

                print("Stack trace : %s" %stack_trace)
            if str(e) in error_counter:
                error_counter[str(e)] += 1
            else:
                error_counter[str(e)] = 1
            # print(ex_value)

        if committees_k1 is None or committees_k2 is None:
            continue
        flag = check_monotonicty(committees_k1, committees_k2)
        if not flag:
            rets.append((k1, k2, committees_k1, committees_k2))
    if len(rets) > 0:
        return rets
    else:
        return None

def find_non_monotone_example(n, c, vote_options):
    global found_counter
    found = False
    for votes in filter(lambda votes: len(set().union(*list(set(vote) for vote in votes))) >= c,product(vote_options, repeat=n)):
        n_votes_per_candidate = [0] * c
        for vote in votes:
            for can in vote:
                n_votes_per_candidate[can] += 1
                
        # if(len(set(n_votes_per_candidate)) != c):
        #     # print(n_votes_per_candidate)
        #     continue
        # else:
        #     pass
            # print(f'checking unique votes: {votes}')

        
        rets = check_monotonicty_all_ks(votes, c)
        if rets is not None:
            for ret in rets: 
                k1, k2, committees_k1, committees_k2 = ret
                # print(f'{"@"*20}FOUND NON-MONOTONE EXAMPLE{"@"*20}')
                # print(f'n: {n}, c: {c}')
                # print(f'votes: {votes}')
                # print(f'k1: {k1}, k2: {k2}')
                # print()
                # print(f'committees_k1: {committees_k1}')
                # print(f'committees_k2: {committees_k2}')
                found = True
                found_counter += 1
            # break
        
    if not found:
        print(f'No non-monotone example found for n = {n}, c = {c}')

# find_non_monotone_example(3, 4)
if __name__ == '__main__':
    # print('dababy')
    for c in range(1, 5):
        print(f'c: {c}')
        vote_options = set().union(*[set(combinations(range(c), i)) for i in range(1,c+1)])
        # print(f'c: {c}')
        for n in range(1, 5):
            print(f'\tn: {n}')
            find_non_monotone_example(n, c, vote_options)
        
        print('~'*10)
        print('')

    print(f'total runs: {counter}')
    print(f'error_counter: {error_counter}')
    print(f'valid runs: {counter - sum(error_counter.values())}')
    print(f'found_counter: {found_counter}')