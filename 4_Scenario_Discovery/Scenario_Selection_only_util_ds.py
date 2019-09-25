import os
os.chdir(os.getcwd())

import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd

from ema_workbench.util.utilities import load_results

def normalize_out_dic(outcomes):
    norm_outcomes = {}
    for ooi in outcomes.keys():
        data = outcomes[ooi]
        mx = max(data)
        mx = max(data)
        mn = min(data)
        if mx == mn:
            norm_outcomes[ooi] = data - mn
        else:
            norm_outcomes[ooi] = (data - mn)/(mx-mn)
    return norm_outcomes
def calculate_distance(data, oois, scenarios=None, distance='euclidean'):
    '''data is the outcomes of exploration results,
    scenarios is a list of scenario indices (decision variables),
    oois is a list of variable names,
    distance is to choose the distance metric. options:
            bray-curtis, canberra, chebyshev, cityblock (manhattan), correlation,
            cosine, euclidian, mahalanobis, minkowski, seuclidian,
            sqeuclidian, wminkowski
    returns a list of distance values
    '''
    #make a matrix of the data n_scenarios x oois
    scenario_data = np.zeros((len(scenarios), len(oois)))
    for i, s in enumerate(scenarios):
        for j, ooi in enumerate(oois):
            scenario_data[i][j] = data[ooi][s]

    distances = pdist(scenario_data, distance)
    return distances

"""
######## IMPORT SET OF SCENARIOS #######
"""
fn = 'scenario_selection_30k_scen_100k_nfe_only_util.tar.gz'
print(fn)
try:
    results = load_results(fn)
except:
    print('skipping results')
    exit(0)
exp, outcomes = results
norm_new_out = normalize_out_dic(outcomes)
oois = list(outcomes.keys())

def evaluate_diversity_single(x, data=norm_new_out, oois=oois, weight=0.5, distance='euclidean'):
    '''
    takes the outcomes and selected scenario set (decision variables),
    returns a single 'diversity' value for the scenario set.
    outcomes : outcomes dictionary of the scenario ensemble
    decision vars : indices of the scenario set
    weight : weight given to the mean in the diversity metric. If 0, only minimum; if 1, only mean
    '''
    distances = calculate_distance(data, oois, list(x), distance)
    minimum = np.min(distances)
    mean = np.mean(distances)
    diversity = (1-weight)*minimum + weight*mean

    return [diversity]

def find_maxdiverse_scenarios(combinations):
    diversity = 0.0
    solutions = []
    ct = int(len(combinations)/10)
    for idx, sc_set in enumerate(combinations):
        if idx % ct == 0: print('Completing index ' + str(idx), int(idx/ct))

        temp_div = evaluate_diversity_single(list(sc_set))
        if temp_div[0] > diversity:
            diversity = temp_div[0]
            solutions = []
            solutions.append(sc_set)
        elif temp_div[0] == diversity:
            solutions.append(sc_set)
        # print(solutions)
    return diversity, solutions

totalResults = []
def startSelection(combos, count):
    reses = []
    no_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=no_workers)

    worker_data = np.array_split(combos, no_workers)

    result = pool.imap(find_maxdiverse_scenarios, worker_data)

    #find the max of these
    max_diversity = 0.0
    solutions = []
    toreturn = []
    for r in result:
        print(r[0])
        if r[0] >= max_diversity:
            max_diversity = r[0]
            solutions = []
            solutions.append(r[1])
            toreturn = []
            toreturn.append(r)
        elif r[0] == max_diversity:
            solutions.append(r[1])
            toreturn.append(r)

    # Uncomment to write the results of each chunk of combinations
    # with open(str(count) + '_output_scenarioselection.txt', 'a') as file:
    #      print('==================Saving for {} {}====================='.format(max_diversity, solutions))
    #      file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
    #      print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
    
    pool.close()
    pool.join()

    return toreturn

if __name__ == "__main__":
    n_scen = len(outcomes[oois[0]])
    indices = range(n_scen)

    combos = []
    count = 0
    set_size = int(4)
    solution_options = []
    start = time.time()
    for item in itertools.combinations(indices, set_size):

        if len(combos) < 10000000:
            combos.append(item)
            continue
        sols = startSelection(list(combos), count)
        solution_options.extend(sols)
        count += 1
        print(len(combos)*count)
        combos = []
    end = time.time()
    print('Plot time is ' + str(round((end - start) / 60)) + ' mintues')
    print("out of loop")
    if len(combos) > 0:
        print('Working on count', count+1)
        startSelection(combos, count+1)

    print("Solution options: "+str(len(solution_options)))

    max_diversity = 0.0
    solutions = []
    for r in solution_options:
        if r[0] >= max_diversity:
            max_diversity = r[0]
            solutions = []
            solutions.append(r[1])
        elif r[0] == max_diversity:
            solutions.append(r[1])

    with open('output_scenarioselection.txt', 'a') as file:
        print('==================Saving for {} {}====================='.format(max_diversity, solutions))
        file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
        print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
        