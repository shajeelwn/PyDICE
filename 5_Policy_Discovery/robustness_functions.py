import numpy as np
import pandas as pd
from ema_workbench import ScalarOutcome

def s_to_n(experiments, outcomes, scalar_outcomes):   
    overall_scores = {}
    for policy in np.unique(experiments['policy']):
        scores = {}

        logical = experiments['policy']==policy

        for outcome in scalar_outcomes:
            value  = outcomes[outcome.name][logical]
            mean = np.mean(value)
            std = np.std(value) 
            if outcome.kind == ScalarOutcome.MAXIMIZE:
                sn_ratio = mean/std
            else:
                sn_ratio = mean*std
            scores[outcome.name] = sn_ratio
        overall_scores[policy] = scores
    scores = pd.DataFrame.from_dict(overall_scores).T
    return scores

def max_regret(experiments, outcomes, scalar_outcomes):
    overall_regret = {}
    max_regret = {}
    policy_column = experiments ['policy']
    for outcome in scalar_outcomes:
        # create a DataFrame with all the relevent information
        # i.e., policy, scenario_id, and scores
        data = pd.DataFrame({outcome.name: outcomes[outcome.name], 
                             "policy":experiments['policy'],
                             "scenario":experiments['scenario']})

        # reorient the data by indexing with policy and scenario id
        data = data.pivot(index='scenario', columns='policy')

        # flatten the resulting hierarchical index resulting from 
        # pivoting, (might be a nicer solution possible)
        data.columns = data.columns.get_level_values(1)

        # we need to control the broadcasting. 
        # max returns a 1d vector across scenario id. By passing
        # np.newaxis we ensure that the shape is the same as the data
        # next we take the absolute value
        #
        # basically we take the difference of the maximum across 
        # the row and the actual values in the row
        
        # if outcome.kind == ScalarOutcome.MINIMIZE:
        if  outcome.name[0:3] == 'Atm' or outcome.name[0:3] == 'Dam':
            outcome_regret = (data.min(axis=1)[:, np.newaxis] - data).abs()
        else:
            outcome_regret = (data.max(axis=1)[:, np.newaxis] - data).abs()
        
        
        overall_regret[outcome.name] = outcome_regret
        max_regret[outcome.name] = outcome_regret.max()
        
    max_regret = pd.DataFrame(max_regret)
        
    return max_regret, overall_regret