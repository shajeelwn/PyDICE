if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import time
    from ema_workbench.analysis import clusterer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    import os

    os.chdir(os.getcwd())
    import sys

    # insert at 1, 0 is the script path (or '' in REPL)
    pydice_folder = os.path.dirname(os.getcwd()) + "\\1_Model"
    sys.path.insert(1, pydice_folder)

    from specify import specify_levers
    from ema_workbench import (save_results)
    from ema_workbench import (perform_experiments, Model, Policy, RealParameter, 
                               IntegerParameter, ScalarOutcome, ema_logging, MultiprocessingEvaluator)

    ema_logging.log_to_stderr(ema_logging.INFO)
    from PyDICE_V4 import PyDICE

    
    model = PyDICE()
    dice_sm = Model('dicesmEMA', function = model)
    dice_opt = pd.read_excel("DICE2013R.xlsm" ,sheet_name = "Opttax", index_col = 0)

    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                             IntegerParameter('t2xco2_dist',0 , 2),
                             IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr', 0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]

    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('irstp',  0.001, 0.015),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]


    dice_sm.outcomes = [ScalarOutcome('Atmospheric Temperature 2010', ScalarOutcome.INFO),
                        ScalarOutcome('Damages 2010', ScalarOutcome.INFO),
                        ScalarOutcome('Utility 2010', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2010', ScalarOutcome.INFO),
                        ScalarOutcome('Atmospheric Temperature 2050', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2050', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2050', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2050', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2100', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2100', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2100', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2100', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2150', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2150', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2150', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2150', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2200', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2200', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2200', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2200', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2300', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2300', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2300', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Total Output 2300', ScalarOutcome.MAXIMIZE)]

    n_scenarios=30000
    nord_optimal_policy = Policy('nord_optimal_policy', **specify_levers(np.mean(dice_opt.iloc[129]),0.015,0,29))


    start = time.time()
    with MultiprocessingEvaluator(dice_sm) as evaluator:
        results = evaluator.perform_experiments(scenarios=n_scenarios, policies=nord_optimal_policy)
    end = time.time()

    print('Experiment time is '+str(round((end - start)/60))+' mintues')

    file_name = 'exploration_V4_'+str(n_scenarios)+'scen_'+'nordhaus_optimal_policy_'+str(4)+'obj'+'.tar.gz'
    save_results(results, file_name)

    experiments, outcomes = results

    noutcomes = {}
    for i in range(int(len(outcomes.keys())/5)):
        arr = np.stack((outcomes[x] for x in list(outcomes.keys())[i::4]), axis=-1)
        key = list(outcomes.keys())[i][:-5]
        noutcomes[key] = arr

    outcome_name = []
    for i in range(4):
        outcome_name.append(str(dice_sm.outcomes[i])[15:-7])
    outcome_name.sort()

    print ("time series clustering")
    for i in range(4):
        data = noutcomes[outcome_name[i]]

        # calcuate distances
        start = time.time()
        distances = clusterer.calculate_cid(data)
        #np.save('TSC_30k_scen_damages_distances', distances)
        end = time.time()
        print('Calculation time is ' + str(round((end - start)/60)) + ' mintues')

        #calculate silhouette width
        sil_score_lst = []
        start = time.time()
        for n_clusters in range(2,11):
            clusterers = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage="complete")
            cluster_labels = clusterers.fit_predict(distances)
            silhouette_avg = silhouette_score(distances, cluster_labels, metric="precomputed")
            sil_score_lst.append(silhouette_avg)
            print(outcome_name[i] + ": For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        end = time.time()
        print('Clustering time is ' + str(round((end - start)/60)) + ' mintues')

        with open(outcome_name[i] + '_cluster_silhouette_width.txt', 'w') as f:
            for s in sil_score_lst:
                f.write(str(s) + "\n")

        # do agglomerative clustering on the distances
        start = time.time()
        for j in range(2, 6):
            clusters = clusterer.apply_agglomerative_clustering(distances,
                                                                n_clusters=j)
            x = experiments.copy()
            x['clusters'] = clusters.astype('object')
            x.to_csv('TSC_30k_' + outcome_name[i] + '_cluster_' + str(j) + '.csv')
        end = time.time()
        print('Clustering time is ' + str(round((end - start) / 60)) + ' mintues')
