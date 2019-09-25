import pandas as pd

import time
import os 
os.chdir(os.getcwd())
import sys
# insert at 1, 0 is the script path (or '' in REPL)
pydice_folder = os.path.dirname(os.getcwd())+"\\1_Model"
sys.path.insert(1, pydice_folder)

from ema_workbench import (Model, Constraint, Scenario, RealParameter, IntegerParameter, ScalarOutcome, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging

from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework.optimization import EpsilonProgress

ema_logging.log_to_stderr(ema_logging.INFO)
BaseEvaluator.reporting_frequency = 0.1
# ema_logging.log_to_stderr(ema_logging.DEBUG)

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

dice_sm.outcomes = [ScalarOutcome('Atmospheric Temperature 2050', ScalarOutcome.MINIMIZE),
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

sel_scenDF = pd.read_csv("selected_scenarios_only_util_tsc_V4.csv",index_col = 0)

scen_list = []

for i in range (4):
    sel_scen_dict = sel_scenDF.iloc[i].to_dict()
    for _ in ["cback", "fdamage", "t2xco2_dist", "t2xco2_index"]:
        sel_scen_dict[_] = int(sel_scen_dict[_])
    scen_list.append(Scenario("scenario " + str(sel_scenDF.iloc[i].name), **sel_scen_dict))

eps = [0.001, 0.1, 0.1, 0.1]*(int(len(dice_sm.outcomes)/4.0))
nfe = 500000
convergence_metrics = [EpsilonProgress()]

constraints = [Constraint("Utility 2300", outcome_names="Utility 2300",
                          function=lambda x:max(0, -x)),
               Constraint("Atmospheric Temperature 2050", outcome_names="Atmospheric Temperature 2050",
                          function=lambda x:max(0, x-4)),
               Constraint("Atmospheric Temperature 2100", outcome_names="Atmospheric Temperature 2100",
                          function=lambda x:max(0, x-4)),
               Constraint("Atmospheric Temperature 2150", outcome_names="Atmospheric Temperature 2150",
                          function=lambda x:max(0, x-4)),
               Constraint("Atmospheric Temperature 2200", outcome_names="Atmospheric Temperature 2200",
                          function=lambda x:max(0, x-4)),
               Constraint("Atmospheric Temperature 2300", outcome_names="Atmospheric Temperature 2300",
                          function=lambda x:max(0, x-4))]

if __name__ == "__main__":
    for i in range(len(scen_list)):    
        scenarios = scen_list[i]
        start = time.time()
        print("starting search for wcs", flush=True)
        with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                                      searchover='levers',
                                                      reference=scenarios,
                                                      epsilons=eps,
                                                      convergence=convergence_metrics,
                                                      constraints=constraints)
        #results.to_csv("wcs.csv")
        #convergence.to_csv("wcs_con.csv")
        end = time.time()
        print('MORDM time is ' + str(round((end - start)/60)) + ' mintues', flush=True)
        results.to_csv("wcs_"+ str(sel_scenDF.iloc[i].name) +".csv")
        convergence.to_csv("wcs_con_"+ str(sel_scenDF.iloc[i].name) +".csv")
        