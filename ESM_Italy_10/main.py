# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 17:01:02 2023

@author: SESAM
"""

#%% Model setup

from esm import Model
from time import time
from esm import Plots

import json

def run_model(model, path):
    model.read_input_excels(path)
    model.Base._data_assigment()
    model.Base._model_completion()
    model.Base._model_run(solver="GUROBI", verbose=True)

timings = {}
results = {}

main_folder = "case_studies"
case_studies = [
    {"name": "Italy24", "scenarios": [
        "a.1_STEPS_res",
        "a.2.0_STEPS_inv",
        "a.2.1_STEPS_inv_RESconst",
        "a.2.2_STEPS_inv_RES50",
        "a.2.3_STEPS_inv_CTgrow",
        "a.2.4_STEPS_inv_ELZ50",
        "b.0_NUC",
        "c.NZE",
        "d.0_NZE-NUC",
        ]},
]

scenario_color = {
    "a.1_STEPS_res": "red",
    "a.2.0_STEPS_inv": "pink",
    "a.2.1_STEPS_inv_RESconst": "yellow",
    "a.2.2_STEPS_inv_RES50": "green",
    "a.2.3_STEPS_inv_CTgrow": "blue",
    "a.2.4_STEPS_inv_ELZ50":'black',
    "b.0_NUC": "black",
    "c.NZE": "black",
    "d.0_NZE-NUC": "blue",
}

objective = "cost_discount"  # options: cost_discount, cost, production, CO2_emiss

#%% Model initialization
for inputs in case_studies:
    model = Model("{}/{}/Sets.xlsx".format(main_folder, inputs["name"]))

# #%% Generate clean clusters template
# for inputs in case_studies:
#    model.generate_clusters_excel("{}/{}/clean_inputs/Time_clusters.xlsx".format(main_folder, inputs["name"]))


# #%% Create clean model inputs template
# for inputs in case_studies:
#     model.read_clusters("{}/{}/Time_clusters.xlsx".format(main_folder, inputs["name"]))
#     model.create_input_excels("{}/{}/clean_inputs".format(main_folder, inputs["name"]))

#%% Solve model

for inputs in case_studies:
    print("\nModel generation for {}...".format(inputs["name"]), end="")
    
    # Model Generation
    model = Model("{}/{}/Sets.xlsx".format(main_folder, inputs["name"]))
    model.read_clusters("{}/{}/Time_clusters.xlsx".format(main_folder, inputs["name"]))

    # Generate the equations
    # model.Base._model_generation(obj_fun=objective)

    # Initialize time record
    timings[inputs["name"]] = {}

    results[inputs["name"]] = Plots()


    # solving model for scenarios
    for scenario in inputs["scenarios"]:
        model.Base._model_generation(obj_fun=objective)

        print("\nSolving for scenario {}...\n".format(scenario))
        start = time()
        path = "{}/{}/{}/{}/{}".format(main_folder, inputs["name"], "scenarios", scenario, "inputs")
        run_model(model, path)
        end = time()
        timings[inputs["name"]][scenario] = end - start

        # if model is solved, results upload and export to file
        if hasattr(model.Base, "results"):
            print("\nModel solved")
            results[inputs["name"]].upload_from_model(model=model, scenario=scenario, color=scenario_color[scenario])
            delattr(model.Base, "results")
        else:
            print("\nModel NOT solved")

    print("\nResults export...", end="")
    results[inputs["name"]].save_results(scenario=inputs["scenarios"],_format="csv",path="{}/{}/{}/{}".format(main_folder,inputs["name"],"scenarios","results"),
    )
    print(" DONE")

with open("timing.json","w") as fp:
    json.dump(timings,fp)


# %%
