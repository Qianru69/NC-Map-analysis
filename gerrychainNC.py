#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:26:42 2024

@author: eveomett

Author: Ellen Veomett

for AI for Redistricting

Lab 3, spring 2024
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election, \
    GeographicPartition
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain.random import random
from functools import partial
import time
import pandas as pd

start_time = time.time()
random.seed(12345678)
outdir = "./NC_recom_CD/"

# Need to adjust the total steps
total_steps_in_run = 5000
save_district_graph_mod = 1
save_district_plot_mod = 100

os.makedirs(outdir, exist_ok=True)
nc_graph = Graph.from_file("./NC/NC.geojson")

# Set up the initial partition as "CD22"
my_updaters = {'cut_edges': cut_edges, 'population': Tally('TOTPOP', alias='population'),
               'voting age population': Tally('VAP', alias='voting age population'),
               'latino population': Tally('HISP', alias='latino population'),
               'black population': Tally('NH_BLACK', alias='black population'),
               'white population': Tally('NH_WHITE', alias='white population'),
               'asian population': Tally('NH_ASIAN', alias='asian population'),
               'NHPI population': Tally('NH_NHPI', alias='NHPI population'),
               # Native Havaiian and Pacific Islander, non-hispantic
               'AMIN population': Tally('NH_AMIN', alias='AMIN population'),
               # American Indian and Alaska Native, non-hispantic
               'other population': Tally('NH_ASIAN', alias='other population')
               # Other race, non-hispanic
               }

"""
Election Abbreviations:
ATG - Attorney General
AUD - Auditor
GOV - Governor
LTG - Lieutenant Governor
PRE - President
SOS - Secretary of State
TRE - Treasurer
USS - U.S. Senate
"""

elections = [
    Election("G20PRE", {"Democratic": "G20PRED", "Republican": "G20PRER"}),
    Election("G20USS", {"Democratic": "G20USSD", "Republican": "G20USSR"}),
    Election("G20ATG", {"Democratic": "G20ATGD", "Republican": "G20ATGR"}),
    Election("G20AUD", {"Democratic": "G20AUGD", "Republican": "G20AUDR"}),
    Election("G20GOV", {"Democratic": "G20GOVD", "Republican": "G20GOVR"}),
    Election("G20LTG", {"Democratic": "G20LTGD", "Republican": "G20LTGR"}),
    Election("G20SOS", {"Democratic": "G20SOSD", "Republican": "G20SOSR"}),
    Election("G20TRE", {"Democratic": "G20TRED", "Republican": "G20TRER"}),
]

election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition_22 = GeographicPartition(nc_graph, assignment="CD22", updaters=my_updaters)
initial_partition_23 = GeographicPartition(nc_graph, assignment="CD23", updaters=my_updaters)

num_dist = len(initial_partition_22)

# Set up ideal population of each district
ideal_population = sum(initial_partition_22['population'].values()) / num_dist

proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                   )

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2 * len(initial_partition_22["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition_22, 0.05)

chain_22 = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=always_accept,
    initial_state=initial_partition_22,
    total_steps=total_steps_in_run
)

# Initiate lists to store the counts
efficiency_gap_ensemble = []
mean_median_diff_ensemble = []
partisan_bias_ensemble = []
cutedge_ensemble = []
population_ensemble = {
    "latino population": [],
    "black population": [],
    "white population": [],
    "asian population": [],
    "NHPI population": [],  # Native Hawaiian or Pacific Islander
    "AMIN population": [],  # American Indian
    "other population": []
}
democratic_won_ensemble = []
election_name = "G20PRE"

# Run through chain, building plots
for t, part in enumerate(chain_22):
    if t % 50 == 0:
        print("... step", t, "...")

    efficiency_gap_ensemble.append(part[election_name].efficiency_gap())
    mean_median_diff_ensemble.append(part[election_name].mean_median())
    partisan_bias_ensemble.append(part[election_name].partisan_bias())
    cut_edges_count = len(part['cut_edges'])
    cutedge_ensemble.append(cut_edges_count)

    # Calculate number of latino-majority districts
    num_majorities = {
        "latino population": 0,
        "black population": 0,
        "white population": 0,
        "asian population": 0,
        "NHPI population": 0,  # Native Hawaiian or Pacific Islander
        "AMIN population": 0,  # American Indian
        "other population": 0
    }

    for i in range(1, num_dist + 1):
        total_population = part['population'][i]

        for pop_type in num_majorities.keys():
            if part[pop_type][i]/total_population >= 0.5:
                num_majorities[pop_type] += 1

    for pop_type in num_majorities.keys():
        population_ensemble[pop_type].append(num_majorities[pop_type])

    # Analyze partition results
    num_dem_wins = 0
    num_dem_wins = part[election_name].wins("Democratic")
    democratic_won_ensemble.append(num_dem_wins)

print(efficiency_gap_ensemble)
print(mean_median_diff_ensemble)
print(partisan_bias_ensemble)
print(cutedge_ensemble)
print(population_ensemble)
print(democratic_won_ensemble)

# 1. The number of cut edges in the plan
plt.figure()
plt.title('NC recom cut edges plot')
plt.xlabel('Cutting Edges')
plt.ylabel('Counts')
plt.hist(cutedge_ensemble, align='left')
cutedge_output_file = outdir + "cut_edges_plot(" + election_name + ")" + str(total_steps_in_run) + ".png"
plt.savefig(cutedge_output_file)
plt.close()

plt.figure()
plt.title('NC recom efficiency gap plot')
plt.xlabel('Efficiency Gap')
plt.ylabel('Counts')
plt.hist(efficiency_gap_ensemble, align='left')
efficiency_gap_output_file = outdir + "efficiency_gap_plot(" + election_name + ")" + str(total_steps_in_run) + ".png"
plt.savefig(efficiency_gap_output_file)
plt.close()

plt.figure()
plt.title('NC recom mean median difference plot')
plt.xlabel('Mean-Median Difference')
plt.ylabel('Counts')
plt.hist(mean_median_diff_ensemble, align='left')
Mean_Median_Difference_output_file = outdir + "mean_median_difference_plot(" + election_name + ")" + str(total_steps_in_run) + ".png"
plt.savefig(Mean_Median_Difference_output_file)
plt.close()

plt.figure()
plt.title('NC recom partisan bias difference plot')
plt.xlabel('Partisan Bias Difference')
plt.ylabel('Counts')
plt.hist(partisan_bias_ensemble, align='left')
partisan_bias_output_file = outdir + "partisan_bias_difference_plot(" + election_name + ")" + str(total_steps_in_run) + ".png"
plt.savefig(partisan_bias_output_file)
plt.close()

# 2. The number of certain race majority districts in the plan
for pop_type in population_ensemble.keys():
    plt.figure()
    plt.title('NC recom ' + pop_type + ' majority plot')
    # bins = range(3)
    # plt.hist(latinmaj_ensemble, bins=bins, align = 'left')
    plt.xlabel('Number of ' + pop_type + ' Majority Districts')  # Add x-axis label
    plt.ylabel('Counts')  # Add y-axis label
    ensemble = population_ensemble[pop_type]
    plt.hist(ensemble, bins=np.arange(min(ensemble) - 0.5, max(ensemble) + 1.5, 1), align='mid')
    plt.xticks(np.arange(min(ensemble), max(ensemble) + 1, 1))
    output_file = outdir + pop_type + "_majority_plot_" + election_name + "_" + str(total_steps_in_run) + ".png"
    plt.savefig(output_file)
    plt.close()

# 3. The number of Democratic-won districts in the plan
# (you’ll need to add some “updaters” in order to do this.
# You may want to see https://gerrychain. readthedocs.io/en/latest/api.html?
# highlight=updaters#module-gerrychain. updaters).
plt.figure()
plt.title('NC recom democratic win plot')
plt.xlabel('Number of Democratic Wins')
plt.ylabel('Counts')
plt.hist(democratic_won_ensemble,
         bins=np.arange(min(democratic_won_ensemble) - 0.5, max(democratic_won_ensemble) + 1.5, 1), align='mid')
plt.xticks(np.arange(min(democratic_won_ensemble), max(democratic_won_ensemble) + 1, 1))
demwin_output_file = outdir + "democratic_win_plot(" + election_name + ")" + str(total_steps_in_run) + ".png"
plt.savefig(demwin_output_file)
plt.close()

end_time = time.time()
print("The time of execution of above program is :",
      (end_time - start_time) / 60, "mins")
