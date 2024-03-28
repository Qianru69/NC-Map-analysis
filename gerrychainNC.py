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
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election, GeographicPartition
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain.random import random
from functools import partial
import time
import pandas as pd

start_time = time.time()
random.seed(12345678)
outdir="./NC_recom_CD/"

# Need to adjust the total steps
total_steps_in_run = 500
save_district_graph_mod = 1
save_district_plot_mod = 100

os.makedirs(outdir, exist_ok=True)
nc_graph = Graph.from_file("./NC/NC.geojson")

# Set up the initial partition as "CD"
my_updaters = {'cut_edges': cut_edges, 'population': Tally('TOTPOP', alias='population'),
               'latino population': Tally('HISP', alias='latino population')}

elections = [
    Election("G20PRE", {"Democratic": "G20PRED", "Republican": "G20PRER"}),
    Election("G20USS", {"Democratic": "G20USSD", "Republican": "G20USSR"}),
]

election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(nc_graph, assignment="CD", updaters=my_updaters)

num_dist = len(initial_partition)

# Set up ideal population of each district
ideal_population = sum(initial_partition['population'].values()) / num_dist

proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=total_steps_in_run
    )

# Initiate lists to store the counts
cutedge_ensemble = []
latinmaj_ensemble = []
democratic_won_ensemble = []
# Run through chain, building
for t, part in enumerate(chain):
    if t % 10 == 0:
        print("... step", t, "...")
    cut_edges_count = len(part['cut_edges'])
    cutedge_ensemble.append(cut_edges_count)
    # Calculate number of latino-majority districts
    num_maj_latin = 0
    for i in range(1, num_dist + 1):
        l_perc = part["latino population"][i] / part['population'][i]
        if l_perc >= 0.5:
              num_maj_latin = num_maj_latin + 1
    latinmaj_ensemble.append(num_maj_latin)

    # Analyze partition results
    num_dem_wins = 0
    election_name = "G20PRE"
    #num_dem_wins = part[election_name].wins("Democratic")
    D_votes = part[election_name].votes("Democratic")
    R_votes = part[election_name].votes("Republican")
    for i in range(len(D_votes)):
        if D_votes[i] > R_votes[i]:
            num_dem_wins += 1
    democratic_won_ensemble.append(num_dem_wins)


print(cutedge_ensemble)
print(latinmaj_ensemble)
print(democratic_won_ensemble)

# 1. The number of cut edges in the plan
plt.figure()
plt.title('NC recom cut edges plot')
plt.xlabel('Cutting Edges')
plt.ylabel('Counts')
plt.hist(cutedge_ensemble, align='left')
cutedge_output_file = outdir + "cut_edges_plot_" + str(total_steps_in_run) + ".png"
plt.savefig(cutedge_output_file)
plt.close()

# 2. The number of majority-Latino districts in the plan
# (remember that this is in the HISP information for this file).
plt.figure()
plt.title('NC recom latino majority plot')
#bins = range(3)
#plt.hist(latinmaj_ensemble, bins=bins, align = 'left')
plt.xlabel('Number of Latino Majority Districts')  # Add x-axis label
plt.ylabel('Counts')  # Add y-axis label
plt.hist(latinmaj_ensemble, bins=np.arange(min(latinmaj_ensemble)-0.5, max(latinmaj_ensemble)+1.5, 1), align='mid')
plt.xticks(np.arange(min(latinmaj_ensemble), max(latinmaj_ensemble)+1, 1))
latinmaj_output_file = outdir + "latino_majority_plot_" + str(total_steps_in_run) + ".png"
plt.savefig(latinmaj_output_file)
plt.close()


# 3. The number of Democratic-won districts in the plan
# (you’ll need to add some “updaters” in order to do this.
# You may want to see https://gerrychain. readthedocs.io/en/latest/api.html?
# highlight=updaters#module-gerrychain. updaters).
plt.figure()
plt.title('NC recom democratic win plot')
plt.xlabel('Number of Democratic Wins')
plt.ylabel('Counts')
plt.hist(democratic_won_ensemble, bins=np.arange(min(democratic_won_ensemble)-0.5, max(democratic_won_ensemble)+1.5, 1), align='mid')
plt.xticks(np.arange(min(democratic_won_ensemble), max(democratic_won_ensemble)+1, 1))
demwin_output_file = outdir + "democratic_win_plot_" + str(total_steps_in_run) + ".png"
plt.savefig(demwin_output_file)
plt.close()

end_time = time.time()
print("The time of execution of above program is :",
      (end_time-start_time)/60, "mins")