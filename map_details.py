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

total_steps_in_run = 5000
save_district_graph_mod = 1
save_district_plot_mod = 100

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
initial_partitions = {"2022 map": initial_partition_22, "2023 map": initial_partition_23}

election_name = "G20PRE"

for map in initial_partitions:
    part = initial_partitions[map]
    print("Efficiency Gap:", part[election_name].efficiency_gap())
    print("Mean Median Difference:", part[election_name].mean_median())
    print("Partisan Bias:", part[election_name].mean_median())
    print("Cut Edges:", len(part['cut_edges']))
    print("Democratic winning districts:", part[election_name].wins("Democratic"))
    num_majorities = {
        "latino population": 0,
        "black population": 0,
        "white population": 0,
        "asian population": 0,
        "NHPI population": 0,  # Native Hawaiian or Pacific Islander
        "AMIN population": 0,  # American Indian
        "other population": 0,
    }
    for i in range(1, len(part) + 1):
        total_population = part['population'][i]

        for pop_type in num_majorities.keys():
            if part[pop_type][i] / total_population >= 0.5:
                num_majorities[pop_type] += 1

    for pop_type in num_majorities.keys():
        print("pop_type:", num_majorities[pop_type])
