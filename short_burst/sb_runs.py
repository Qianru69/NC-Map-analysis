import argparse
import geopandas as gpd
import numpy as np
import pickle
from functools import partial
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
from gerrychain.tree import recursive_tree_part
from gingleator import Gingleator
from little_helpers import *

## Read in 
parser = argparse.ArgumentParser(description="SB Chain run", 
                                 prog="sb_runs.py")
parser.add_argument("states", metavar="state_id", type=str,
                    choices=["NC"],
                    help="which states to run chains on")
parser.add_argument("iters", metavar="chain_length", type=int,
                    help="how long to run each chain")
parser.add_argument("l", metavar="burst_length", type=int,
                    help="The length of each short burst")
parser.add_argument("col", metavar="column", type=str,
                    help="Which column to optimize")
parser.add_argument("score", metavar="score_function", type=int,
                    help="How to count gingles districts",
                    choices=[0,1,2,3,4])
args = parser.parse_args()

num_h_districts = {"NC": 14}


score_functs = {0: None, 1: Gingleator.reward_partial_dist, 
                2: Gingleator.reward_next_highest_close,
                3: Gingleator.penalize_maximum_over,
                4: Gingleator.penalize_avg_over}

BURST_LEN = args.l
NUM_DISTRICTS = num_h_districts[args.states]
ITERS = args.iters
POP_COL = "TOTPOP"
N_SAMPS = 10
SCORE_FUNCT = None #score_functs[args.score]
EPS = 0.045
MIN_POP_COL = args.col
THRESH = 0.5


## Setup graph, updaters, elections, and initial partition

print("Reading in Data/Graph", flush=True)

#graph = Graph.from_json("./state_experiments/state_block_group_graphs/BG_{}.json".format(args.states))
nc_graph = Graph.from_file("./NC/NC.geojson")

my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "BVAP": Tally("BVAP"),
               "HVAP": Tally("HVAP"),
               "WVAP": Tally("WVAP"),
               "nWVAP": lambda p: {k: v - p["WVAP"][k] for k,v in p["VAP"].items()},
               "cut_edges": cut_edges}


print("Creating seed plan", flush=True)

total_pop = sum([nc_graph.nodes()[n][POP_COL] for n in nc_graph.nodes()])

init_partition = Partition(nc_graph, assignment="CD22", updaters=my_updaters)

gingles = Gingleator(init_partition, pop_col=POP_COL,
                     threshold=THRESH, score_funct=SCORE_FUNCT, epsilon=EPS,
                     minority_perc_col="{}_perc".format(MIN_POP_COL))

gingles.init_minority_perc_col(MIN_POP_COL, "VAP", 
                               "{}_perc".format(MIN_POP_COL))

num_bursts = int(ITERS/BURST_LEN)

print("Starting Short Bursts Runs", flush=True)

for n in range(N_SAMPS):
    sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    print("\tFinished chain {}".format(n), flush=True)

    print("\tSaving results", flush=True)

    f_out = "./short_burst/data/states/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_th{:.1%}_score{}_{}.npy".format(args.states,
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, THRESH, args.score, n)
    np.save(f_out, sb_obs[1])

    f_out_part = "./short_burst/data/states/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_th{:.1%}_score{}_{}_max_part.p".format(args.states,
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, THRESH, args.score, n)

    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "BVAP": sb_obs[0][0]["BVAP"],
                 "WVAP": sb_obs[0][0]["WVAP"],
                 "HVAP": sb_obs[0][0]["HVAP"],}

    with open(f_out_part, "wb") as f_out:
        pickle.dump(max_stats, f_out)
