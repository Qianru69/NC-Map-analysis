# NC-Map-analysis

This repository contains four parts:
* `/NC` folder contains North Carolina's shapefiles.
* `/figures` folder contains all graphs we generated for this project, including graphs for outlier analysis and short burst.
* `/short_burst` folder contains code for running short burst.
* Two jupyter notebooks (`NC_recom.ipynb`, `data_cleaning.ipynb`) and two python files (`gerrychainNC.py`, `map_details.py`).

## Ensemble Analysis Related Files

### `data_cleaning.ipynb`
This notebook is for cleaning up and repair North Carolina shapefiles and adding elections into the shapefiles, resulting shapefile can be found in `/NC` folder and it is ready to use for gerrychain.

### `NC_recom.ipynb`
This notebook is for running shapefile generated from [`data_cleaning.ipynb`](###`data_cleaning.ipynb`) on gerrychain and generate boxplot for the signature of gerrymandering.

### `gerrychainNC.py` 
This file runs gerrychain on NC shapefile, does ensemble analysis for cutting edges, efficiency gap, mean median difference, partisan bias ,and democratic won and generate graphs

### `map_details.py`
This file is to calculate cutting edges, efficiency gap, mean median difference, partisan bias and democratic won for 2022 and 2023 maps.

We added the calculation as orange and green line in our graphs.

## `short_burst`
`un_runs.py`, `sb_runs.py`, `gingleator.py` are modified from [shortbursts-gingles](https://github.com/vrdi/shortbursts-gingles)
* `un_runs.py` is for doing unbiased runs on minority population and it'll generate corresponding pickle files for further analysis
* `sb_runs.py` is for doing short burst runs on minority population and it'll generate corresponding numpy files and pickle files for further analysis
* `gingleator.py` is an utility class and methods defined in it are used in `un_runs.py` and `sb_runs.py`

`NC_results.ipynb`
This notebook contains graphs we generated based on the pickle files and numpy files generated by doing unbiased and short burst runs

## Note
`/figures/outlier analysis-histogram (2022 seed)` folder contains all the extra graphs we created but didn't do detailed analysis on

Raw data are too large to be pushed onto the Github, they can be downloaded through this [Google Drive Link](https://drive.google.com/drive/folders/18qkU5U9KdR3jj6UFOhHfDNnx25gbqb1Y?usp=sharing)