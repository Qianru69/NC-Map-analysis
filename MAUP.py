#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:32:51 2023

@author: eveomett

Lab 3: MAUP and data.  See details on Canvas page

Make sure to say where/when you got your data!
"""

# Reference: example code provided by Professor Veomett

import pandas as pd
import geopandas as gpd
import maup
import time

maup.progress.enabled = True

# Import Population Data
# Source: https://redistrictingdatahub.org/dataset/north-carolina-block-pl-94171-2020-by-table/
# Downloaded on March 22, 2024

start_time = time.time()
# This census file has population, Hispanic and non-Hispanic details.
population_df = gpd.read_file("./nc_pl2020_b/nc_pl2020_p2_b.shp")
end_time = time.time()
print("The time to import nc_pl2020_p2_b.shp is:", (end_time-start_time)/60, "mins")

start_time = time.time()
# This census file has voting age population (VAP), Hispanic and non-Hispanic details.
vap_df = gpd.read_file("./nc_pl2020_b/nc_pl2020_p4_b.shp")
end_time = time.time()
print("The time to import nc_pl2020_p4_b.shp is:", (end_time-start_time)/60, "mins")

# The data set below has 2020 presidential election results by precinct
# Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7760H
# Downloaded on March 24, 2024
start_time = time.time()
election_df = gpd.read_file("./nc_vest_2020/nc_2020.shp")
end_time = time.time()
print("The time to import nc_2020.shp is:", (end_time-start_time)/60, "mins")

# The data set below is a shapefile of the congressional districts
# Source: https://redistrictingdatahub.org/dataset/2022-north-carolina-congressional-districts-approved-plan/
# Downloaded on March 22, 2024
start_time = time.time()
cong_df = gpd.read_file("./nc_cong_adopted_2022/NC_SMmap2_Statewide.shp")
end_time = time.time()
print("The time to import NC_SMmap2_Statewide.shp is:", (end_time-start_time)/60, "mins")

# Column names of each imported file
print("Column names of each imported file:\npopulation_df:")
print(population_df.columns)
print("vap_df:")
print(vap_df.columns)
print("election_df")
print(election_df.columns)
print("cong_df:")
print(cong_df.columns)

# Checking if the data has any holes or overlaps
print(maup.doctor(election_df))
election_df = maup.smart_repair(election_df)
print("Repair finished")
print(maup.doctor(election_df))

# checking the columns of the cong_df dataset
columns = ['OBJECTID', 'District_A', 'Shape_Leng', 'Shape_Area']
print(cong_df[columns])
print("Number of different values from columns OBJECTID and District_A:", sum(cong_df["OBJECTID"] != cong_df["District_A"]))
print(cong_df.dtypes)

# I use "OBJECTED" as the district column name
district_col_name = "OBJECTID"

# Adjust the crs in election_df to make sure it is the same as population_df
print(population_df.crs)
print(election_df.crs)
election_df = election_df.to_crs(population_df.crs)
print(population_df.crs)
print(election_df.crs)

# Now put data into same geometry units
blocks_to_precincts_assignment = maup.assign(population_df.geometry, election_df.geometry)
vap_blocks_to_precincts_assignment = maup.assign(vap_df.geometry, election_df.geometry)

# Select population columns that we are interested in
pop_column_names = ['P0020001', 'P0020002', 'P0020005', 'P0020006', 'P0020007',
                    'P0020008', 'P0020009', 'P0020010', 'P0020011']

vap_column_names = ['P0040001', 'P0040002', 'P0040005', 'P0040006', 'P0040007',
                    'P0040008', 'P0040009', 'P0040010', 'P0040011']

# Now we will put all the population columns into the election dataframe
for name in pop_column_names:
    election_df[name] = population_df[name].groupby(blocks_to_precincts_assignment).sum()
for name in vap_column_names:
    election_df[name] = vap_df[name].groupby(vap_blocks_to_precincts_assignment).sum()

print(population_df['P0020001'].sum())
print(election_df['P0020001'].sum())
print(vap_df['P0040001'].sum())
print(election_df['P0040001'].sum())

# Check if the combined dataset has any gaps or overlaps
print(maup.doctor(election_df))

# Assign Congressional Districts
precincts_to_districts_assignment = maup.assign(election_df.geometry, cong_df.geometry)
election_df["CD"] = precincts_to_districts_assignment

print(set(election_df["CD"]))
for precinct_index in range(len(election_df)):
    election_df.at[precinct_index, "CD"] = cong_df.at[election_df.at[precinct_index, "CD"], district_col_name]
print(set(cong_df[district_col_name]))
print(set(election_df["CD"]))

# Rename columns
rename_dict = {'P0020001': 'TOTPOP', 'P0020002': 'HISP', 'P0020005': 'NH_WHITE', 'P0020006': 'NH_BLACK', 'P0020007': 'NH_AMIN',
                    'P0020008': 'NH_ASIAN', 'P0020009': 'NH_NHPI', 'P0020010': 'NH_OTHER', 'P0020011': 'NH_2MORE',
                    'P0040001': 'VAP', 'P0040002': 'HVAP', 'P0040005': 'WVAP', 'P0040006': 'BVAP', 'P0040007': 'AMINVAP',
                                        'P0040008': 'ASIANVAP', 'P0040009': 'NHPIVAP', 'P0040010': 'OTHERVAP', 'P0040011': '2MOREVAP',
                                        'G20PREDBID': 'G20PRED', 'G20PRERTRU': 'G20PRER', 'G20USSRTIL': 'G20USSR',
                                        'G20USSDCUN': 'G20USSD', 'G20GOVRFOR': 'G20GOVR', 'G20GOVDCOO': 'G20GOVD',
               'G20LTGRROB': 'G20LTGR', 'G20LTGDHOL': 'G20LTGD', 'G20ATGRONE': 'G20ATGR', 'G20ATGDSTE': 'G20ATGD',
               'G20TRERFOL': 'G20TRER', 'G20TREDCHA': 'G20TRED', 'G20SOSRSYK': 'G20SOSR', 'G20SOSDMAR': 'G20SOSD',
               'G20AUDRSTR': 'G20AUDR', 'G20AUDDWOO': 'G20AUDD'}

'''
*ATG - Attorney General
*AUD - Auditor
*LTG - Lieutenant Governor
*PRE - President
*SOS - Secretary of State
*TRE - Treasurer
*USS - U.S. Senate
'''

# Rename columns and drop the columns not needed
election_df.rename(columns=rename_dict, inplace = True)
election_df.drop(columns=['G20PRELJOR', 'G20PREGHAW', 'G20PRECBLA', 'G20PREOWRI', 'G20USSLBRA', 'G20USSCHAY',
                               'G20GOVLDIF', 'G20GOVCPIS', 'G20AGRRTRO', 'G20AGRDWAD',
       'G20INSRCAU', 'G20INSDGOO', 'G20LABRDOB', 'G20LABDHOL', 'G20SPIRTRU',
       'G20SPIDMAN', 'G20SSCRNEW', 'G20SSCDBEA', 'G20SSCRBER', 'G20SSCDINM',
       'G20SSCRBAR', 'G20SSCDDAV', 'G20SACRWOO', 'G20SACDSHI', 'G20SACRGOR',
       'G20SACDCUB', 'G20SACRDIL', 'G20SACDSTY', 'G20SACRCAR', 'G20SACDYOU',
       'G20SACRGRI', 'G20SACDBRO'], inplace=True)

print(election_df.columns)

print(election_df.loc[election_df["CD"] == 1, "TOTPOP"].sum())
print(election_df.loc[election_df["CD"] == 2, "TOTPOP"].sum())
pop_vals = [election_df.loc[election_df["CD"] == n, "TOTPOP"].sum() for n in range(1, 15)]
print(pop_vals)

# The created dataframe contains NA values
print(election_df[election_df.isnull().any(axis=1)])

# Compare the population sum to see if the NA values should be filled by 0
print(population_df[['P0020001', 'P0020002', 'P0020005', 'P0020006', 'P0020007',
                    'P0020008', 'P0020009', 'P0020010', 'P0020011']].sum(),
        vap_df[['P0040001', 'P0040002', 'P0040005', 'P0040006', 'P0040007',
                    'P0040008', 'P0040009', 'P0040010', 'P0040011']].sum())
print(election_df[['TOTPOP', 'HISP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 'NH_ASIAN', 'NH_NHPI', 'NH_OTHER', 'NH_2MORE',
                    'VAP', 'HVAP', 'WVAP', 'BVAP', 'AMINVAP', 'ASIANVAP', 'NHPIVAP', 'OTHERVAP', '2MOREVAP']].sum())

# Population sum values can match, before and after merging the data columns
# Fill 0 at the NA values
election_df = election_df.fillna(0)
print(election_df[election_df.isnull().any(axis=1)])

election_df.to_file("./NC/NC.shp")

shp_file = gpd.read_file('./NC/NC.shp')

shp_file.to_file('./NC/NC.geojson', driver='GeoJSON')


