import pandas as pd
import scipy.stats
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
import geopandas as gpd
import json
from shapely.geometry import Point

def family_system(df):
    res = pd.DataFrame(index = ["extended", "equal", "equal2", "system"])
    for id in df.index:
        cur = df.loc[id]
        structure = -1
        extended = -1
        equal  = -1
        equal2 = -1
        polygyny = -1
        if True:
            if cur["SCCS210"] < 6:
                extended = 0
            elif  cur["SCCS210"] > 5:
                extended = 1
            if cur["SCCS281"] == 1:
                equal = 1
            elif cur["SCCS281"] in [2, 3, 4]:
                equal = 0
            if extended >= 0 and equal >= 0:
                structure = 2 * extended + 1 * equal
            if cur["SCCS280"] == 1:
                equal2 = 1
            elif cur["SCCS280"] in [2, 3, 4]:
                equal2 = 0
            res[id] = [extended, equal, equal2, structure]
    return res

def correlation_analysis(data_pivot, col, condition, path):
    if "extended" == col:
        col1 = "nuclear"
        col2 = "extended"
    else:
        col1 = "unequal"
        col2 = "equal"

    var_sample.index = var_sample["id"]
    df_structure = data_pivot[condition]
    id_ls = var_sample["id"].tolist()
    id_ls.append(col)
    df_structure = df_structure[df_structure.columns & id_ls]

    df_structure.replace(88, np.nan, inplace = True)
    df_structure.replace(99, np.nan, inplace = True)
    df_structure["SCCS1721"].replace(21, 15, inplace = True)
    df_structure["SCCS1723"].replace(21, 15, inplace = True)
    df_structure["SCCS1721"].replace(22, 25, inplace = True)
    df_structure["SCCS1723"].replace(22, 25, inplace = True)
    df_structure["SCCS1720"].replace(3, 2, inplace = True)
    df_structure["SCCS1720"].replace(4, 2, inplace = True)

    res = pd.DataFrame(index = ["corr.", "p"])
    for col_ in df_structure.columns:
        df2 = df_structure[[col, col_]].dropna()
        x = df2[col].values
        y = df2[col_].values
        a, b = spearmanr(np.ravel(x), np.ravel(y))
        if b > 0:
            res[col_] = [a, b]

    df_res0 = pd.DataFrame()

    df_res0[["corr.", "p"]] = abs(res.T[["corr.", "p"]])
    df_res0["title"] = var_sample.loc[df_res0.index].title
    df_res0 = df_res0.sort_values("corr.", ascending =  False)
    df_res0["null"] = df_structure.isnull().sum()
    df_res0["null"] = round((len(df_structure.index) - df_res0["null"]) / len(df_structure.index), 2)
    df_agg = np.round(df_structure.groupby(col).mean()[df_res0.index].T, 2)
    df_agg[["p", "corr.", "title", "null"]] = df_res0.loc[df_agg.index][["p", "corr.", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 0, 1, "p",  "corr.", "null"])
    df_agg.columns = ["title", col1, col2, "p", "corr.", "null"]

    df_res = df_res0[df_res0["null"] > 0.2]
    id_ls = df_res.index.tolist()
    df_agg = np.round(df_structure.groupby(col).mean()[id_ls].T, 2)
    df_agg[["p", "corr.", "title", "ratio"]] = df_res0.loc[df_agg.index][["p", "corr.", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 0, 1, "p", "corr.", "ratio"])
    df_agg.columns = ["title",col1, col2, "p", "corr.", "ratio"]
    df_agg.to_csv(f"{path}.csv")

def family_system_worldmap(data_pivot):
    current_palette = sns.color_palette("colorblind", 4)
    cur_pal = current_palette.as_hex()

    df_structure = data_pivot[(data_pivot["system"] > -1)]
    geo_df = gpd.GeoDataFrame(index = ["type", "name", "marker", "marker-color", "marker-size", "geometry"])

    for key in df_structure.index:
        if df_structure.at[key, "SCCS246"] > 4:
            geo_df[len(geo_df.columns)] = ["Feature", key, "o", cur_pal[df_structure.at[key, "system"]], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
        else:
            geo_df[len(geo_df.columns)] = ["Feature", key, "v", cur_pal[df_structure.at[key, "system"]], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]

    geo_df[len(geo_df.columns)] = ["Feature", "England", "*", cur_pal[0], "small", Point([-1.230469, 52.268157])]
    geo_df[len(geo_df.columns)] = ["Feature", "Netherlands", "*", cur_pal[0], "small", Point([5.361328, 52.079506])]
    geo_df[len(geo_df.columns)] = ["Feature", "Spain", "*", cur_pal[1], "small", Point([-3.295898, 40.446947])]
    geo_df[len(geo_df.columns)] = ["Feature", "Paris", "*", cur_pal[1], "small", Point([2.285156, 48.048710])]
    geo_df[len(geo_df.columns)] = ["Feature", "Germany", "*", cur_pal[2], "small", Point([9.492188, 51.481383])]
    geo_df[len(geo_df.columns)] = ["Feature", "Sweden", "*", cur_pal[2], "small", Point([16.787109, 63.937372])]
    geo_df[len(geo_df.columns)] = ["Feature", "Russia", "*", cur_pal[3], "small", Point([38.144531, 56.462490])]
    geo_df[len(geo_df.columns)] = ["Feature", "China", "*", cur_pal[3], "small", Point([111.687012, 36.332828])]

    geo_df = geo_df.T

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    map_df.plot(ax = ax, color = "grey")
    geo_df[geo_df["marker"] == "o"].plot(ax = ax, color = geo_df[geo_df["marker"] == "o"]["marker-color"], markersize = 20, marker = "o")
    geo_df[geo_df["marker"] == "v"].plot(ax = ax, color = geo_df[geo_df["marker"] == "v"]["marker-color"], markersize = 20, marker = "^")
    geo_df[geo_df["marker"] == "*"].plot(ax = ax, color = geo_df[geo_df["marker"] == "*"]["marker-color"], markersize = 20, marker = "*")
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig("family_system_worldmap.pdf")
    plt.close('all')



data_whole = pd.read_csv(os.path.join("data/data.csv"))
var_whole = pd.read_csv(os.path.join("data/variables.csv"))
var_sample = pd.read_csv("data/variables_sample.csv")

data_pivot = data_whole.pivot_table(index = "soc_id", columns = "var_id", values="code")

data_pivot[["extended", "equal", "equal2", "system"]] = family_system(data_pivot).T
correlation_analysis_for_test(data_pivot, "equal", (data_pivot["equal"] > -1) & (data_pivot["SCCS246"] > 4), "variables_corr_equal_movable")
correlation_analysis_for_test(data_pivot, "equal2", (data_pivot["equal2"] > -1) & (data_pivot["SCCS246"] > 4), "variables_corr_equal_land2")
correlation_analysis_for_test(data_pivot, "extended", (data_pivot["extended"] > -1) & (data_pivot["SCCS246"] > 4), "variables_corr_extended")

map_df = gpd.read_file(os.path.join(geo_dir,'data/level2.json'))
tdwg_open = open(os.path.join(geo_dir,'data/societies_tdwg.json'), 'r')
tdwg = json.load(tdwg_open)
family_system_worldmap(data_pivot)
