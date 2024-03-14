import pandas as pd
import matplotlib.pyplot as plt
import os


def get_and_prepare_newest_file():
    folder_path = "/Users/oas/Downloads/benchmarking_results"
    # folder_path = "/Users/oas/Documents/VOD/vod/benchmarking_results"
    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]
    files.sort()  # Sort the files by name
    newest_file = files[-1]  # Pick the newest
    print(newest_file)

    df_results = pd.read_csv(newest_file)
    df_results.shape

    df_results.query("RecallMean != -1")  # ["Index parameters."].str[-40:]

    df_results = df_results.query("Index != 'None'")  # ignore failed runs
    df_results = df_results.query("RecallMean != -1.0")  # ignore failed runs

    # this should be empty
    df_results[df_results.benchmarkingMean.isna()]

    # extract index information
    df_results["IndexProvider"] = df_results.Index.str[7:].str.split(",").str[0].str.split(" ").str[0]
    df_results["IndexType"] = df_results.Index.str[7:].str.split(",").str[0].str.split(" ").str[1]

    # extract search parameters from parameter string

    df_hnsw_parameters = (
        df_results.query("IndexType == 'HNSW'")["IndexParameters"]
        .str.replace("=", ", ")
        .str.split(", ", expand=True)[[2, 4, 6, 7, 8]]
    )
    df_hnsw_parameters.columns = ["M", "EfConstruction", "EfSearch", "Compression", "Metric"]

    df_ivf_parameters = pd.DataFrame()
    # TODO fix that this makes errors when no IVF are in benchmark
    df_ivf_parameters = (
        df_results.query("IndexType == 'IVF'")["IndexParameters"]
        .str.replace(",", "")
        .str.replace("=", " ")
        .str.split(" ", expand=True)[[2, 4, 5, 6]]
    )
    df_ivf_parameters.columns = ["NPartitions", "NProbe", "Compression", "Metric"]

    # add search parameters to df_results
    df_parameters = df_hnsw_parameters.combine_first(df_ivf_parameters)
    df_results = pd.concat((df_results, df_parameters), axis=1)

    df_results.columns = [col[0].upper() + col[1:] for col in df_results.columns]
    return df_results


df_results = get_and_prepare_newest_file()
print(df_results.columns)
print(df_results.head())

########### PLOTTING ###########

import seaborn as sns

# # IVF SQ SWEEP
# df_plot = df_results[df_results.Compression.str[:2] == "SQ"].query("IndexType == 'IVF'")
# sns.scatterplot(
#     data=df_plot,
#     x="SearchSpeedAverage",
#     y="RecallAt100Mean",
#     style="Compression",
#     hue="TimerBuildIndexMean",
#     palette="flare",
#     s=70,
# )
# plt.title("Faiss IVF SQ SWEEP")
# plt.ylabel("Recall")
# plt.xlabel("Search speed (ms)")
# plt.show()

df_plot = df_results[df_results.Compression.str[:2] == "PQ"].query("IndexType == 'HNSW'")
sns.scatterplot(
    data=df_plot,
    x="SearchSpeedAverage",
    y="RecallAt1000Mean",
    style="Compression",
    hue="TimerBuildIndexMean",
    palette="flare",
    s=70,
)
for line in range(df_plot.shape[0]):
    point = df_plot.iloc[line]
    plt.text(point["SearchSpeedAverage"], point["RecallAt1000Mean"], f"   {point['M']}", fontsize=9)
plt.title("Faiss HNSW SQ SWEEP")
plt.ylabel("Recall")
plt.xlabel("Search speed (ms)")
plt.show()
