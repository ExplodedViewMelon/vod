# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# %%
folder_path = "/Users/oas/Downloads/benchmarking_results"
# folder_path = "/Users/oas/Documents/VOD/vod/benchmarking_results"
files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))
]
files.sort()  # Sort the files by name
newest_file = files[-2]  # Pick the next newest
print(newest_file)

# %%
df_results = pd.read_csv(newest_file)
df_results.shape

# %%
df_results.query(
    "Dataset == 'gist-960-euclidean' and ingesting_max != -1"
)  # ["Index parameters."].str[-40:]

# %%
df_results = df_results.query("Index != 'None'")  # ignore failed runs
df_results = df_results.query("benchmarking_mean != -1")  # ignore failed runs

# %%
# extract index information
df_results["IndexProvider"] = (
    df_results.Index.str[7:].str.split(",").str[0].str.split(" ").str[0]
)
df_results["IndexType"] = (
    df_results.Index.str[7:].str.split(",").str[0].str.split(" ").str[1]
)

# %%
# extract search parameters from parameter string

df_hnsw_parameters = (
    df_results.query("IndexType == 'HNSW'")["Index parameters."]
    .str.replace("=", ", ")
    .str.split(", ", expand=True)[[2, 4, 6, 7, 8]]
)
df_hnsw_parameters.columns = [
    "M",
    "ef_construction",
    "ef_search",
    "Compression",
    "Metric",
]

df_ivf_parameters = pd.DataFrame()
# TODO fix that this makes errors when no IVF are in benchmark
# df_ivf_parameters = (df_results.query("IndexType == 'IVF'")["Index parameters."]
#     .str.replace(",", "")
#     .str.replace("=", " ")
#     .str.split(" ", expand=True)[[2, 3, 4]]
# )
# df_ivf_parameters.columns = ["nPartitions", "Compression", "Metric"]
""

# %%
# add search parameters to df_results
df_parameters = df_hnsw_parameters.combine_first(df_ivf_parameters)
df_results = pd.concat((df_results, df_parameters), axis=1)

# %%
df_results.shape

# %%
# df_results = df_results.dropna()

# %%
df_results.columns

# %%
import plotly.express as px
import plotly.graph_objects as go

# Your existing DataFrame and plot
fig = px.scatter(
    df_results,
    x="Recall avg",
    y="Search speed avg. (ms)",
    hover_data=["Index", "ingesting_max", "benchmarking_mean", "Dataset"],
    color="IndexProvider",
    size=df_results["ingesting_max"],
    symbol="Dataset",
)

# # Adding a similar scatter plot but with semi-transparency and different size mapping
# fig.add_trace(
#     go.Scatter(
#         x=df_results["Recall avg"],
#         y=df_results["Search speed avg. (ms)"],
#         mode="markers",
#         marker=dict(
#             size=df_results["ingesting_max"] / 25,  # Size based on 'ingesting_max' column
#             opacity=0.1,  # Semi-transparent markers
#             # color=fig.data[0].marker.color,  # Use the same color as the first plot
#             # line=dict(color="MediumPurple", width=2),
#         ),
#         hoverinfo="text",  # You can customize hover info as needed
#         name="Ingesting max",
#     )
# )

fig.show()

# %%
df_results.query("Dataset == 'gist-960-euclidean'")

# %%
df_plot = df_results.query("Compression == 'SQ8'")
plt.scatter(
    df_plot.nPartitions, df_plot["Search speed avg. (ms)"] / 10, label="SQ8", c="red"
)
df_plot = df_results.query("Compression == 'PQ8'")
plt.scatter(df_plot.nPartitions, df_plot["Search speed avg. (ms)"] / 10, label="PQ8")
plt.title("Faiss IVF SQ8 - latency vs n partitions")
plt.xlabel("n partitions")
plt.ylabel("Mean search latency (ms)")
plt.legend()
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_results,
    x="nPartitions",
    y="Search speed avg. (ms)",
    hue="benchmarking_mean",
    style="Compression",
    palette="flare",
    s=70,
)
plt.title("Faiss IVF SQ8 - latency vs n partitions")
plt.xlabel("n partitions")
plt.ylabel("Mean search latency (ms)")
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_results,
    x="nPartitions",
    y="Index build speed (s)",
    hue="benchmarking_mean",
    style="Compression",
    palette="flare",
    s=70,
)
plt.title("Faiss IVF SQ8 - latency vs n partitions")
plt.xlabel("n partitions")
plt.ylabel("Index build speed (s)")
plt.show()

# %%
df_results.columns

# %%
import seaborn as sns

# plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_results,
    x="Search speed avg. (ms)",
    y="Recall avg",
    style="Dataset",
    hue="Index build speed (s)",
    palette="flare",
    s=70,
)
plt.title("Faiss IVF SQ8 - latency vs n partitions")
plt.ylabel("Recall")
plt.xlabel("Search speed (ms)")
plt.show()

# %%
df_plot = df_results.query("Dataset == 'glove-25-angular'")

# plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_plot,
    x="Search speed avg. (ms)",
    y="Recall avg",
    style="Metric",
    hue="Index build speed (s)",
    palette="flare",
    s=70,
)
plt.ylabel("Recall")
plt.xlabel("Search speed (ms)")
plt.show()

# %%
df_results.query('Metric == "IP" and Dataset == "glove-25-angular"')

# %%
df_results.n

# %%
df_plot = df_results.query("IndexType == 'IVF'")

# plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_plot,
    x="Recall avg",
    y="Search speed avg. (ms)",
    style="Metric",
    hue="IndexProvider",
    palette="flare",
    s=70,
)
plt.xlabel("Recall")
plt.ylabel("Search speed (ms)")
plt.show()

# %%
df_plot

# %%
df_plot = df_results.query("IndexType == 'IVF'")

# plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_plot,
    x="Recall avg",
    y="Search speed avg. (ms)",
    style="Metric",
    hue="nPartitions",
    palette="flare",
    s=70,
)
plt.xlabel("Recall")
plt.ylabel("Search speed (ms)")
plt.show()

# %%
df_results.query("Dataset == ")
