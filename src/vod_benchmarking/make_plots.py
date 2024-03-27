import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

folder_path = "/Users/oas/Documents/VOD/vod/benchmarking_results/2024-03-24_11-58-54"

# print file_names
files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))
    and (file.split("/")[-1][0] != ".")
]
files.sort()  # Sort the files by name
for i, file in enumerate(files):
    print(i, file)


def make_plot(
    df_plot,
    x: pd.Series,
    y: pd.Series,
    log_x=False,
    log_y=False,
    labels: pd.Series = pd.Series(),
    show_labels=False,
    hue=None,
    title="",
    save_as: str = "",
):

    plt.figure(figsize=(6.4, 4.8))

    palette = {
        "faiss": "blue",
        "milvus": "red",
        "qdrant": "green",
    }

    sns.scatterplot(
        data=df_plot,
        x=x,
        y=y,
        style=labels if not show_labels else None,
        hue=hue,
        palette=palette,
    )
    names = {
        "timingsSearch": "Search speed (s)",
        "recall_at_1": "Recall@1",
        "memoryBenchmark": "Mean memory consumption (MB)",
        "timingBuildIndex": "Index build duration (s)",
    }
    plt.xlabel(names[x.name])  # type: ignore
    plt.ylabel(names[y.name])  # type: ignore

    if len(labels) > 0 and show_labels:
        for i in df_plot.index:
            plt.text(
                x[i],
                y[i],
                "   " + str(labels[i]),
                ha="left",
                va="top",
                rotation=-35,
                clip_on=True,
            )
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    # Calculate the x-axis limits with padding, taking log scale into consideration
    if log_x:
        x_lim = plt.xlim()
        x_pad = (np.log10(x_lim[1]) - np.log10(x_lim[0])) * 0.3
        plt.xlim(10 ** (np.log10(x_lim[0]) - x_pad), 10 ** (np.log10(x_lim[1]) + x_pad))
    else:
        x_lim = plt.xlim()
        x_pad = (x_lim[1] - x_lim[0]) * 0.3
        plt.xlim(x_lim[0] - x_pad, x_lim[1] + x_pad)

    if log_y:
        y_lim = plt.ylim()
        y_pad = (np.log10(y_lim[1]) - np.log10(y_lim[0])) * 0.15
        plt.ylim(10 ** (np.log10(y_lim[0]) - y_pad), 10 ** (np.log10(y_lim[1])))
    else:
        y_lim = plt.ylim()
        y_pad = (y_lim[1] - y_lim[0]) * 0.15
        plt.ylim(y_lim[0] - y_pad, y_lim[1])

    # Add a grid to the plot for better readability
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.title(title)

    if save_as:
        plt.savefig(save_as + ".png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    plt.clf()


def make_all_plots(
    df_plot,
    benchmark_label: str,
    sweep_param: str,
    save_folder: str = "",
    log_x1=True,  # search speed
    log_y1=False,  # search recall
    log_x2=False,  # index build
    log_y2=False,  # memory consumption
    show_labels=False,
    title="",
):

    save_as = f"{save_folder}/{benchmark_label}"

    make_plot(
        df_plot,
        df_plot.timingsSearch,
        df_plot.recall_at_1,
        labels=df_plot[sweep_param],
        hue=df_plot.indexProvider,
        # title=f"speed vs recall - {df_plot.iloc[0].label}",
        title=title + " - Search",
        save_as=save_as + "_speed" if save_folder else "",
        log_x=log_x1,
        log_y=log_y1,
        show_labels=show_labels,
    )
    make_plot(
        df_plot,
        df_plot.timingBuildIndex,
        df_plot.memoryBenchmark,
        labels=df_plot[sweep_param],
        hue=df_plot.indexProvider,
        # title=f"memory vs index build time - {df_plot.iloc[0].label}",
        title=title + " - Index",
        save_as=save_as + "_mem" if save_folder else "",
        log_x=log_x2,
        log_y=log_y2,
        show_labels=show_labels,
    )


dfs = {}
for file in files:
    df = pd.read_csv(file)
    df = df[df.error.isna()]
    df.memoryIngesting[df.memoryIngesting == -1] = pd.NA  # such that it does not show
    df.memoryBenchmark = df.memoryBenchmark - df.memoryBaseline
    df.memoryIngesting = df.memoryIngesting - df.memoryBaseline

    filename = file.split("/")[-1][:-4]
    dfs[filename] = df

image_folder = f"{folder_path}/images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# auxillary tests

# make_all_plots(
#     dfs["SpreadTest_n_batches_1"],
#     "SpreadTest_n_batches_1",
#     "indexType",
#     save_folder=image_folder,
#     title="Results variance test",
# )

# df_st_10 = dfs["SpreadTest_n_batches_10"]
# df_st_10["nBatches"] = 10
# df_st_1 = dfs["SpreadTest_n_batches_1"]
# df_st_1["nBatches"] = 1
# df_st = pd.concat([df_st_1, df_st_10]).reset_index()

# df_st_subset = df_st.query("indexProvider == 'faiss' and indexType == 'IVF'")
# make_all_plots(
#     df_st_subset,
#     "SpreadTestFaissIVF",
#     "nBatches",
#     save_folder=image_folder,
# )
# df_st_subset = df_st.query("indexProvider == 'faiss' and indexType == 'HNSW'")
# make_all_plots(
#     df_st_subset,
#     "SpreadTestFaissHNSW",
#     "nBatches",
#     save_folder=image_folder,
# )

# df_st_subset = df_st.query("indexProvider == 'milvus' and indexType == 'IVF'")
# make_all_plots(
#     df_st_subset,
#     "SpreadTestMilvusIVF",
#     "nBatches",
#     save_folder=image_folder,
# )
# df_st_subset = df_st.query("indexProvider == 'milvus' and indexType == 'HNSW'")
# make_all_plots(
#     df_st_subset,
#     "SpreadTestMilvusHNSW",
#     "nBatches",
#     save_folder=image_folder,
# )

# df_st_subset = df_st.query("indexProvider == 'qdrant' and indexType == 'HNSW'")
# make_all_plots(
#     df_st_subset,
#     "SpreadTestQdrantHNSW",
#     "nBatches",
#     save_folder=image_folder,
# )

# df_overhead1x1000 = dfs["BatchSpeedUpTest_1x1000_batch"]
# df_overhead1000x1 = dfs["BatchSpeedUpTest_1000x1_batch"]
# df_overhead1x1000["shape"] = "1x1000"
# df_overhead1000x1["shape"] = "1000x1"
# df_overhead = pd.concat((df_overhead1000x1, df_overhead1x1000)).reset_index()

# make_all_plots(
#     df_overhead,
#     "BatchSpeedUpTest",
#     "shape",
#     save_folder=image_folder,
#     title="Batch overhead speed test",
# )

# # fix that memoryBenchmark = -1 due to too fast searches.
# dfs["DatasetTest"].loc[
#     dfs["DatasetTest"].memoryBenchmark < 0, "memoryBenchmark"
# ] = pd.NA
# make_all_plots(
#     dfs["DatasetTest"].query("indexType == 'IVF'"),
#     "DatabaseTestIVF",
#     "dataset",
#     save_folder=image_folder,
#     title="Dataset test IVF",
# )
# make_all_plots(
#     dfs["DatasetTest"].query("indexType == 'HNSW'"),
#     "DatabaseTestHNSW",
#     "dataset",
#     save_folder=image_folder,
#     title="Dataset test HNSW",
# )

# # parameter sweep tests

# make_all_plots(
#     dfs["Sweep_HNSW_M"],
#     "Sweep_HNSW_M",
#     "M",
#     save_folder=image_folder,
#     show_labels=True,
#     title="HNSW M test",
# )
# make_all_plots(
#     dfs["Sweep_HNSW_efConstruction"],
#     "Sweep_HNSW_efConstruction",
#     "efConstruction",
#     save_folder=image_folder,
#     show_labels=True,
#     title="HNSW ef-construction test",
# )
# make_all_plots(
#     dfs["Sweep_HNSW_efSearch"],
#     "Sweep_HNSW_efSearch",
#     "efSearch",
#     save_folder=image_folder,
#     show_labels=True,
#     title="HNSW ef-search test",
# )
# make_all_plots(
#     dfs["Sweep_IVF_nPartitions"],
#     "Sweep_IVF_nPartitions",
#     "nPartitions",
#     save_folder=image_folder,
#     show_labels=True,
#     title="IVF n-partitions test",
# )
# make_all_plots(
#     dfs["Sweep_IVF_nProbe"],
#     "Sweep_IVF_nProbe",
#     "nProbe",
#     save_folder=image_folder,
#     show_labels=True,
#     title="IVF n-probe test",
# )


df_plot = dfs["Sweep_SQ_HNSW"]
df_plot["preprocessing"] = df_plot["preprocessing"] + ", " + df_plot["M"].astype(str)
make_all_plots(
    df_plot,
    "Sweep_SQ_HNSW",
    "preprocessing",
    save_folder=image_folder,
    show_labels=True,
    title="SQ HNSW test",
)

df_plot = dfs["Sweep_SQ_IVF"]
df_plot["preprocessing"] = (
    df_plot["preprocessing"] + ", " + df_plot["nPartitions"].astype(str)
)
make_all_plots(
    df_plot,
    "Sweep_SQ_IVF",
    "preprocessing",
    save_folder=image_folder,
    show_labels=True,
    title="SQ IVF test",
)

df_plot = dfs["Sweep_PQ"]
df_plot["preprocessing"] = df_plot["preprocessing"] + ", " + df_plot["indexType"]
make_all_plots(
    dfs["Sweep_PQ"],
    "Sweep_PQ",
    "preprocessing",
    save_folder=image_folder,
    show_labels=True,
    title="PQ test",
)

print("Success")


# ____ MISC ____

# make_all_plots(
#     dfs["Sweep_HNSW_M"],
#     "Sweep_HNSW_M",
#     "M",
#     log_x1=True,
# )
#

# make_plot(
#     df_plot,
#     df_plot.timingBuildIndex,
#     df_plot.memoryBenchmark,
#     labels=df_plot["M"],
#     style=None,
#     hue=df_plot.indexProvider,
#     title=f"memory vs index build time - {df_plot.iloc[0].label}",
#     save_as="",
#     log_x=False,
#     log_y=False,
# )

# palette = {
#     "faiss": "blue",
#     "milvus": "red",
#     "qdrant": "green",
# }
# light_palette = {
#     "faiss": "#add8e6",
#     "milvus": "#f08080",
#     "qdrant": "#90ee90",
# }
# df_plot = dfs["Sweep_HNSW_M"]

# df_plot.memoryBenchmark = df_plot.memoryBenchmark - df_plot.memoryBaseline
# df_plot.memoryIngesting[df_plot.memoryIngesting == -1] = pd.NA
# df_plot.memoryIngesting = df_plot.memoryIngesting - df_plot.memoryBaseline

# # benchmarking memory
# x = df_plot.timingBuildIndex
# y1 = df_plot.memoryBenchmark
# y2 = df_plot.memoryIngesting
# style = None
# hue = df_plot.indexProvider
# labels = df_plot["M"]
# sns.lineplot(
#     data=df_plot,
#     x=x,
#     y=y1,
#     style=style,
#     hue=hue,
#     palette=palette,
# )
# sns.lineplot(
#     data=df_plot, x=x, y=y2, style=style, hue=hue, palette=light_palette, legend=False
# )
# plt.xlabel(x.name)  # type: ignore
# plt.ylabel(y1.name)  # type: ignore
# if len(labels) > 0:
#     for i in df_plot.index:
#         plt.text(x[i], y1[i], "   " + str(labels[i]), ha="left", va="top", clip_on=True)
# plt.legend()
# plt.show()
