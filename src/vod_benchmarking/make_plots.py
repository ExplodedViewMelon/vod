import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = "/Users/oas/Downloads/benchmarking_results/2024-03-24_11-58-54"
# print file_names
files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))
    and (file.split("/")[-1][0] != ".")
]
files.sort()  # Sort the files by name
for file in files:
    print(file)


def make_plots(
    df_plot,
    x: pd.Series,
    y: pd.Series,
    log=False,
    labels: pd.Series = pd.Series(),
    style=None,
    hue=None,
    title="",
    save: bool = False,
):
    sns.scatterplot(data=df_plot, x=x, y=y, style=style, hue=hue)
    plt.xlabel(x.name)  # type: ignore
    plt.ylabel(y.name)  # type: ignore

    if len(labels) > 0:
        for i in df_plot.index:
            plt.text(
                x[i], y[i], "   " + str(labels[i]), ha="left", va="top", clip_on=True
            )

    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)
    if save:
        plt.savefig("test_fig.png")
    else:
        plt.show()


def make_all_plots(df_plot, sweep_param: str):
    make_plots(
        df_plot,
        df_plot.timingsSearch,
        df_plot.recall,
        labels=df_plot[sweep_param],
        style=None,
        hue=df_plot.indexProvider,
        title=f"speed vs recall - {df_plot.iloc[0].label}",
    )
    make_plots(
        df_plot,
        df_plot.timingsSearch,
        df_plot.recall,
        labels=df_plot[sweep_param],
        style=None,
        hue=df_plot.indexProvider,
        log=True,
        title=f"(log) speed vs recall - {df_plot.iloc[0].label}",
    )
    make_plots(
        df_plot,
        df_plot.timingBuildIndex,
        df_plot.memoryBenchmark,
        labels=df_plot[sweep_param],
        style=None,
        hue=df_plot.indexProvider,
        title=f"memory vs index build time - {df_plot.iloc[0].label}",
    )
    make_plots(
        df_plot,
        df_plot.timingBuildIndex,
        df_plot.memoryBenchmark,
        labels=df_plot[sweep_param],
        style=None,
        hue=df_plot.indexProvider,
        log=True,
        title=f"(log) memory vs index build time - {df_plot.iloc[0].label}",
    )


dfs = []
for file in files:

    df = pd.read_csv(file)
    df = df[df.error.isna()]
    dfs.append(df)


make_all_plots(dfs[4], "M")
