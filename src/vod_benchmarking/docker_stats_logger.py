# import os
# import subprocess
from datetime import datetime
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import threading
import os
import warnings
from numpy import ndarray
from typing import Tuple
import random

# DONE implement function for making plots
pd.options.mode.chained_assignment = None  # default='warn'


class DockerMemoryLogger:
    def __init__(
        self,
        timestamp: str,
        *,
        overwrite_logs: bool = True,
        timeout=600,
    ):
        self.folder_path: str = f"./docker_memory_logs/{timestamp}/"
        self.begin_baseline: str = "-1"
        self.done_baseline: str = "-1"
        self.begin_ingesting: str = "-1"
        self.done_ingesting: str = "-1"
        self.begin_benchmarking: str = "-1"
        self.done_benchmarking: str = "-1"
        self.timeout: int = timeout

        self.index_specification = str(random.randint(0, 1000000))
        self.path_to_log = f"{self.folder_path}{self.index_specification}.csv"

        # Create folder if it does not exist
        os.makedirs(self.folder_path, exist_ok=True)

        self.start_logging()
        if overwrite_logs:
            with open(self.path_to_log, "w") as file:  # make file or overwrite
                pass

        print(f"Writing logs to {self.path_to_log}")

    def start_logging(self):
        print("starting docker logging")
        # start script
        script = f"""
        start_time=$(date +%s)
        end_time=$((start_time + {self.timeout}))

        while true; do

            current_time=$(date +%s)
            if [ $current_time -ge $end_time ]; then
                break
            fi

            docker stats --no-stream | while read line; do
                echo "$(date -u +"%Y-%m-%d %H:%M:%S")   $line" >> {self.path_to_log}
            done
            sleep 0.1
        done
        """
        # Start the script in a background thread
        self.thread = threading.Thread(target=self.run_script, args=(script,))
        self.thread.start()
        sleep(1)

    def stop_logging(self):
        print("stopping docker logging...")
        n_tries = 5
        while self.process.poll() is None and n_tries > 0:
            # Terminate the script when exiting the context
            self.process.terminate()
            self.thread.join()
            n_tries -= 1
            sleep(0.5)

    def get_current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_begin_baseline(self):
        self.begin_baseline = self.get_current_datetime()

    def set_done_baseline(self):
        self.done_baseline = self.get_current_datetime()

    def set_begin_ingesting(self):
        self.begin_ingesting = self.get_current_datetime()

    def set_done_ingesting(self):
        self.done_ingesting = self.get_current_datetime()

    def set_begin_benchmarking(self):
        self.begin_benchmarking = self.get_current_datetime()

    def set_done_benchmarking(self):
        self.done_benchmarking = self.get_current_datetime()

    def run_script(self, script):
        # Run the script
        self.process = subprocess.Popen(["bash", "-c", script])

    def make_plots(self):
        def plot_memory_usage(df):
            for process in df.NAME.unique():
                df_subset = df.query(f"NAME == '{process}'")
                # t = range(0,len(memory_usage))
                plt.plot(
                    df_subset.TIMESTAMP,
                    df_subset.MEMORY_USAGE_MB,
                    label=process,
                    marker="x",
                )
                plt.xticks(rotation=45)
                # plt.index_specification(f"{process} memory usage")
                plt.xlabel("timestamp")
                plt.ylabel("Memory Usage (MiB)")
            plt.legend()

        df = self.get_data()

        plt.figure(figsize=(15, 5))
        plot_memory_usage(df)

        timestamps_raw = [
            timestamp
            for timestamp in [
                self.begin_baseline,
                self.done_baseline,
                self.begin_ingesting,
                self.done_ingesting,
                self.begin_benchmarking,
                self.done_benchmarking,
            ]
            if timestamp != "-1"
        ]
        timestamps = pd.to_datetime(timestamps_raw)
        labels = [
            "begin_baseline",
            "done_baseline",
            "begin_ingesting",
            "done_ingesting",
            "begin_benchmarking",
            "done_benchmarking",
        ]

        plt.vlines(timestamps, 0, df.MEMORY_USAGE_MB.max())
        for label, timestamp in zip(labels, timestamps):  # type: ignore
            plt.text(
                timestamp,  # type: ignore
                df.MEMORY_USAGE_MB.max(),
                label,
                rotation=45,
                verticalalignment="bottom",
                c="darkblue",
            )
        plt.savefig(f"{self.folder_path}{self.index_specification}.png")

    def get_data(self):
        df = pd.read_csv(
            f"{self.folder_path}{self.index_specification}.csv",
            delimiter=r"\s\s+",
            engine="python",
        )
        df = df.query("NAME != 'NAME'")  # remove headers
        df = df.query("NAME != '0.00%'")  # remove headers
        columns = df.columns.tolist()  # change name of timestamp
        columns[0] = "TIMESTAMP"
        df.columns = columns
        df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)  # change type to datetime
        # df.TIMESTAMP = df.TIMESTAMP + pd.Timedelta(hours=1) # NOTE fixes the problem on mac.

        def convert_memory_usage(s):
            if "KiB" in s:
                return float(s.split("/")[0][:-4]) / 1024
            elif "MiB" in s:
                return float(s.split("/")[0][:-4])
            elif "GiB" in s:
                return float(s.split("/")[0][:-4]) * 1024
            else:
                return None

        df["MEMORY_USAGE_MB"] = df["MEM USAGE / LIMIT"].apply(convert_memory_usage)
        return df

    def get_statistics(self) -> Tuple[ndarray, ndarray, ndarray]:
        df = self.get_data()

        # # only measure the memory consumption of the process that we are benchmark
        # df = df[df["NAME"].str.split("-").str[0] == self.searchMasterName]
        assert len(df), "docker memory logger has no entries"

        # get cummulative of each process, if multiple
        df = df.groupby("TIMESTAMP").sum()

        df_baseline: pd.Series = df.query(
            f"TIMESTAMP > '{self.begin_baseline}' and TIMESTAMP < '{self.done_baseline}'"
        ).MEMORY_USAGE_MB
        df_ingesting: pd.Series = df.query(
            f"TIMESTAMP > '{self.begin_ingesting}' and TIMESTAMP < '{self.done_ingesting}'"
        ).MEMORY_USAGE_MB
        df_benchmarking: pd.Series = df.query(
            f"TIMESTAMP > '{self.begin_benchmarking}' and TIMESTAMP < '{self.done_benchmarking}'"
        ).MEMORY_USAGE_MB

        return (
            df_baseline.to_numpy(),
            df_ingesting.to_numpy(),
            df_benchmarking.to_numpy(),
        )

        # return {
        #     "memoryBaselineMax": df_baseline.max(),
        #     "memoryBaselineMean": df_baseline.mean(),
        #     "memoryIngestingMax": df_ingesting.max(),
        #     "memoryIngestingMean": df_ingesting.mean(),
        #     "memoryBenchmarkingMax": df_benchmarking.max(),
        #     "memoryBenchmarkingMean": df_benchmarking.mean(),
        # }


# if __name__ == "__main__":
#     dm = DockerMemoryLogger(index_specification="wet_run_test", overwrite_logs=False)
#     dm.set_begin_benchmarking()
#     dm.stop_logging()
#     dm.make_plots()


# meow meow
