# import os
# import subprocess
from datetime import datetime
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import threading

# implement function for making plots
# implement temporary folder for logs


class DockerStatsLoggerOld:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __enter__(self):
        print("Starting docker logging")
        # Define the bash script
        # script = f"while true; do docker stats --no-stream | cat >> ./{self.filename}; sleep 0.1; done"
        script = f"""
        while true; do 
            docker stats --no-stream | while read line; do
                echo "$(date -u +"%Y-%m-%d %H:%M:%S")   $line" >> ./{self.filename}
            done
            sleep 0.5
        done
        """

        # Start the script in a background thread
        self.thread = threading.Thread(target=self.run_script, args=(script,))
        self.thread.start()
        return self

    def run_script(self, script):
        # Run the script
        self.process = subprocess.Popen(["bash", "-c", script])

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Ending docker logging")
        # Terminate the script when exiting the context
        self.process.terminate()
        self.thread.join()


# if __name__ == "__main__":
#     # Example usage
#     with DockerStatsLogger(filename="test_w_timestamp") as logger:
#         sleep(10)


class DockerMemoryLogger:
    def __init__(self, filename: str, overwrite_logs: bool = True):
        self.filename = filename
        self.begin_ingesting: str = "-1"
        self.done_ingesting: str = "-1"
        self.begin_benchmarking: str = "-1"
        self.done_benchmarking: str = "-1"
        self.start_logging()
        if overwrite_logs:
            with open(self.filename, "w") as file:  # make file or overwrite
                pass

    def start_logging(self):
        print("starting docker logging")
        # start script
        script = f"""
        while true; do 
            docker stats --no-stream | while read line; do
                echo "$(date -u +"%Y-%m-%d %H:%M:%S")   $line" >> ./{self.filename}
            done
            sleep 0.1
        done
        """
        # Start the script in a background thread
        self.thread = threading.Thread(target=self.run_script, args=(script,))
        self.thread.start()
        sleep(1)

    def stop_logging(self):
        print("stopping docker logging")
        # Terminate the script when exiting the context
        self.process.terminate()
        self.thread.join()

    def get_current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                plt.plot(df_subset.TIMESTAMP, df_subset.MEMORY_USAGE_MB, label=process, marker="x")
                plt.xticks(rotation=45)
                # plt.title(f"{process} memory usage")
                plt.xlabel("timestamp")
                plt.ylabel("Memory Usage (MiB)")
            plt.legend()

        df = self.get_data()

        plt.figure(figsize=(15, 5))
        plot_memory_usage(df)

        timestamps_raw = [
            timestamp
            for timestamp in [
                self.begin_ingesting,
                self.done_ingesting,
                self.begin_benchmarking,
                self.done_benchmarking,
            ]
            if timestamp != "-1"
        ]
        timestamps = pd.to_datetime(timestamps_raw)
        labels = ["begin_ingesting", "done_ingesting", "begin_benchmarking", "done_benchmarking"]

        plt.vlines(timestamps, 0, df.MEMORY_USAGE_MB.max())
        for label, timestamp in zip(labels, timestamps):
            plt.text(
                timestamp,  # type: ignore
                df.MEMORY_USAGE_MB.max(),
                label,
                rotation=45,
                verticalalignment="bottom",
                c="darkblue",
            )
        plt.savefig("./figure_memory_logs.png")

    def get_data(self):
        df = pd.read_csv(f"/Users/oas/Documents/VOD/vod/{self.filename}", delimiter=r"\s\s+", engine="python")
        df = df.query("NAME != 'NAME'")  # remove headers
        columns = df.columns.tolist()  # change name of timestamp
        columns[0] = "TIMESTAMP"
        df.columns = columns
        df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)  # change type to datetime
        df.TIMESTAMP = df.TIMESTAMP + pd.Timedelta(hours=1)

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

    def get_statistics(self) -> dict[str, float]:
        df = self.get_data()

        df_ingesting = df.query(
            f"TIMESTAMP > {self.begin_ingesting} and TIMESTAMP < {self.done_ingesting}"
        ).MEMORY_USAGE_MB
        df_benchmarking = df.query(
            f"TIMESTAMP > {self.begin_benchmarking} and TIMESTAMP < {self.done_benchmarking}"
        ).MEMORY_USAGE_MB

        # get either milvus or qdrant here. or faiss.

        return {
            "ingesting_max": df_ingesting.MEMORY_USAGE_MB.max(),
            "ingesting_mean": df_ingesting.MEMORY_USAGE_MB.mean(),
            "benchmarking_max": df_benchmarking.MEMORY_USAGE_MB.max(),
            "benchmarking_mean": df_benchmarking.MEMORY_USAGE_MB.mean(),
        }


if __name__ == "__main__":
    dm = DockerMemoryLogger("docker_memory_logs.csv", overwrite_logs=False)
    dm.stop_logging()
    dm.make_plots()
