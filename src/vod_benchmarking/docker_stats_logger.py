# import os
# import subprocess
from time import sleep


# class DockerMonitor:
#     def __init__(self) -> None:
#         # Run the bash process asynchronously
#         # self.bash_command = 'while true; do docker stats --no-stream | cat >> ./`date -u +"%Y%m%d.csv"`; sleep 1; done'
#         self.bash_command: list[str] = 'while true; do docker stats --no-stream | cat >> ./`date -u +"%Y%m%d.csv"`; sleep 10; done'.split(" ")
# "

#     def __enter__(self):
#         self.process = subprocess.Popen(self.bash_command)

#     def __exit__(self, type, value, traceback):
#         self.process.kill()

#     def get_data(self):
#         pass


# if __name__ == "__main__":
#     with DockerMonitor() as dm:
#         sleep(5)

#     print("done")


import subprocess
import threading


class DockerStatsLogger:
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
            sleep 0.1
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


if __name__ == "__main__":
    # Example usage
    with DockerStatsLogger(filename="test_w_timestamp") as logger:
        sleep(10)
