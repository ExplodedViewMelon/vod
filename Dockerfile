# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git

# Clone the specific branch directly to avoid checkout
RUN git clone --branch Vector-database-benchmark https://github.com/ExplodedViewMelon/vod.git /vod

# Set the working directory
WORKDIR /vod

# Update the code to the latest version
RUN git pull

# Set the PYTHONPATH environment variable to include the src directory
ENV PYTHONPATH="/vod/src:$PYTHONPATH"

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install h5py pymilvus

# EXPOSE command is used to inform Docker that the container listens on the specified network port(s) at runtime
EXPOSE 6637

# Command to run the faiss server
CMD ["python3", "src/vod_search/faiss_search/server.py", "--index-path", "faiss_index/index.faiss"]
