# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/ExplodedViewMelon/vod.git
WORKDIR /vod
RUN git checkout Vector-database-benchmark
RUN git pull

# it does not actually need to work, it just needs to run the faiss server.
# the above comment shows the stupidity of the approach...
RUN pip install -r requirements.txt
RUN pip install h5py pymilvus
# RUN apt-get install -y curl
# RUN curl -sSL https://install.python-poetry.org | python3 -
# ENV PATH="/root/.local/bin:${PATH}"
# RUN poetry install

EXPOSE 6637

ENV PYTHONPATH="/usr/src:$PYTHONPATH"

# CMD ["python3", "vod_search/faiss_search/server.py", "--index-path", "~/faiss_index/index.faiss"]
