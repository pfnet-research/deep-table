FROM python:slim-buster

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./ /deep-table/
WORKDIR /deep-table/

RUN pip install -e .
WORKDIR /workspace/
