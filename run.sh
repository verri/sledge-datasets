#!/bin/bash
pip install -r requirements.txt

parallel () {
  while [ "$(jobs | wc -l)" -ge $(nproc) ]
  do
    wait -n
  done
  echo "$@"
  "$@" &
}

trap 'kill $(jobs -p); exit 1' SIGINT SIGTERM SIGKILL

mkdir -p data/

parallel python abalone.py
parallel python iris.py

wait
