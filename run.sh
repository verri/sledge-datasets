#!/bin/sh
pip install -r requirements.txt

mkdir -p data/
python abalone.py
