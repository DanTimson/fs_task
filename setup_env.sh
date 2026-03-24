#!/bin/bash

nvidia-smi

env | sort

echo "> Activating env"
uv sync

echo "> Downloading from hf sources"
hf download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B 
hf download Elriggs/openwebtext-100k --repo-type dataset --local-dir ./datasets/openwebtext-100k

echo "> Done"
