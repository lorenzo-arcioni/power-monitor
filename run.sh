#!/bin/bash

uv sync

source .venv/bin/activate

python pzem_server.py 1> /dev/null 2> server.log &