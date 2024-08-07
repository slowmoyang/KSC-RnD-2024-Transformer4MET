#!/usr/bin/env bash
export PROJECT_PREFIX=$(readlink -f $(dirname ${(%):-%N}))
export PYTHONPATH=${PROJECT_PREFIX}/src:${PYTHONPATH}
micromamba activate diffmet-py311
