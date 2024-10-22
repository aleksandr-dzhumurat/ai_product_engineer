#!/usr/bin/env bash

set -o errexit      # make your script exit when a command fails.
set -o nounset      # exit when your script tries to use undeclared variables.

case "$1" in
  train)
    PYTHONPATH=$(pwd)/code python3 code/train.py
    ;;
  param_search)
    PYTHONPATH=$(pwd)/code python3 code/parameters_tuning.py
    ;;
  serve)
    uvicorn src.main:app --host $FASTAPI_HOST --port $FASTAPI_PORT --reload
    ;;
  jupyter)
    jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser
    ;;
  *)
    exec "$@"
esac