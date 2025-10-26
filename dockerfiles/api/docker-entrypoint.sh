#!/usr/bin/env bash

set -o errexit      # make your script exit when a command fails.
set -o nounset      # exit when your script tries to use undeclared variables.

case "$1" in
  train)
    PYTHONPATH=$(pwd)/src python3 src/train.py
    ;;
  serve)
    PYTHONPATH=$(pwd)/src uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  streamlit)
    PYTHONPATH=$(pwd)/src streamlit run src/streamlit_app.py --server.port 8501
    ;;
  hello)
    echo "Hello, engineer!"
    ;;
  *)
    exec "$@"
esac