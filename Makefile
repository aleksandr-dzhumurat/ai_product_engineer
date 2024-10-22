CURRENT_DIR = $(shell pwd)
# include .env
export

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 
