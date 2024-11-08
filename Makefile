CURRENT_DIR = $(shell pwd)
# include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/minio || true && \
    mkdir -p ${CURRENT_DIR}/data/mlflow || true

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 

build-sagemaker:
	docker build -f ${CURRENT_DIR}/dockerfiles/sagemaker/Dockerfile -t sagemaker:latest .

run-sagemaker-train:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/src:/opt/ml/model" \
	-v "${CURRENT_DIR}/src:/opt/ml/code" \
	sagemaker:latest train

run-param-tuning:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/src:/opt/ml/model" \
	-v "${CURRENT_DIR}/src:/opt/ml/code" \
	--network ml_for_products_prj_network \
	sagemaker:latest param_search

run-mlflow:
	docker-compose --env-file .env up mlflow
