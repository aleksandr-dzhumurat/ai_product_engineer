CURRENT_DIR = $(shell pwd)
# include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/minio || true && \
    mkdir -p ${CURRENT_DIR}/data/mlflow || true && \
    mkdir -p ${CURRENT_DIR}/data/zinc_data || true

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	RUN_ENV=LOCAL \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 

build-sagemaker:
	docker build -f ${CURRENT_DIR}/dockerfiles/sagemaker/Dockerfile -t sagemaker:latest .

build-api:
	docker build -f ${CURRENT_DIR}/dockerfiles/api/Dockerfile -t api:latest .

build-tg:
	docker build -f ${CURRENT_DIR}/dockerfiles/tg_backend/Dockerfile -t tg_bot:latest .

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

run-search:
	docker-compose --env-file .env up search

run-api:
	docker-compose --env-file .env up api

run-tg:
	docker-compose --env-file .env up tg_bot
