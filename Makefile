CURRENT_DIR = $(shell pwd)
# include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/minio || true && \
    mkdir -p ${CURRENT_DIR}/data/mlflow || true && \
    mkdir -p ${CURRENT_DIR}/data/zinc_data || true && \
    mkdir -p ${CURRENT_DIR}/data/pipelines-data || true && \
    mkdir -p ${CURRENT_DIR}/data/models || true && \
    mkdir -p ${CURRENT_DIR}/data/nltk-data || true

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	ENV_PATH=${CURRENT_DIR}/.env \
	RUN_ENV=LOCAL \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 

run-script:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	RUN_ENV=LOCAL \
	uv run python src/${SCRIPT}

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
	--network ai_product_engineer_backtier_network \
	sagemaker:latest param_search

run-model-register:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/src:/opt/ml/model" \
	-v "${CURRENT_DIR}/src:/opt/ml/code" \
	-v "${CURRENT_DIR}/configs:/opt/ml/configs" \
	-e "PYTHONPATH=/opt/ml/code" \
	--network ai_product_engineer_backtier_network \
	sagemaker:latest "python3" code/register_model.py

run-mlflow:
	docker-compose --env-file .env up mlflow

run-search:
	docker-compose --env-file .env up search

run-api:
	docker-compose --env-file .env up api

run-tg:
	docker-compose --env-file .env up tg_bot

labelstudio:
	uv run label-studio

run-train:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/dockerfiles/api/src:/srv/src" \
	-v "${CURRENT_DIR}/src:/srv/src/ml_tools" \
	api:latest train

run-service:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-e CONFIG_DIR=/srv/configs \
	-p 8002:8000 \
	-v "${CURRENT_DIR}/configs:/srv/configs" \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/dockerfiles/api/src:/srv/src" \
	-v "${CURRENT_DIR}/src:/srv/src/ml_tools" \
	--name demo_api_service \
	--network ai_product_engineer_backtier_network \
	api:latest serve

run-streamlit:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-e CONFIG_DIR=/srv/configs \
	-p 8005:8501 \
	-v "${CURRENT_DIR}/configs:/srv/configs" \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/dockerfiles/api/src:/srv/src" \
	-v "${CURRENT_DIR}/src:/srv/src/ml_tools" \
	--name streamlit_service \
	--network ai_product_engineer_backtier_network \
	api:latest streamlit