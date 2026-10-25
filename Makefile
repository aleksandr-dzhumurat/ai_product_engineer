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
	-e DATA_DIR=/opt/ml/data \
	-v "${CURRENT_DIR}/data:/opt/ml/data" \
	-v "${CURRENT_DIR}/src:/opt/ml/model" \
	-v "${CURRENT_DIR}/src:/opt/ml/code" \
	sagemaker:latest train

run-param-tuning:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/opt/ml/data" \
	-v "${CURRENT_DIR}/src:/opt/ml/model" \
	-v "${CURRENT_DIR}/src:/opt/ml/code" \
	--network ai_product_engineer_backtier_network \
	sagemaker:latest param_search

run-model-register:
	docker run -it --rm \
	--env-file ${CURRENT_DIR}/.env  \
	-v "${CURRENT_DIR}/data:/opt/ml/data" \
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
	-e DATA_DIR=/srv/data \
	-p 8002:8000 \
	-v "${CURRENT_DIR}/data:/srv/data" \
	-v "${CURRENT_DIR}/dockerfiles/api/src:/srv/src" \
	-v "${CURRENT_DIR}/src:/srv/ml_tools" \
	--name demo_api_service \
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

run-chroma:
	docker-compose --env-file .env up chroma

run-weaviate:
	docker-compose --env-file .env up weaviate

ingest: run-chroma
	uv run python src/ingestion.py "$(SOURCE_DIR)" \
	  --embedding-model $(EMBED_MODEL) \
	  --ollama-host $(OLLAMA_HOST) \
	  --batch-size $(BATCH_SIZE)

rag-data-ingestion:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	RUN_ENV=LOCAL \
	.venv/bin/python src/rag/ingestion.py  --embedding-model qwen2.5:1.5b --collection docs1 "ai_product_engineer/jupyter_notebooks"
#	.venv/bin/python src/rag/ingestion.py "ai_product_engineer/jupyter_notebooks" --reset-collection
#	.venv/bin/python src/rag/ipynb2md.py ${CURRENT_DIR}/jupyter_notebooks
chat:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	RUN_ENV=LOCAL \
	uv run python src/assistant/chat.py

run-tg-interviewer:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	DOTENV_FILE=${CURRENT_DIR}/.env \
	uv run python src/rag/tg_bot.py --mode rag

run-english-teacher:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	DOTENV_FILE=${CURRENT_DIR}/.env \
	uv run python src/rag/tg_bot.py --mode eng

REFS ?= data/references
MANIFEST ?= $(REFS)/MANIFEST

# Make treats positional arguments as targets. Consume repository goals only
# when clone is requested, so other invocations still report unknown targets.
ifneq ($(filter clone,$(MAKECMDGOALS)),)
override CLONE_GOALS := $(strip $(foreach goal,$(MAKECMDGOALS),$(if $(findstring /,$(goal)),$(goal))))
ifneq ($(CLONE_GOALS),)
.PHONY: $(CLONE_GOALS)
$(CLONE_GOALS): ;
endif
endif

.PHONY: clone
clone:
	@usage() { echo "usage: make clone owner/repo (or make clone REPO_PATH=https://github.com/owner/repo)" >&2; exit 1; }; \
	if [ -n "$$REPO_PATH" ] && [ -n "$$CLONE_GOALS" ]; then usage; fi; \
	repo_path="$${REPO_PATH:-$$CLONE_GOALS}"; \
	repo_path=$${repo_path#https://github.com/}; \
	repo_path=$${repo_path%/}; \
	case "$$repo_path" in \
		*/*/*|/*|*/|*[!a-zA-Z0-9_./-]*) usage ;; \
		*/*) ;; \
		*) usage ;; \
	esac; \
	repo_name=$${repo_path##*/}; \
	repo_name=$${repo_name%.git}; \
	case "$$repo_name" in ''|.|..) usage ;; esac; \
	repo_url="https://github.com/$$repo_path"; \
	dest="$$REFS/$$repo_name"; \
	if [ -d "$$dest" ]; then \
		echo "$$dest already exists, skipping"; \
	else \
		mkdir -p "$$REFS" && \
		git clone --depth 1 --quiet "$$repo_url" "$$dest" && \
		commit=$$(git -C "$$dest" rev-parse --short HEAD) && \
		printf "%s\t%s\t%s\t%s\n" "$$repo_name" "$$repo_url" "$$commit" "$$(date +%F)" >> "$$MANIFEST" && \
		echo "cloned $$repo_url -> $$dest @ $$commit"; \
	fi
