CURRENT_DIR = $(shell pwd)
SSH_KEY ?= ~/.ssh/nebius1
SSH = ssh -i $(SSH_KEY)
PROJECT_NAME ?= jupyter
-include .env
export


deploy-env:
	scp -i $(SSH_KEY) ${CURRENT_DIR}/.env ${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}

deploy-makefile:
	scp -i $(SSH_KEY) ${CURRENT_DIR}/Makefile ${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}

TUNNEL_PORTS = 8899

check-ports:
	@for port in $(TUNNEL_PORTS); do \
		if lsof -ti :$$port > /dev/null 2>&1; then \
			echo "ERROR: port $$port already in use (PID $$(lsof -ti :$$port))"; \
			exit 1; \
		fi; \
	done
	@echo "All tunnel ports are free"

tunnel: check-ports
	$(SSH) -N \
		-L 8899:localhost:8899 \
		${DEPLOY_USERNAME}@${PROD_HOST}

deploy-secrets: deploy-env deploy-makefile
deploy:
	scp -i $(SSH_KEY) \
		${CURRENT_DIR}/adzhumurat_toxic_homework.ipynb \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/adzhumurat_toxic_homework.ipynb
	scp -i $(SSH_KEY) \
		${CURRENT_DIR}/requirements.txt \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/requirements.txt

connect:
	$(SSH) ${DEPLOY_USERNAME}@${PROD_HOST}

create-env:
	uv venv --python 3.12
	. .venv/bin/activate && uv pip install -r requirements.txt

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	ENV_PATH=${CURRENT_DIR}/.env \
	RUN_ENV=LOCAL \
	jupyter notebook . --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 
