CURRENT_DIR = $(shell pwd)
PROJECT_NAME = blogfeed
# Own venv local to src/scraping/ (uv venv + uv pip install -r requirements.txt)
# - self-contained, matches what `make deploy` ships to the server.
# Override if needed: `make run-force PYTHON=python3`.
PYTHON ?= .venv/bin/python
include .env
export


.PHONY: run-detach run-force stop follow deploy setup connect pull-worker-logs

stop:
	@if [ -f run.pid ]; then \
		if kill $$(cat run.pid) 2>/dev/null; then \
			echo "stopped pid $$(cat run.pid)"; \
		else \
			echo "pid $$(cat run.pid) was not running"; \
		fi; \
		rm -f run.pid; \
	else \
		echo "no run.pid found - nothing to stop"; \
	fi

run-detach: stop
	nohup $(PYTHON) -u feed.py > run.log 2>&1 & echo $$! > run.pid
	@echo "started, pid $$(cat run.pid), logs in run.log"

run-force: stop
	nohup $(PYTHON) -u feed.py --force > run.log 2>&1 & echo $$! > run.pid
	@echo "started (forced immediate cycle), pid $$(cat run.pid), logs in run.log"

follow:
	@touch run.log
	tail -f run.log


# feed.py orchestrates both legs (RSS: blog_rss_scrapers; browser/mhtml:
# browser_use.feed + blog_feed_scrapers.run_parse_cycle, headless+forced on
# the server) plus metrics (Grafana) and, via --generate, llm_client
# (digest generation) and tg_pusher (Telegram send) - every module feed.py
# or its imports touch has to ship together, or it's an ImportError at
# startup, not a degraded feature.
deploy:
	ssh -i $(SSH_KEY) ${DEPLOY_USERNAME}@${PROD_HOST} "mkdir -p /home/${DEPLOY_USERNAME}/${PROJECT_NAME}"
	rsync -avz -e "ssh -i $(SSH_KEY)" \
		$(CURRENT_DIR)/feed.py \
		$(CURRENT_DIR)/blog_rss_scrapers.py \
		$(CURRENT_DIR)/blog_feed_scrapers.py \
		$(CURRENT_DIR)/browser_use.py \
		$(CURRENT_DIR)/metrics.py \
		$(CURRENT_DIR)/llm_client.py \
		$(CURRENT_DIR)/tg_pusher.py \
		$(CURRENT_DIR)/utils.py \
		$(CURRENT_DIR)/blog_sources.jsonl \
		$(CURRENT_DIR)/browser_sites.jsonl \
		$(CURRENT_DIR)/Makefile \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/

# Ships requirements.txt and rebuilds the remote .venv from scratch - run
# this once up front (before the first `deploy`) and again whenever
# requirements.txt changes. `deploy` itself only syncs code, not the venv.
# Also installs the apt packages headless Chromium (via cloakbrowser) needs
# at launch - without these, cloakbrowser's first Chromium launch on a
# fresh host fails with "error while loading shared libraries: libatk-1.0.so.0
# ...". Installed one-by-one (each best-effort, || true) rather than as one
# apt-get call: package names drift across distro versions (e.g. Ubuntu
# 24.04's t64 transition makes libasound2 an ambiguous virtual package, and
# apt-get aborts the *entire* install if any one name is unresolvable), so
# both the old and t64 names are listed and whichever doesn't apply is
# skipped without blocking the rest.
setup:
	ssh -i $(SSH_KEY) ${DEPLOY_USERNAME}@${PROD_HOST} "mkdir -p /home/${DEPLOY_USERNAME}/${PROJECT_NAME}"
	rsync -avz -e "ssh -i $(SSH_KEY)" \
		$(CURRENT_DIR)/requirements.txt \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/
	ssh -i $(SSH_KEY) -t ${DEPLOY_USERNAME}@${PROD_HOST} \
		"sudo apt-get update && for pkg in \
			libatk1.0-0 libatk1.0-0t64 libatk-bridge2.0-0 libatk-bridge2.0-0t64 \
			libnss3 libnspr4 libcups2 libcups2t64 \
			libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
			libxrandr2 libgbm1 libasound2 libasound2t64 libpangocairo-1.0-0 libpango-1.0-0 \
			libcairo2 libatspi2.0-0 libatspi2.0-0t64; \
			do sudo apt-get install -y \$$pkg || true; done"
	ssh -i $(SSH_KEY) -t ${DEPLOY_USERNAME}@${PROD_HOST} \
		"cd /home/${DEPLOY_USERNAME}/${PROJECT_NAME} && rm -rf .venv && uv venv && uv pip install --python .venv/bin/python -r requirements.txt"

connect:
	ssh -i $(SSH_KEY) -t ${DEPLOY_USERNAME}@${PROD_HOST} "cd /home/${DEPLOY_USERNAME}/${PROJECT_NAME} && exec \$$SHELL -l"

pull-worker-logs:
	mkdir -p $(CURRENT_DIR)/data
	rsync -avz -e "ssh -i $(SSH_KEY)" \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/data/runs_journal.jsonl \
		$(CURRENT_DIR)/data/runs_journal.jsonl
	rsync -avz -e "ssh -i $(SSH_KEY)" \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/data/events_log.jsonl \
		$(CURRENT_DIR)/data/events_log.jsonl
	rsync -avz -e "ssh -i $(SSH_KEY)" \
		${DEPLOY_USERNAME}@${PROD_HOST}:/home/${DEPLOY_USERNAME}/${PROJECT_NAME}/data/article_date_cache.json \
		$(CURRENT_DIR)/data/article_date_cache.json
	@echo "pulled runs_journal.jsonl, events_log.jsonl, article_date_cache.json to $(CURRENT_DIR)/data/"
