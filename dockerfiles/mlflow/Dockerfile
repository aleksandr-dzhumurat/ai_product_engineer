FROM python:3.12

COPY requirements.txt /srv/requirements.txt
WORKDIR /srv
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install setuptools && \
    pip install --upgrade setuptools

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]