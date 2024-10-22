FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY dockerfiles/api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY dockerfiles/api/requirements.txt /srv/requirements.txt

WORKDIR /srv

RUN uv pip install --system --no-cache -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]