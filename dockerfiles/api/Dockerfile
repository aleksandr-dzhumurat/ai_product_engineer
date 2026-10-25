FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY dockerfiles/api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY dockerfiles/api/requirements.txt /srv/requirements.txt

WORKDIR /srv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

EXPOSE 8000

COPY src /srv/ml_tools
COPY dockerfiles/api/src /srv/src
COPY configs /srv/configs

ENTRYPOINT ["docker-entrypoint.sh"]
