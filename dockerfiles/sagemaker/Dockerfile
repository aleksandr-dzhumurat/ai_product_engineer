FROM python:3.12

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt /opt/ml/requirements.txt

WORKDIR /opt/ml

# Install dependencies with uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY src /opt/ml/code/

COPY dockerfiles/sagemaker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]