FROM python:3.12

COPY requirements.txt /opt/ml/requirements.txt

WORKDIR /opt/ml
RUN pip install --upgrade setuptools & \
    python -m pip install --no-cache-dir -r requirements.txt
COPY src /opt/ml/code/

COPY dockerfiles/sagemaker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]