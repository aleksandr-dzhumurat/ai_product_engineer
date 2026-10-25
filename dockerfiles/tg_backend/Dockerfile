FROM python:3.12

ENV PYTHONPATH=/srv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/

COPY dockerfiles/tg_backend/requirements.txt /srv/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY dockerfiles/tg_backend/src /srv/src

CMD ["python3.12", "src/app.py"]
