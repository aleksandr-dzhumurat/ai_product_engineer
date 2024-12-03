FROM python:3.12

COPY dockerfiles/api/requirements.txt /srv/requirements.txt
WORKDIR /srv
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]