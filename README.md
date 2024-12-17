# ml_for_products

Install deps

```bash
brew install openssl xz gdbm
```

Install python versions

```
pyenv install 3.12 && \
pyenv virtualenv 3.12 mlproducts-env \
source ~/.pyenv/versions/mlproducts-env/bin/activate
```

Install requirements
```
python3 -m pip install -r requirements.txt
```

run jupyter
```
make run-jupyter
```

# Data preparation

Step 1: download data to the local machine or copy to our google drive: [ML for priducts](https://drive.google.com/drive/folders/1FMLKfNZZyFgzOhWjOiyeN3XvCsjT5-ET?usp=drive_link)


Extact data

```shell
ROOT_DATA_DIR=$(pwd)/data python3 src/bidmachine/prepare_data.py
```

# Lectures

Lesson 1: train.py

Homework 1: prepare predict.py
* load model from `.cb` file
* apply to valid data
* eval metrics

```python

```



# Lesson 2

run search
```
docker-compose up search
```



API

```shell
make build-api
```

```shell
make run-api
```

```shell
http://0.0.0.0:8000/docs
```

Build bot

```shell
build-tg
```

t.me/mai_2024_crash_course_bot