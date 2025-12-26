[![Лекция 04 vol 1: Организация кода в ML проектах](http://img.youtube.com/vi/yFGYz8XAw30/0.jpg)](http://www.youtube.com/watch?v=yFGYz8XAw30 "Лекция 04 vol 1: Организация кода в ML проектах")

start with [installing uv](https://docs.astral.sh/uv/getting-started/installation/)

Creating python environment 
 
```shell
uv venv --python 3.13;
source .venv/bin/activate;
uv pip install -r requirements.txt
```

# Old way: pyenv 
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


```shell
cd python-env & \
	uv pip install -r ../requirements.txt  \
	uv pip freeze > temp_requirements.txt \
	uv add $(cat temp_requirements.txt | tr '\n' ' ')
	cd ..
```

Докер для винды (обе опции) https://docs.docker.com/desktop/setup/install/windows-install/

Подробнее про WSL: https://learn.microsoft.com/en-us/windows/wsl/install

# Remote 

Плюс действительно интересная опция с Cloud.ru https://cloud.ru/, действительно есть возможность  Удаленно подключиться к убунте и настроить докер по инструкции для убунты
