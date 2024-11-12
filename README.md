# ml_for_products

Extact data

```shell
ROOT_DATA_DIR=$(pwd)/data python3 src/bidmachine/prepare_data.py
```

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