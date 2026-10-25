"""Load the OCR dataset from ``DATA_DIR`` and run a grid search over
neural network architectures for binary text classification.

Expects ``ocr_dataset.zip`` to be present in the directory specified by the
``DATA_DIR`` environment variable.

Runs on Apple Silicon GPU (MPS) when available, otherwise falls back to CPU.
"""

import itertools
import json
import logging
import os
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
UNIQUE_LABELS_N = 2  # Количество классов для бинарной классификации

# выбираем устройство: GPU на Apple Silicon (MPS), при отсутствии — CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_EPOCHS = 80
BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)

# подавляем безвредное предупреждение deprecation внутри pytorch_lightning
warnings.filterwarnings(
    'ignore',
    message=r'`isinstance\(treespec, LeafSpec\)` is deprecated',
)

# сетка гиперпараметров для перебора архитектур
GRID_SEARCH = {
    "hidden_dims": [None, [128], [256, 64], [512, 128]],
    "dropout": [0.0, 0.3],
    "lr": [1e-2, 1e-3],
}


def split_dataset(source_df: pd.DataFrame, valid_train_set_size=0.3):
    y = source_df.featured_approval_content_declined

    content_ids = source_df['content_id']

    x_train, x_test_ids, y_train, y_test = (
        train_test_split(
            content_ids, y, test_size=valid_train_set_size, random_state=RANDOM_SEED, stratify=y
        )
    )

    test_df = pd.DataFrame({'content_id': x_test_ids, 'y': y_test})
    train_df = pd.DataFrame({'content_id': x_train, 'y': y_train})
    print('Positive target share %.4f' % (train_df['y'].value_counts(normalize=True)[1]))

    return train_df, test_df


def prepare_tfidf_df(content_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    # присоединяем текстовую информацию и заполняем пропуски
    ocr_dataset = (
        content_df[['content_id']]
        .merge(text_df, how='left')
    )
    ocr_dataset['text'] = ocr_dataset['text'].fillna('')

    return ocr_dataset


def prepare_tfidf(content_df, text_df, tfidf_vectorizer) -> csr_matrix:
    # вычисляем tf-idf по подготовленному датафрейму с текстами
    ocr_dataset = prepare_tfidf_df(content_df, text_df)
    ocr_features = tfidf_vectorizer.transform(ocr_dataset['text'].values)

    return ocr_features


class SparseFeaturesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(
            self.features[idx].toarray()[0]
        ).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label


class TextClassifier(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, hidden_dims=None, dropout=0.0, lr=1e-2):
        super().__init__()
        # Сохраняем гиперпараметры, чтобы они автоматически логгировались
        self.save_hyperparameters()

        # Создаем нейросеть: линейный слой или MLP с настраиваемой архитектурой
        if not hidden_dims:
            self.model = nn.Linear(vocab_size, num_classes)
        else:
            layers = []
            in_dim = vocab_size
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, num_classes))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

        # Настраиваем ReduceLROnPlateau через словарь PyTorch Lightning
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def config_key(config):
    # стабильный ключ конфигурации для сравнения с уже завершёнными
    return json.dumps(config, sort_keys=True)


def load_completed_results(results_file):
    # загружаем результаты конфигураций, завершённых в прошлых запусках
    if os.path.exists(results_file):
        with open(results_file, encoding='utf-8') as f:
            return json.load(f)
    return []


def save_results(results_file, results):
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def find_checkpoint_for_config(csv_logs_dir, config, logger):
    # ищем checkpoint незавершённой конфигурации по сохранённым гиперпараметрам
    if not os.path.isdir(csv_logs_dir):
        return None
    version_dirs = sorted(
        (d for d in os.listdir(csv_logs_dir) if d.startswith('version_')),
        key=lambda d: int(d.split('_')[-1]),
        reverse=True,
    )
    for version_dir in version_dirs:
        hparams_file = os.path.join(csv_logs_dir, version_dir, 'hparams.yaml')
        if not os.path.isfile(hparams_file):
            continue
        with open(hparams_file, encoding='utf-8') as f:
            hparams = yaml.safe_load(f) or {}
        saved = {k: hparams.get(k) for k in ('hidden_dims', 'dropout', 'lr')}
        if saved != config:
            continue
        ckpt_dir = os.path.join(csv_logs_dir, version_dir, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            continue
        ckpts = [
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')
        ]
        if not ckpts:
            continue
        latest = max(ckpts, key=os.path.getmtime)
        # проверяем целостность: прерванная запись checkpoint делает его непригодным
        try:
            if zipfile.is_zipfile(latest) and zipfile.ZipFile(latest).testzip() is None:
                return latest
        except Exception:
            pass
        logger.warning('Checkpoint повреждён, обучаем конфигурацию с нуля: %s' % latest)
        return None
    return None


def evaluate_predictions(predicted_classes, y_true, logger, prefix=''):
    # считаем метрики качества и записываем их в лог
    metrics = {
        'accuracy': accuracy_score(y_true, predicted_classes),
        'f1_score': f1_score(y_true, predicted_classes),
        'precision': precision_score(y_true, predicted_classes),
        'recall': recall_score(y_true, predicted_classes),
    }
    msg = '%saccuracy = %.4f, f1_score = %.4f, precision = %.4f, recall = %.4f' % (
        prefix, metrics['accuracy'], metrics['f1_score'], metrics['precision'], metrics['recall']
    )
    print(msg)
    logger.info(msg)
    return metrics


def main():
    root_data_dir = os.getenv('DATA_DIR')

    ocr_dataset_df = pd.read_csv(os.path.join(root_data_dir, 'ocr_dataset.zip'), compression='zip')
    print('OCR dataset columns %s' % ocr_dataset_df.columns)
    labels_dataset_df = pd.read_csv(os.path.join(root_data_dir, 'final_dataset.zip'), compression='zip')
    print('Labels dataset columns %s' % labels_dataset_df.columns)

    print(len(ocr_dataset_df))

    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        lowercase=True,
        token_pattern=r'\b[\w\d]{3,}\b'
    )

    tfidf_vectorizer.fit(ocr_dataset_df['text'].fillna('').values)

    train_df, test_df = split_dataset(labels_dataset_df)
    print('test rows %d, train rows %d' % (test_df.shape[0], train_df.shape[0]))

    train_ocr_dataset = prepare_tfidf(train_df, ocr_dataset_df, tfidf_vectorizer)
    test_ocr_dataset = prepare_tfidf(test_df, ocr_dataset_df, tfidf_vectorizer)

    print('train features shape', train_ocr_dataset.shape)

    # создаём сразу тренировочный и тестовый датасет, pd.series в качестве таргета не пойдёт, нужен numpy.array
    train_dataset = SparseFeaturesDataset(train_ocr_dataset, train_df['y'].values)
    test_dataset = SparseFeaturesDataset(test_ocr_dataset, test_df['y'].values)
    #  мы уже  знаем, что количество фичей соответствует  количеству нейронов в слое, поэтому сохраним это число в отдельную константу
    UNIQUE_WORDS_N = test_ocr_dataset.shape[1]

    log_dir = os.path.join(root_data_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Настройка логирования в файл и консоль
    log_file = os.path.join(log_dir, 'pytorch_training.log')
    RESULTS_FILE = os.path.join(log_dir, 'grid_search_results.json')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info('Используем устройство: %s' % DEVICE)

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        ),
        "valid": DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        ),
    }

    # --- Бейзлайн: логистическая регрессия из sklearn ---
    log_reg = LogisticRegression(max_iter=500).fit(train_ocr_dataset, train_df['y'])
    predicted_classes = log_reg.predict(test_ocr_dataset)
    evaluate_predictions(predicted_classes, test_df['y'], logger, prefix='[sklearn LogisticRegression] ')

    # --- Grid search по архитектурам нейросети ---
    grid_keys = sorted(GRID_SEARCH)
    grid_values = [GRID_SEARCH[k] for k in grid_keys]
    grid_results = load_completed_results(RESULTS_FILE)
    completed_keys = {config_key(r['config']) for r in grid_results}

    total_configs = int(np.prod([len(v) for v in grid_values]))
    logger.info(
        'Запуск grid search: %d конфигураций, %d уже завершено (будут пропущены)' % (
            total_configs, len(grid_results)
        )
    )

    for combo in itertools.product(*grid_values):
        config = dict(zip(grid_keys, combo))
        config_msg = 'hidden_dims=%s, dropout=%.1f, lr=%s' % (
            config['hidden_dims'], config['dropout'], config['lr']
        )
        if config_key(config) in completed_keys:
            logger.info('Конфигурация уже завершена, пропускаем: %s' % config_msg)
            continue

        logger.info('Обучение конфигурации: %s' % config_msg)

        # если конфигурация уже обучалась, но не успела завершиться — продолжаем с checkpoint
        csv_logs_dir = os.path.join(log_dir, 'csv_logs')
        ckpt_path = find_checkpoint_for_config(csv_logs_dir, config, logger)
        if ckpt_path:
            logger.info('Возобновляем конфигурацию с checkpoint: %s' % ckpt_path)

        model = TextClassifier(
            vocab_size=UNIQUE_WORDS_N,
            num_classes=UNIQUE_LABELS_N,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            lr=config['lr'],
        )

        callbacks = [
            EarlyStopping(
                monitor="valid_loss",
                patience=5,
                mode="min",
                verbose=True
            )
        ]

        csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, name="csv_logs")
        trainer = pl.Trainer(
            accelerator=DEVICE,
            devices=1,
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            default_root_dir=log_dir,
            logger=csv_logger,
            enable_checkpointing=True,
            log_every_n_steps=10,
        )

        trainer.fit(
            model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["valid"],
            ckpt_path=ckpt_path,
        )

        # оценка качества на валидационном (тестовом) сете
        prediction_batches = trainer.predict(model, dataloaders=loaders["valid"])
        logits = torch.cat([batch.detach().cpu() for batch in prediction_batches])
        predicted_classes_torch = logits.argmax(dim=1).numpy()

        metrics = evaluate_predictions(
            predicted_classes_torch, test_df['y'], logger,
            prefix='[torch %s] ' % config_msg,
        )
        grid_results.append({'config': config, 'metrics': metrics})
        # сразу сохраняем прогресс, чтобы после прерывания можно было продолжить
        save_results(RESULTS_FILE, grid_results)
        completed_keys.add(config_key(config))

    # --- Итоги grid search ---
    best = max(grid_results, key=lambda r: r['metrics']['f1_score'])
    summary_msg = (
        'Лучший конфиг по f1_score: hidden_dims=%s, dropout=%.1f, lr=%s | '
        'accuracy = %.4f, f1_score = %.4f, precision = %.4f, recall = %.4f' % (
            best['config']['hidden_dims'], best['config']['dropout'], best['config']['lr'],
            best['metrics']['accuracy'], best['metrics']['f1_score'],
            best['metrics']['precision'], best['metrics']['recall'],
        )
    )
    print(summary_msg)
    logger.info(summary_msg)

    logger.info('Grid search завершён, всего конфигураций: %d' % len(grid_results))


if __name__ == '__main__':
    main()
