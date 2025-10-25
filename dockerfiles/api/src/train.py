import logging
import sys

log_file_path = '/srv/data/pipelines-data/service.log'


def setup_logger(log_file=log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)-24s :%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    print()
    logger = setup_logger()

    logger.info('You have run train script づ｡◕‿‿◕｡)づ')
    logger.info('Training process started')

    # Your training code here
    # Example:
    # logger.info('Loading training data')
    # logger.info('Model initialized: %s', model)
    # logger.info('Training completed')