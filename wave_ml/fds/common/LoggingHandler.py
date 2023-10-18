import logging
from logging import Logger, Formatter


class LoggingHandler:
    log: Logger
    formatter: Formatter
    levels: dict

    def __init__(self, name, mode, level):
        self.log = logging.getLogger(name)
        # self.log.propagate = True
        self.formatter = logging.Formatter("%(asctime)s %(lineno)d [%(levelname)s] : %(message)s", "%Y-%m-%d %H:%M:%S")
        self.levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL}

        file_handler = logging.FileHandler(f"{name}.log", mode=mode, encoding="UTF-8")
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)
        self.log.setLevel(self.levels.get(level))

    def get_log(self):
        return self.log
