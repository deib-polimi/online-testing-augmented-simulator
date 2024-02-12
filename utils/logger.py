import logging
import sys

logging.getLogger("matplotlib").setLevel(logging.WARNING)

class CustomLogger:
    def __init__(self, logger_prefix: str):
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=logging.INFO)

            # FIXME: it seems that it is not needed to stream log to sdtout
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)
            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
