import logging
import logging.handlers
import numpy as np


# Define Logger
class Logger:
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                                       encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def find_loca(locate, shapes):
    dimension = len(shapes)
    res = [0 for _ in range(dimension)]
    res[-1] = locate
    for i in range(len(shapes)):
        loca = len(shapes) - 1 - i
        if res[loca] // shapes[loca] > 0:
            tmp = res[loca]
            res[loca] = tmp % shapes[loca]
            res[loca-1] = tmp // shapes[loca]
        else:
            break
    return res


def argmin_nd(arr):
    shapes = arr.shape
    locate = np.nanargmin(arr)
    return find_loca(locate, shapes)


def argmax_nd(arr):
    shapes = arr.shape
    locate = np.nanargmax(arr)
    return find_loca(locate, shapes)


if __name__ == '__main__':
    pass
