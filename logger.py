import logging

def setup_logger(filepath):
    #创建logger文件
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)