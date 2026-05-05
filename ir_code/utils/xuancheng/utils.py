import logging
import arrow
import os
import time
from datetime import datetime

class Logger():
    def __init__(self, web_brand):
        today = arrow.now().format('YYYY-MM-DD')
        # 设置logging模块的默认编码为UTF-8
        logging.basicConfig()
        self.logger = logging.getLogger(web_brand)
        self.logger.setLevel(logging.DEBUG)
        self.Prevent_duplicate()
        os.makedirs(f'E:\\python_output\\{web_brand}', exist_ok=True)
        self.filehandler = logging.FileHandler('E:\\python_output\\{}\\{}.log'.format(web_brand, today))
        self.filehandler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.filehandler.setFormatter(formatter)
        self.logger.addHandler(self.filehandler)

    def Prevent_duplicate(self):
        if self.logger.handlers:
            self.logger.removeHandler(self.logger.handlers[0])

if __name__ == '__main__':
    logger = Logger('s_xc_data').logger
    Start_Time = time.time()
    End_Time = time.time()
    logger.info('任务用时{}'.format(arrow.get(datetime.fromtimestamp(End_Time - Start_Time)).format('mm:ss')))