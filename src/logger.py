import logging
from datetime import datetime
import os

LOGFILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


logs_path = os.path.join(os.getcwd(),"log",LOGFILE)

os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path,LOGFILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s -%(message)s",
    level=logging.INFO


)


