import os
import sys
import numpy as np
import pandas as pd 

import dill

from src.exception import CustomeException
def save_obj(file_path, obj):
    try:
        dit_path = os.path.dirname(file_path)
        os.makedirs(dit_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e,sys)