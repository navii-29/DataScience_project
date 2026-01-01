import pandas as pd
import os
import sys
import numpy as np
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from src.utils import evaluation,save_object    
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score










@dataclass
class model_trainer_config:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class Model_trainer:
    def __init__(self):
        self.model_trainer_path = model_trainer_config()


    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('split training and test set')

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'RF': RandomForestRegressor(),
                'ADB': AdaBoostRegressor(),

                'LR' : LinearRegression(),
                'Ridge': Ridge(),
                'GB': GradientBoostingRegressor()


            }

            model_report:dict = evaluation(X_train=X_train,X_test= X_test,y_train= y_train,
                                           y_test = y_test,models = models)
            
            ## To get best model
            best_model_score = max(sorted(model_report.values()))

            ## to get best model by name

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]


            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info('best model of traing and testing')

            save_object(
                file_path=model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted)
            return score


        except Exception as e:
            raise CustomException(e,sys)
        




