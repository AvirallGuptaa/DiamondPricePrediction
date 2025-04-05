# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: e:\mlprojectspw\diamondpriceprediction2\src\components\model_trainer.py
# Bytecode version: 3.8.0rc1+ (3413)
# Source timestamp: 2023-05-07 08:37:09 UTC (1683448629)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1])
            models = {
                'LinearRegression': LinearRegression(), 
                'Lasso': Lasso(), 
                'Ridge': Ridge(), 
                'Elasticnet': ElasticNet(), 
                'DecisionTree': DecisionTreeRegressor()
                }
            
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')

            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)

# import os,sys

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
# from sklearn.tree import DecisionTreeRegressor

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object
# from src.utils import evaluate_model

# from dataclasses import dataclass

# @dataclass
# class Model_trainer_config:
#     train_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=Model_trainer_config()
    
#     def initate_model_training(self,train_array,test_array):
#         try:
                
#             logging.info("Splitting dependent and independent variables")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             models={
#                 "LinearRegression":LinearRegression(),
#                 "Lasso":Lasso(),
#                 "Ridge":Ridge(),
#                 "ElasticNet":ElasticNet(),
#                 "DecsisionTreeRegressor":DecisionTreeRegressor()
#             }

#             model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
#             print(model_report)
#             print("\n ======================================================================================================")
#             logging.info(f"Model report:{model_report}")

#             best_model_score=max(sorted(model_report.values()))
#             best_model_name=list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model=models[best_model_name]
#             print(f"Best model found {best_model} with r2 score {best_model_score}")
#             print("\n =============================================================================")
#             logging.info(f"Best model found {best_model} with r2 score {best_model_score}")
#             save_object(
#                 file_path=self.model_trainer_obj.train_model_file_path,
#                 obj=best_model
#             )
#         except Exception as e:
#             raise CustomException(e,sys)

