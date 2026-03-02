import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os 
from sklearn.compose  import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomeException
from src.logger import logging
from src.utilis import save_obj

class DataTransformationConfig:
    Preprocessor_obj_file_path = os.path.join('artifacts',"Preprocessor.pkl")

class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformer_object(self):
        try:
            numercal_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num",num_pipeline, numercal_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_transformer_object()

            Target_column_name = "math_score" 
            numercal_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[Target_column_name],axis=1)
            target_feature_train_df = train_df[Target_column_name]

            
            input_feature_test_df = test_df.drop(columns=[Target_column_name],axis=1)
            target_feature_test_df = test_df[Target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_obj(
                file_path = self.data_transformation_config.Preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.Preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomeException(e,sys)