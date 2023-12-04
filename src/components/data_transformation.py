import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

        self.num_features = ["writing_score", "reading_score"]
        self.cat_features = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

    def get_transformer_obj(self):
        """
        This function is responsible for building the data transformer object (preprocessor)

        Returns:
            preprocess: transformer object for manipulating data
        """

        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler()),
                ]
            )

            # logging.info("Numerical features' standard-scaling completed")
            # logging.info("Categorical features' one-hot encoding completed")
            logging.info(f"Numerical features: {self.num_features}")
            logging.info(f"Categorical features: {self.cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, self.num_features),
                    ("categorical_pipeline", cat_pipeline, self.cat_features),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loading train and test data completed")
            logging.info("Obtaining data transformer")

            preprocessor = self.get_transformer_obj()

            target_column = "math_score"

            input_train_df = train_df.drop(columns=[target_column], axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column], axis=1)
            target_test_df = test_df[target_column]

            logging.info("Applying preprocessor on train and test data")

            input_train_arr = preprocessor.fit_transform(input_train_df)
            input_test_arr = preprocessor.transform(input_test_df)

            train_arr = np.c_[(input_train_arr, np.array(target_train_df))]
            test_arr = np.c_[(input_test_arr, np.array(target_test_df))]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except:
            pass
