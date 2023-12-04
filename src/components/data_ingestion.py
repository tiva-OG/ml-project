import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "data.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path))
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train-test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_pth, test_data_pth = obj.initiate_ingestion()
    
    transformer = DataTransformation()
    transformer.initiate_transformation(train_data_pth, test_data_pth)
