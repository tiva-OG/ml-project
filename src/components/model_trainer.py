import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_training(self, train_array, test_array):
        try:
            logging.info("Split train and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Linear Regressor": LinearRegression(),
                "RandomForest Regressor": RandomForestRegressor(),
                "XGBoost Regressor": XGBRegressor(),
            }

            models_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # get best model and score
            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("no best model found", sys)

            logging.info(
                f"Best model found on both train and test data: {best_model_name}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file, obj=best_model
            )

            y_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_pred)

            return test_score

        except Exception as e:
            pass
