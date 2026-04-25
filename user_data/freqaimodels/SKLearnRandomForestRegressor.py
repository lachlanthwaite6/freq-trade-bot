import logging
from typing import Any

from sklearn.ensemble import RandomForestRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class SKLearnRandomForestRegressor(BaseRegressionModel):
    """
    Scikit-learn RandomForest regressor for continuous FreqAI targets.

    This matches strategies that set numeric targets such as "&-s_close".
    It avoids LightGBM/XGBoost's OpenMP runtime dependency on macOS.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()

        if y.shape[1] == 1:
            y = y[:, 0]

        train_weights = data_dictionary["train_weights"]

        model = RandomForestRegressor(**self.model_training_parameters)
        model.fit(X=X, y=y, sample_weight=train_weights)

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            test_features = data_dictionary["test_features"].to_numpy()
            test_labels = data_dictionary["test_labels"].to_numpy()
            if test_labels.shape[1] == 1:
                test_labels = test_labels[:, 0]
            logger.info("Score: %s", model.score(test_features, test_labels))

        return model
