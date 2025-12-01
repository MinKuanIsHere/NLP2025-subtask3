from dataclasses import dataclass
import joblib
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class VARegressor:
    valence_model: GradientBoostingRegressor
    arousal_model: GradientBoostingRegressor

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
