from dataclasses import dataclass
import sys

import joblib
from sklearn.ensemble import GradientBoostingRegressor


def _register_legacy_aliases():
    """Expose this module under historical import paths for joblib compatibility."""
    module = sys.modules[__name__]
    aliases = set()
    if __name__ == "va_regression.model":
        aliases.update(
            {
                "src.va_regression.model",
                "postpreprocess.va_regression.model",
            }
        )
    elif __name__ == "postpreprocess.va_regression.model":
        aliases.update(
            {
                "src.va_regression.model",
                "va_regression.model",
            }
        )
    elif __name__ == "src.va_regression.model":
        aliases.update(
            {
                "va_regression.model",
                "postpreprocess.va_regression.model",
            }
        )
    for alias in aliases:
        sys.modules.setdefault(alias, module)


_register_legacy_aliases()


@dataclass
class VARegressor:
    valence_model: GradientBoostingRegressor
    arousal_model: GradientBoostingRegressor

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
