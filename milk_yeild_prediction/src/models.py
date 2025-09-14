from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_stacked_model():
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=5,
                             reg_lambda=2, reg_alpha=1, random_state=42, n_jobs=-1)),
        ("lgbm", LGBMRegressor(n_estimators=120, learning_rate=0.05, max_depth=5,
                               reg_lambda=2, random_state=42, n_jobs=-1)),
        ("cat", CatBoostRegressor(verbose=0, n_estimators=120, learning_rate=0.05,
                                  depth=5, l2_leaf_reg=3, random_state=42))
    ]
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        passthrough=False,
        n_jobs=-1
    )
    return stacked_model
