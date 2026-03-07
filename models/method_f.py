import logging
import numpy as np
from catboost import CatBoostClassifier
from tqdm import tqdm

def train_predict_catboost(X, y, final_feature, model_config, lottery_config):
    """
    Trains and predicts using CatBoost model.
    Model F: CatBoost (Categorical Boosting)
    Goal: Achieve better generalization on smaller lottery datasets through unbiased gradient estimation.
    Principle: Uses Ordered Boosting to prevent prediction shift and can natively handle non-linear relationships of numbers as categorical features.
    """
    logging.info("Method F: 训练 CatBoost 分类模型...")

    TOTAL_NUMBERS = lottery_config['total_numbers']
    probs = np.zeros(TOTAL_NUMBERS)
    conf = model_config
    all_importances = []

    # Train independent binary classifiers with progress bar
    for i in tqdm(range(TOTAL_NUMBERS), desc="CatBoost 训练", leave=False):
        if len(np.unique(y[:, i])) > 1:
            # Base parameters
            params = {
                'iterations': conf.get('iterations', 500),
                'learning_rate': conf.get('learning_rate', 0.03),
                'depth': conf.get('depth', 6),
                'l2_leaf_reg': conf.get('l2_leaf_reg', 3),
                'random_seed': conf.get('random_state', 42),
                'verbose': 0,  # Suppress verbose output
                'allow_writing_files': False # Suppress creation of catboost_info dir
            }
            
            # Note: cat_features would be specified here if the feature set contained categorical columns.
            # The current feature set is purely numerical.
            
            clf = CatBoostClassifier(**params)
            clf.fit(X, y[:, i])
            probs[i] = clf.predict_proba(final_feature.reshape(1, -1))[0, 1]
            all_importances.append(clf.feature_importances_)
        else:
            probs[i] = 1.0 if y[0, i] == 1 else 0.0

    avg_importance = np.mean(all_importances, axis=0) if all_importances else np.zeros(X.shape[1])
    return probs, avg_importance