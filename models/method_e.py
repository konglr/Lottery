import logging
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

def train_predict_lgbm(X, y, final_feature, model_config, lottery_config):
    """
    Trains and predicts using LightGBM model.
    Model E: LightGBM (Light Gradient Boosting Machine)
    Goal: Significantly improve training speed for high-dimensional lotteries (like KL8) while maintaining accuracy.
    Principle: Uses Gradient-based One-Side Sampling (GOSS) to filter low-gradient samples and Exclusive Feature Bundling (EFB) to reduce feature dimensions.
    """
    logging.info("Method E: 训练 LightGBM 分类模型...")

    TOTAL_NUMBERS = lottery_config['total_numbers']
    probs = np.zeros(TOTAL_NUMBERS)
    conf = model_config
    all_importances = []

    # Train independent binary classifiers with progress bar
    for i in tqdm(range(TOTAL_NUMBERS), desc="LightGBM 训练", leave=False):
        if len(np.unique(y[:, i])) > 1:
            # Base parameters
            params = {
                'n_estimators': conf.get('n_estimators', 400),
                'num_leaves': conf.get('num_leaves', 31),
                'learning_rate': conf.get('learning_rate', 0.05),
                'random_state': conf.get('random_state', 42),
                'n_jobs': conf.get('n_jobs', -1),
                'verbose': -1, # Suppress verbose output
            }
            # High-dimension specific params
            if 'feature_fraction' in conf:
                params['feature_fraction'] = conf['feature_fraction']
            if 'bagging_fraction' in conf:
                params['bagging_fraction'] = conf['bagging_fraction']
                params['bagging_freq'] = 1 # Must be > 0 for bagging
            # Imbalance specific params
            if 'is_unbalance' in conf:
                params['is_unbalance'] = conf['is_unbalance']
            if 'scale_pos_weight' in conf:
                params['scale_pos_weight'] = conf['scale_pos_weight']

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X, y[:, i])
            probs[i] = clf.predict_proba(final_feature.reshape(1, -1))[0, 1]
            all_importances.append(clf.feature_importances_)
        else:
            probs[i] = 1.0 if y[0, i] == 1 else 0.0

    avg_importance = np.mean(all_importances, axis=0) if all_importances else np.zeros(X.shape[1])
    return probs, avg_importance