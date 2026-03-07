import logging
import numpy as np
import xgboost as xgb
from tqdm import tqdm

def train_predict_xgb(X, y, final_feature, model_config, lottery_config):
    logging.info("Method C: 训练 XGBoost 分类模型...")

    TOTAL_NUMBERS = lottery_config['total_numbers']
    probs = np.zeros(TOTAL_NUMBERS)
    conf = model_config
    all_importances = []
    # Train independent binary classifiers with progress bar
    for i in tqdm(range(TOTAL_NUMBERS), desc="XGBoost 训练", leave=False):
        if len(np.unique(y[:, i])) > 1:
            clf = xgb.XGBClassifier(
                n_estimators=conf['n_estimators'],
                max_depth=conf['max_depth'],
                learning_rate=conf['learning_rate'],
                subsample=conf.get('subsample', 1.0),
                colsample_bytree=conf.get('colsample_bytree', 1.0),
                gamma=conf.get('gamma', 0),
                scale_pos_weight=conf.get('scale_pos_weight', 1),
                random_state=conf['random_state'],
                n_jobs=conf['n_jobs'],
                eval_metric=conf['eval_metric']
            )
            clf.fit(X, y[:, i])
            probs[i] = clf.predict_proba(final_feature.reshape(1, -1))[0, 1]
            all_importances.append(clf.feature_importances_)
        else:
            probs[i] = 1.0 if y[0, i] == 1 else 0.0

    avg_importance = np.mean(all_importances, axis=0) if all_importances else np.zeros(X.shape[1])
    return probs, avg_importance