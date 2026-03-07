import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_predict_rf(X, y, final_feature, model_config, lottery_config):
    logging.info("Method B: 训练 RandomForest 分类模型...")

    TOTAL_NUMBERS = lottery_config['total_numbers']

    # Parameters from Centralized CONFIG
    conf = model_config
    model = RandomForestClassifier(
        n_estimators=conf['n_estimators'],
        max_depth=conf.get('max_depth'),
        min_samples_split=conf.get('min_samples_split', 2),
        min_samples_leaf=conf.get('min_samples_leaf', 1),
        max_features=conf.get('max_features', 'sqrt'),
        class_weight='balanced_subsample',
        random_state=conf['random_state'],
        n_jobs=conf['n_jobs']
    )
    model.fit(X, y)

    # Predict probabilities for final_feature
    raw_probs = model.predict_proba(final_feature.reshape(1, -1))
    probs = np.zeros(TOTAL_NUMBERS)

    # Robust probability extraction (handling potential missing classes '0' or '1' in training)
    for i in range(TOTAL_NUMBERS):
        if hasattr(raw_probs[i], "shape") and raw_probs[i].shape[1] == 2:
            probs[i] = raw_probs[i][0, 1]
        elif hasattr(raw_probs[i], "shape") and raw_probs[i].shape[1] == 1:
            # If only one class exists in training data for this number
            if model.classes_[i][0] == 1:
                probs[i] = 1.0
            else:
                probs[i] = 0.0

    return probs, model.feature_importances_