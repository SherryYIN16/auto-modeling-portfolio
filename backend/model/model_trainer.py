import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, output_dir='./models', eval_metric='roc_auc', time_limit=60):
        self.output_dir = output_dir
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.predictor = None
        
    def train(self, train_df, target_col='label', id_col='id'):
        train_data = train_df.drop(columns=[id_col])
        self.predictor = TabularPredictor(
            label=target_col,
            eval_metric=self.eval_metric,
            path=self.output_dir,
            problem_type='binary',
            verbosity=2
        )
        self.predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets='medium_quality_faster_train',
            raise_on_no_models_fitted=False
        )
        return self.predictor
    
    def evaluate(self, val_df, target_col='label', id_col='id'):
        val_data = val_df.drop(columns=[id_col])
        y_pred_proba = self.predictor.predict_proba(val_data)
        from sklearn.metrics import roc_auc_score
        y_true = val_data[target_col]
        auc_score = roc_auc_score(y_true, y_pred_proba[1])
        # ¼ÆËã¼òÒ×KSºÍAR
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[1])
        ks_score = max(tpr - fpr)

        # AR: 2 * AUC - 1
        ar_score = 2 * auc_score - 1

        return {'auc': auc_score, 'ks': ks_score, 'ar': ar_score}
