  # Modeling Configuration
CONFIG = {
    'target': 'label',
    'data_path': 'backend/data\3c7d8497-1018-4194-82de-b1e6637ed4d7_train.csv',
    'model_type': 'xgb',
    'model_params': {
        'xgb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
        'lgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31},
        'lr': {'C': 1.0, 'solver': 'liblinear'}
    },
    'data_cleaning': {
        'missing_threshold': 0.5,
        'unique_threshold': 0.95,
        'corr_threshold': 0.9
    }
}

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def calculate_iv(df, target):
    iv_dict = {}
    for col in df.columns:
        if col == target or df[col].dtype == 'object': continue
        try:
            temp_df = pd.DataFrame({'col': df[col], 'target': df[target]})
            temp_df['bin'] = pd.qcut(temp_df['col'], q=10, duplicates='drop')
            stats = temp_df.groupby('bin')['target'].agg(['count', 'sum'])
            stats['bad'] = stats['sum'];
            stats['good'] = stats['count'] - stats['sum']
            total_good = stats['good'].sum();
            total_bad = stats['bad'].sum()
            stats['good_pct'] = stats['good'] / total_good
            stats['bad_pct'] = stats['bad'] / total_bad
            stats['woe'] = np.log(stats['good_pct'] / stats['bad_pct'].replace(0, 0.0001))
            stats['iv'] = (stats['good_pct'] - stats['bad_pct']) * stats['woe']
            iv_dict[col] = stats['iv'].sum()
        except:
            iv_dict[col] = 0
    return pd.Series(iv_dict).sort_values(ascending=False)


def data_cleaning(df, target):
    print("Running automated data cleaning...")
    # 1. Missing values
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > CONFIG['data_cleaning']['missing_threshold']].index
    df = df.drop(columns=drop_cols)

    # 2. Numerical & Categorical Handling
    for col in df.columns:
        if col == target: continue
        if df[col].dtype == 'object':
            # Categorical: Fill missing with mode, then LabelEncode
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            # Numerical: Fill missing with median, then Scale
            df[col] = df[col].fillna(df[col].median())

    # 3. High Unique Columns
    for col in df.columns:
        if col != target and df[col].nunique() / len(df) > CONFIG['data_cleaning']['unique_threshold']:
            df = df.drop(columns=[col])

    # 4. Correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_corr = [column for column in upper.columns if any(upper[column] > CONFIG['data_cleaning']['corr_threshold'])]
    df = df.drop(columns=drop_corr)

    return df


def get_model():
    m_type = CONFIG['model_type']
    params = CONFIG['model_params'].get(m_type, {})
    if m_type == 'xgb': return XGBClassifier(**params)
    if m_type == 'lgbm': return LGBMClassifier(**params)
    return LogisticRegression(**params)


def run_pipeline():
    data = pd.read_csv(CONFIG['data_path'])
    data = data_cleaning(data, CONFIG['target'])

    print("Calculating IV...")
    iv_values = calculate_iv(data, CONFIG['target'])
    print(f"\nIV Summary (Top 10):\n{iv_values.head(10)}")

    X = data.drop(columns=[CONFIG['target']])
    y = data[CONFIG['target']]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nStarting {CONFIG['model_type']} training...")
    model = get_model()
    model.fit(train_x, train_y)

    y_proba = model.predict_proba(test_x)[:, 1]
    auc = roc_auc_score(test_y, y_proba)
    print(f"\nEvaluation Results:\nAUC: {auc:.4f}")

    importance = pd.DataFrame({'feature': train_x.columns,
                               'importance': getattr(model, 'feature_importances_', np.zeros(len(train_x.columns)))})
    print(
        f"\nTop 10 Feature Importance:\n{importance.sort_values(by='importance', ascending=False).head(10).to_string(index=False)}")


if __name__ == '__main__':
    run_pipeline()
