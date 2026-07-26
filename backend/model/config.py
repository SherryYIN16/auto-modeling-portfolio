"""
配置文件 - 模型训练参数
"""

# 数据路径配置
DATA_CONFIG = {
    'train_main': 'data/train/train.csv',
    'train_flow': 'data/train/train_bank_statement.csv',
    'test_main': 'data/test/testaa.csv',
    'test_flow': 'data/test/testaa_bank_statement.csv',
}

# 模型训练配置
MODEL_CONFIG = {
    'random_state': 42,
    'eval_metric': 'roc_auc',
    'time_limit': 3600,  # 训练时间限制（秒）
    'num_bag_folds': 5,  # Bagging折数
    'num_stack_levels': 2,  # Stacking层数
}

# 特征工程配置
FEATURE_CONFIG = {
    'use_static_features': True,
    'use_flow_features': True,
    'use_interaction_features': True,
}

print("Configuration loaded successfully!")

