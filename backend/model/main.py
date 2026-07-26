"""
主函数 - 协调整个金融风控建模流程
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import prepare_features
from model_trainer import ModelTrainer


def main():
    """主函数：执行完整的建模流程"""
    
    print("\n" + "="*70)
    print(" "*15 + "金融风控二分类模型建模")
    print("="*70 + "\n")
    
    # 设置随机种子
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    
    # 创建模型保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'./models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}\n")
    
    # ========== 步骤1: 特征工程 ==========
    print("\n" + "="*70)
    print("STEP 1: Feature Engineering")
    print("="*70 + "\n")
    
    # 生成训练集特征
    print("Processing training data...")
    df_train = prepare_features(is_train=True)
    
    # 保存特征数据
    train_feature_path = './data/train_features.csv'
    df_train.to_csv(train_feature_path, index=False)
    print(f"Training features saved to: {train_feature_path}")
    
    # 生成测试集特征
    print("\nProcessing test data...")
    df_test = prepare_features(is_train=False)
    
    # 保存特征数据
    test_feature_path = './data/test_features.csv'
    df_test.to_csv(test_feature_path, index=False)
    print(f"Test features saved to: {test_feature_path}")
    
    print("\nFeature Engineering Summary:")
    print(f"  - Training set shape: {df_train.shape}")
    print(f"  - Test set shape: {df_test.shape}")
    print(f"  - Number of features: {df_train.shape[1] - 2}")  # 减去id和label
    print(f"  - Label distribution: {df_train['label'].value_counts().to_dict()}")
    print(f"  - Label ratio (positive): {df_train['label'].mean():.4f}")
    
    # ========== 步骤2: 模型训练 ==========
    print("\n" + "="*70)
    print("STEP 2: Model Training")
    print("="*70 + "\n")
    
    # 初始化训练器
    trainer = ModelTrainer(
        output_dir=model_dir,
        eval_metric='roc_auc',
        time_limit=3600,  # 1小时训练时间
        random_state=RANDOM_STATE
    )
    
    # 划分训练集和验证集
    train_df, val_df = trainer.prepare_data(
        df_train,
        target_col='label',
        id_col='id',
        test_size=0.2
    )
    
    # 训练模型
    predictor = trainer.train(
        train_df=train_df,
        target_col='label',
        id_col='id'
    )
    
    # ========== 步骤3: 模型评估 ==========
    print("\n" + "="*70)
    print("STEP 3: Model Evaluation")
    print("="*70 + "\n")
    
    # 评估模型
    results = trainer.evaluate(
        val_df=val_df,
        target_col='label',
        id_col='id'
    )
    
    # 保存评估结果
    val_auc = results['auc']
    results_path = os.path.join(model_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Validation AUC: {val_auc:.6f}\n")
        f.write(f"\nModel Leaderboard:\n")
        f.write(results['leaderboard'].to_string())
    print(f"Evaluation results saved to: {results_path}")
    
    # ========== 步骤4: 测试集预测 ==========
    print("\n" + "="*70)
    print("STEP 4: Test Set Prediction")
    print("="*70 + "\n")
    
    # 预测测试集
    submission_path = f'./submission_{timestamp}.csv'
    submission = trainer.predict(
        test_df=df_test,
        id_col='id',
        output_path=submission_path
    )
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("MODELING COMPLETED!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"  - Validation AUC: {val_auc:.6f}")
    print(f"  - Model saved to: {model_dir}")
    print(f"  - Submission file: {submission_path}")
    print(f"\n✅ All tasks completed successfully!")
    print("="*70 + "\n")
    
    return {
        'val_auc': val_auc,
        'model_dir': model_dir,
        'submission_path': submission_path,
        'trainer': trainer
    }


def quick_predict(model_path: str, test_feature_path: str = './data/test_features.csv'):
    """
    快速预测函数（用于已训练模型的预测）
    
    Args:
        model_path: 已训练模型路径
        test_feature_path: 测试集特征文件路径
    """
    print("\n" + "="*70)
    print("Quick Prediction Mode")
    print("="*70 + "\n")
    
    # 加载测试集特征
    if not os.path.exists(test_feature_path):
        print(f"Test features not found at {test_feature_path}")
        print("Generating test features...")
        df_test = prepare_features(is_train=False)
        df_test.to_csv(test_feature_path, index=False)
    else:
        print(f"Loading test features from {test_feature_path}...")
        df_test = pd.read_csv(test_feature_path)
    
    # 加载模型
    trainer = ModelTrainer()
    trainer.load_model(model_path)
    
    # 预测
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'./submission_{timestamp}.csv'
    submission = trainer.predict(
        test_df=df_test,
        id_col='id',
        output_path=submission_path
    )
    
    print(f"\n✅ Prediction completed!")
    print(f"Submission file: {submission_path}")
    
    return submission


if __name__ == "__main__":
    # 运行主流程
    results = main()
    
    print("\nYou can also use the quick_predict() function to make predictions")
    print("with an already trained model:")
    print("  >>> from main import quick_predict")
    print("  >>> quick_predict('models/model_xxxxx')")

