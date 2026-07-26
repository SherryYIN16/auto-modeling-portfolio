"""
特征工程模块
包含静态特征处理和交易流水特征构造
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程类，负责所有特征的构造和转换"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.static_features = []
        self.flow_features = []
        
    def load_data(self, 
                  main_path: str, 
                  flow_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载主表和交易流水数据
        
        Args:
            main_path: 主表文件路径
            flow_path: 交易流水文件路径
            
        Returns:
            主表数据和交易流水数据
        """
        print(f"Loading main data from {main_path}...")
        df_main = pd.read_csv(main_path)
        print(f"Main data shape: {df_main.shape}")
        
        print(f"Loading bank statement from {flow_path}...")
        df_flow = pd.read_csv(flow_path)
        print(f"Bank statement shape: {df_flow.shape}")
        
        return df_main, df_flow
    
    def create_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建静态特征
        
        Args:
            df: 主表数据
            
        Returns:
            增强后的数据框
        """
        print("Creating static features...")
        df = df.copy()
        
        # 基础特征
        # 1. 贷款相关特征
        df['loan_to_limit'] = df['loan'] / (df['balance_limit'] + 1)
        df['balance_to_limit'] = df['balance'] / (df['balance_limit'] + 1)
        df['loan_to_balance'] = df['loan'] / (df['balance'] + 1)
        
        # 2. 时间相关特征
        df['history_years'] = df['history_time'] / (365 * 24 * 3600)
        df['record_to_issue'] = (df['issue_time'] - df['record_time']) / (24 * 3600)
        df['history_to_issue'] = (df['issue_time'] - df['history_time']) / (24 * 3600)
        
        # 3. 账户相关特征
        df['balance_accounts_ratio'] = df['balance_accounts'] / (df['total_accounts'] + 1)
        df['avg_balance_per_account'] = df['balance'] / (df['balance_accounts'] + 1)
        df['avg_limit_per_account'] = df['balance_limit'] / (df['balance_accounts'] + 1)
        
        # 4. 贷款压力指标
        df['loan_per_year'] = df['loan'] / (df['history_years'] + 1)
        df['monthly_payment'] = df['loan'] / df['term']
        df['payment_to_balance'] = df['monthly_payment'] / (df['balance'] + 1)
        df['payment_to_limit'] = df['monthly_payment'] / (df['balance_limit'] + 1)
        
        # 5. 利率相关特征
        df['total_interest'] = df['loan'] * df['interest_rate'] * df['term'] / (100 * 12)
        df['total_repayment'] = df['loan'] + df['total_interest']
        df['monthly_repayment'] = df['total_repayment'] / df['term']
        
        # 6. 交叉特征：等级编码
        # 将level转换为数值
        level_map = {
            'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
            'B0': 6, 'B1': 7, 'B2': 8, 'B3': 9, 'B4': 10, 'B5': 11,
            'C0': 12, 'C1': 13, 'C2': 14, 'C3': 15, 'C4': 16, 'C5': 17,
            'D0': 18, 'D1': 19, 'D2': 20, 'D3': 21, 'D4': 22, 'D5': 23,
            'E0': 24, 'E1': 25, 'E2': 26, 'E3': 27, 'E4': 28, 'E5': 29,
        }
        df['level_encoded'] = df['level'].map(level_map).fillna(-1)
        
        # 7. 风险相关交叉特征
        df['high_risk_loan'] = ((df['loan'] > df['loan'].quantile(0.75)) & 
                                (df['interest_rate'] > df['interest_rate'].quantile(0.75))).astype(int)
        df['low_balance_high_loan'] = ((df['balance'] < df['balance'].quantile(0.25)) & 
                                       (df['loan'] > df['loan'].quantile(0.75))).astype(int)
        
        # 8. 分组统计特征（基于非label列）
        # 按career分组
        if 'career' in df.columns:
            career_stats = df.groupby('career')['loan'].agg(['mean', 'std', 'median']).reset_index()
            career_stats.columns = ['career', 'career_loan_mean', 'career_loan_std', 'career_loan_median']
            df = df.merge(career_stats, on='career', how='left')
            df['loan_vs_career_mean'] = df['loan'] / (df['career_loan_mean'] + 1)
        
        # 按term分组
        term_stats = df.groupby('term')['interest_rate'].agg(['mean', 'std']).reset_index()
        term_stats.columns = ['term', 'term_rate_mean', 'term_rate_std']
        df = df.merge(term_stats, on='term', how='left')
        df['rate_vs_term_mean'] = df['interest_rate'] / (df['term_rate_mean'] + 1)
        
        # 按level分组
        level_stats = df.groupby('level')['loan'].agg(['mean', 'median']).reset_index()
        level_stats.columns = ['level', 'level_loan_mean', 'level_loan_median']
        df = df.merge(level_stats, on='level', how='left')
        
        # 9. 多项式特征（高阶特征）
        df['loan_squared'] = df['loan'] ** 2
        df['interest_rate_squared'] = df['interest_rate'] ** 2
        df['loan_times_rate'] = df['loan'] * df['interest_rate']
        df['loan_times_term'] = df['loan'] * df['term']
        
        # 10. 缺失值标记
        df['career_missing'] = df['career'].isna().astype(int)
        df['balance_limit_missing'] = df['balance_limit'].isna().astype(int)
        
        # 11. 基于Top特征的深度特征工程
        # 账户相关深度特征
        df['balance_accounts_squared'] = df['balance_accounts'] ** 2
        df['balance_per_total_account'] = df['balance'] / (df['total_accounts'] + 1)
        df['limit_per_total_account'] = df['balance_limit'] / (df['total_accounts'] + 1)
        df['accounts_utilization'] = (df['balance_accounts'] * df['balance']) / (df['total_accounts'] * df['balance_limit'] + 1)
        df['non_balance_accounts'] = df['total_accounts'] - df['balance_accounts']
        df['non_balance_ratio'] = df['non_balance_accounts'] / (df['total_accounts'] + 1)
        
        # 时间特征深度挖掘
        df['issue_time_normalized'] = (df['issue_time'] - df['issue_time'].min()) / (df['issue_time'].max() - df['issue_time'].min() + 1)
        df['record_time_normalized'] = (df['record_time'] - df['record_time'].min()) / (df['record_time'].max() - df['record_time'].min() + 1)
        df['time_gap_ratio'] = df['record_to_issue'] / (df['history_to_issue'] + 1)
        df['history_utilization'] = df['history_years'] / (df['history_years'].max() + 1)
        
        # term相关特征
        df['term_squared'] = df['term'] ** 2
        df['loan_per_term'] = df['loan'] / df['term']
        df['interest_times_term'] = df['interest_rate'] * df['term']
        df['payment_burden'] = (df['monthly_payment'] * df['term']) / (df['balance_limit'] + 1)
        
        # 利息相关深度特征
        df['total_interest_ratio'] = df['total_interest'] / (df['loan'] + 1)
        df['interest_per_month'] = df['total_interest'] / df['term']
        df['interest_to_balance'] = df['total_interest'] / (df['balance'] + 1)
        df['interest_to_limit'] = df['total_interest'] / (df['balance_limit'] + 1)
        
        # residence相关特征
        df['residence_loan'] = df['residence'] * df['loan']
        df['residence_rate'] = df['residence'] * df['interest_rate']
        df['residence_term'] = df['residence'] * df['term']
        
        # 12. 风险组合特征
        df['high_term_high_rate'] = ((df['term'] >= 60) & (df['interest_rate'] > df['interest_rate'].quantile(0.75))).astype(int)
        df['low_balance_accounts'] = (df['balance_accounts'] < df['balance_accounts'].quantile(0.25)).astype(int)
        df['high_payment_burden'] = (df['payment_to_balance'] > df['payment_to_balance'].quantile(0.75)).astype(int)
        df['overutilized_credit'] = (df['balance_to_limit'] > 0.8).astype(int)
        df['minimal_history'] = (df['history_years'] < 5).astype(int)
        
        # 13. 多维度交叉特征（基于重要特征）
        df['balance_accounts_times_term'] = df['balance_accounts'] * df['term']
        df['balance_accounts_times_rate'] = df['balance_accounts'] * df['interest_rate']
        df['total_accounts_times_loan'] = df['total_accounts'] * df['loan']
        df['balance_times_term'] = df['balance'] * df['term']
        df['limit_times_term'] = df['balance_limit'] * df['term']
        
        # 14. 分组排名特征（相对位置特征）
        for col in ['balance_accounts', 'loan', 'interest_rate', 'balance', 'balance_limit']:
            if col in df.columns:
                df[f'{col}_rank_pct'] = df[col].rank(pct=True)
        
        # 15. 离散化特征
        df['loan_bin'] = pd.qcut(df['loan'], q=10, labels=False, duplicates='drop')
        df['balance_accounts_bin'] = pd.cut(df['balance_accounts'], bins=5, labels=False)
        df['term_36_60'] = (df['term'] == 36).astype(int)  # term只有36和60两种
        
        print(f"Static features created. Shape: {df.shape}")
        return df
    
    def create_flow_features(self, df_main: pd.DataFrame, 
                            df_flow: pd.DataFrame) -> pd.DataFrame:
        """
        创建交易流水特征
        
        Args:
            df_main: 主表数据
            df_flow: 交易流水数据
            
        Returns:
            包含流水特征的数据框
        """
        print("Creating bank statement features...")
        
        # 基础聚合统计
        flow_agg = df_flow.groupby('id').agg({
            'amount': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
            'direction': ['sum', 'mean'],
            'time': ['min', 'max', 'std']
        }).reset_index()
        
        # 扁平化列名
        flow_agg.columns = ['id', 
                           'flow_count', 'flow_amount_sum', 'flow_amount_mean', 
                           'flow_amount_std', 'flow_amount_min', 'flow_amount_max', 
                           'flow_amount_median',
                           'flow_direction_sum', 'flow_direction_mean',
                           'flow_time_min', 'flow_time_max', 'flow_time_std']
        
        # 计算时间跨度
        flow_agg['flow_time_span'] = flow_agg['flow_time_max'] - flow_agg['flow_time_min']
        flow_agg['flow_time_span_days'] = flow_agg['flow_time_span'] / (24 * 3600)
        
        # 计算频率
        flow_agg['flow_freq'] = flow_agg['flow_count'] / (flow_agg['flow_time_span_days'] + 1)
        
        # 按方向分别统计
        flow_in = df_flow[df_flow['direction'] == 0].groupby('id').agg({
            'amount': ['count', 'sum', 'mean', 'max']
        }).reset_index()
        flow_in.columns = ['id', 'flow_in_count', 'flow_in_sum', 'flow_in_mean', 'flow_in_max']
        
        flow_out = df_flow[df_flow['direction'] == 1].groupby('id').agg({
            'amount': ['count', 'sum', 'mean', 'max']
        }).reset_index()
        flow_out.columns = ['id', 'flow_out_count', 'flow_out_sum', 'flow_out_mean', 'flow_out_max']
        
        # 合并方向特征
        flow_agg = flow_agg.merge(flow_in, on='id', how='left')
        flow_agg = flow_agg.merge(flow_out, on='id', how='left')
        
        # 填充缺失值（没有该方向交易的用户）
        for col in ['flow_in_count', 'flow_in_sum', 'flow_in_mean', 'flow_in_max']:
            flow_agg[col] = flow_agg[col].fillna(0)
        for col in ['flow_out_count', 'flow_out_sum', 'flow_out_mean', 'flow_out_max']:
            flow_agg[col] = flow_agg[col].fillna(0)
        
        # 收支比特征
        flow_agg['in_out_ratio'] = flow_agg['flow_in_sum'] / (flow_agg['flow_out_sum'] + 1)
        flow_agg['in_out_diff'] = flow_agg['flow_in_sum'] - flow_agg['flow_out_sum']
        flow_agg['net_flow_per_day'] = flow_agg['in_out_diff'] / (flow_agg['flow_time_span_days'] + 1)
        
        # 交易金额波动性
        flow_agg['amount_cv'] = flow_agg['flow_amount_std'] / (flow_agg['flow_amount_mean'] + 1)
        flow_agg['amount_range'] = flow_agg['flow_amount_max'] - flow_agg['flow_amount_min']
        
        # 大额交易特征
        amount_75 = df_flow.groupby('id')['amount'].quantile(0.75).reset_index()
        amount_75.columns = ['id', 'amount_q75']
        flow_agg = flow_agg.merge(amount_75, on='id', how='left')
        
        amount_25 = df_flow.groupby('id')['amount'].quantile(0.25).reset_index()
        amount_25.columns = ['id', 'amount_q25']
        flow_agg = flow_agg.merge(amount_25, on='id', how='left')
        
        flow_agg['amount_iqr'] = flow_agg['amount_q75'] - flow_agg['amount_q25']
        
        # 序列特征：时间序列行为
        # 按时间排序，计算最近N笔交易的统计
        df_flow_sorted = df_flow.sort_values(['id', 'time'])
        
        # 最近10笔交易
        recent_10 = df_flow_sorted.groupby('id').tail(10).groupby('id').agg({
            'amount': ['mean', 'sum', 'std'],
            'direction': ['mean']
        }).reset_index()
        recent_10.columns = ['id', 'recent10_amount_mean', 'recent10_amount_sum', 
                            'recent10_amount_std', 'recent10_direction_mean']
        flow_agg = flow_agg.merge(recent_10, on='id', how='left')
        
        # 最早10笔交易
        early_10 = df_flow_sorted.groupby('id').head(10).groupby('id').agg({
            'amount': ['mean', 'sum'],
        }).reset_index()
        early_10.columns = ['id', 'early10_amount_mean', 'early10_amount_sum']
        flow_agg = flow_agg.merge(early_10, on='id', how='left')
        
        # 行为变化：最近vs最早
        flow_agg['amount_change'] = flow_agg['recent10_amount_mean'] / (flow_agg['early10_amount_mean'] + 1)
        
        # 周期性特征：按天统计
        df_flow['day_of_week'] = pd.to_datetime(df_flow['time'], unit='s').dt.dayofweek
        day_stats = df_flow.groupby(['id', 'day_of_week']).size().groupby('id').agg(['mean', 'std']).reset_index()
        day_stats.columns = ['id', 'day_freq_mean', 'day_freq_std']
        flow_agg = flow_agg.merge(day_stats, on='id', how='left')
        
        # 增强的交易流水特征（基于Top特征）
        # flow_amount_min很重要 - 增加更多最小值相关特征
        flow_agg['amount_min_ratio'] = flow_agg['flow_amount_min'] / (flow_agg['flow_amount_mean'] + 1)
        flow_agg['amount_min_to_max'] = flow_agg['flow_amount_min'] / (flow_agg['flow_amount_max'] + 1)
        
        # flow_direction_mean很重要 - 增加方向相关特征
        flow_agg['direction_imbalance'] = abs(flow_agg['flow_direction_mean'] - 0.5)
        flow_agg['mostly_out'] = (flow_agg['flow_direction_mean'] > 0.7).astype(int)
        flow_agg['mostly_in'] = (flow_agg['flow_direction_mean'] < 0.3).astype(int)
        
        # flow_out相关深度特征
        flow_agg['out_to_in_ratio'] = flow_agg['flow_out_mean'] / (flow_agg['flow_in_mean'] + 1)
        flow_agg['out_max_ratio'] = flow_agg['flow_out_max'] / (flow_agg['flow_out_sum'] + 1)
        flow_agg['out_concentration'] = flow_agg['flow_out_max'] / (flow_agg['flow_out_mean'] + 1)
        flow_agg['out_frequency'] = flow_agg['flow_out_count'] / (flow_agg['flow_time_span_days'] + 1)
        
        # 交易金额分布特征
        flow_agg['amount_skewness_proxy'] = (flow_agg['flow_amount_mean'] - flow_agg['flow_amount_median']) / (flow_agg['flow_amount_std'] + 1)
        flow_agg['amount_dispersion'] = (flow_agg['amount_q75'] - flow_agg['amount_q25']) / (flow_agg['flow_amount_median'] + 1)
        
        # 时间维度特征
        flow_agg['time_regularity'] = 1 / (flow_agg['flow_time_std'] + 1)
        flow_agg['avg_days_between_trans'] = flow_agg['flow_time_span_days'] / (flow_agg['flow_count'] + 1)
        
        # 收入支出平衡特征
        flow_agg['net_flow_ratio'] = flow_agg['in_out_diff'] / (flow_agg['flow_amount_sum'] + 1)
        flow_agg['savings_rate'] = flow_agg['in_out_diff'] / (flow_agg['flow_in_sum'] + 1)
        flow_agg['spending_rate'] = flow_agg['flow_out_sum'] / (flow_agg['flow_in_sum'] + 1)
        
        # 交易行为变化特征（增强）
        flow_agg['recent_vs_early_ratio'] = flow_agg['recent10_amount_mean'] / (flow_agg['early10_amount_mean'] + 1)
        flow_agg['recent_out_trend'] = flow_agg['recent10_direction_mean']  # 越高说明最近支出越多
        flow_agg['behavior_stability'] = 1 - abs(flow_agg['amount_change'] - 1)  # 接近1说明行为稳定
        
        # 异常交易特征
        flow_agg['has_large_out'] = (flow_agg['flow_out_max'] > flow_agg['amount_q75'] * 2).astype(int)
        flow_agg['has_large_in'] = (flow_agg['flow_in_max'] > flow_agg['amount_q75'] * 2).astype(int)
        flow_agg['extreme_transaction_ratio'] = (flow_agg['flow_amount_max'] - flow_agg['amount_q75']) / (flow_agg['flow_amount_mean'] + 1)
        
        # 合并到主表
        df_result = df_main.merge(flow_agg, on='id', how='left')
        
        # 标记是否有交易流水
        df_result['has_flow'] = (~df_result['flow_count'].isna()).astype(int)
        
        # 填充没有交易流水的用户
        flow_cols = [col for col in df_result.columns if col.startswith('flow_') or 
                    col in ['in_out_ratio', 'in_out_diff', 'net_flow_per_day', 
                            'amount_cv', 'amount_range', 'amount_iqr',
                            'recent10_amount_mean', 'recent10_amount_sum', 'recent10_amount_std',
                            'recent10_direction_mean', 'early10_amount_mean', 'early10_amount_sum',
                            'amount_change', 'day_freq_mean', 'day_freq_std',
                            'amount_min_ratio', 'amount_min_to_max', 'direction_imbalance',
                            'mostly_out', 'mostly_in', 'out_to_in_ratio', 'out_max_ratio',
                            'out_concentration', 'out_frequency', 'amount_skewness_proxy',
                            'amount_dispersion', 'time_regularity', 'avg_days_between_trans',
                            'net_flow_ratio', 'savings_rate', 'spending_rate',
                            'recent_vs_early_ratio', 'recent_out_trend', 'behavior_stability',
                            'has_large_out', 'has_large_in', 'extreme_transaction_ratio']]
        
        for col in flow_cols:
            if col in df_result.columns:
                df_result[col] = df_result[col].fillna(0)
        
        # 静态特征与流水特征的交互（基于Top特征）
        # balance_accounts与交易流水的交互
        df_result['balance_accounts_times_flow_count'] = df_result['balance_accounts'] * df_result['flow_count']
        df_result['balance_accounts_times_out_mean'] = df_result['balance_accounts'] * df_result['flow_out_mean']
        df_result['balance_accounts_per_flow'] = df_result['balance_accounts'] / (df_result['flow_count'] + 1)
        
        # loan与交易流水的交互
        df_result['loan_to_flow_in_sum'] = df_result['loan'] / (df_result['flow_in_sum'] + 1)
        df_result['loan_to_flow_out_sum'] = df_result['loan'] / (df_result['flow_out_sum'] + 1)
        df_result['loan_to_net_flow'] = df_result['loan'] / (df_result['in_out_diff'] + 1)
        df_result['monthly_payment_to_flow_in'] = df_result['monthly_payment'] / (df_result['flow_in_mean'] + 1)
        
        # balance与交易流水的交互
        df_result['balance_to_flow_in_sum'] = df_result['balance'] / (df_result['flow_in_sum'] + 1)
        df_result['balance_to_net_flow'] = df_result['balance'] / (abs(df_result['in_out_diff']) + 1)
        df_result['balance_coverage'] = df_result['balance'] / (df_result['flow_out_mean'] * 30 + 1)  # 余额能覆盖几个月支出
        
        # term与交易流水的交互
        df_result['term_times_flow_freq'] = df_result['term'] * df_result['flow_freq']
        df_result['term_times_out_frequency'] = df_result['term'] * df_result['out_frequency']
        
        # 综合风险特征
        df_result['repayment_risk_score'] = (
            df_result['payment_to_balance'] * 0.3 + 
            df_result['loan_to_balance'] * 0.3 + 
            (1 - df_result['balance_to_limit']) * 0.2 +
            df_result['spending_rate'] * 0.2
        )
        
        df_result['flow_health_score'] = (
            df_result['savings_rate'] * 0.4 +
            (1 - df_result['direction_imbalance']) * 0.3 +
            df_result['behavior_stability'] * 0.3
        )
        
        # 有无交易流水的差异特征
        df_result['has_flow_high_loan'] = df_result['has_flow'] * (df_result['loan'] > df_result['loan'].median()).astype(int)
        df_result['no_flow_risk'] = (1 - df_result['has_flow']) * df_result['interest_rate']
        
        print(f"Bank statement features created. Shape: {df_result.shape}")
        return df_result
    
    def create_all_features(self, 
                           main_path: str, 
                           flow_path: str,
                           is_train: bool = True) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            main_path: 主表文件路径
            flow_path: 交易流水文件路径
            is_train: 是否为训练集
            
        Returns:
            完整特征的数据框
        """
        # 加载数据
        df_main, df_flow = self.load_data(main_path, flow_path)
        
        # 创建静态特征
        df_main = self.create_static_features(df_main)
        
        # 创建流水特征
        df_result = self.create_flow_features(df_main, df_flow)
        
        # 处理缺失值
        # 对于数值列，使用0或-1填充
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['id', 'label']:
                df_result[col] = df_result[col].fillna(-1)
        
        # 对于类别列，使用'missing'填充
        categorical_cols = df_result.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'level':  # level保留原样
                df_result[col] = df_result[col].fillna('missing')
        
        # 替换无穷值
        df_result = df_result.replace([np.inf, -np.inf], 0)
        
        print(f"\n{'='*50}")
        print(f"Feature engineering completed!")
        print(f"Final shape: {df_result.shape}")
        if is_train and 'label' in df_result.columns:
            print(f"Label distribution:\n{df_result['label'].value_counts()}")
            print(f"Label ratio: {df_result['label'].mean():.4f}")
        print(f"{'='*50}\n")
        
        return df_result


def prepare_features(is_train: bool = True) -> pd.DataFrame:
    """
    准备特征数据的便捷函数
    
    Args:
        is_train: 是否为训练集
        
    Returns:
        特征数据框
    """
    fe = FeatureEngineer()
    
    if is_train:
        main_path = 'data/train/train.csv'
        flow_path = 'data/train/train_bank_statement.csv'
    else:
        main_path = 'data/test/testab.csv'
        flow_path = 'data/test/testab_bank_statement.csv'
    
    df = fe.create_all_features(main_path, flow_path, is_train)
    return df


if __name__ == "__main__":
    # 测试特征工程
    print("Testing feature engineering on training data...")
    df_train = prepare_features(is_train=True)
    print(f"\nTraining features shape: {df_train.shape}")
    print(f"Training features preview:\n{df_train.head()}")
    
    print("\n" + "="*50 + "\n")
    
    print("Testing feature engineering on test data...")
    df_test = prepare_features(is_train=False)
    print(f"\nTest features shape: {df_test.shape}")
    print(f"Test features preview:\n{df_test.head()}")

