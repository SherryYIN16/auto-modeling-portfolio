# 金融风控二分类模型

## 项目简介

本项目是一个金融风控二分类模型，用于预测用户是否存在违约风险。模型结合了用户信贷静态特征和交易流水动态特征，使用AutoGluon自动机器学习框架进行建模。

## 项目结构

```
fin/
├── data/                          # 数据目录
│   ├── train/                     # 训练数据
│   │   ├── train.csv             # 用户信贷特征和label
│   │   └── train_bank_statement.csv  # 用户交易流水
│   ├── test/                      # 测试数据
│   │   ├── testaa.csv            # 测试集用户信贷特征
│   │   └── testaa_bank_statement.csv  # 测试集交易流水
│   ├── train_features.csv         # 处理后的训练特征（生成）
│   └── test_features.csv          # 处理后的测试特征（生成）
├── models/                        # 模型保存目录（自动创建）
├── feature_engineering.py         # 特征工程模块
├── model_trainer.py               # 模型训练模块
├── main.py                        # 主函数
├── requirements.txt               # 依赖包
└── README.md                      # 项目说明
```

## 特征工程

### 1. 静态特征（信贷特征）

- **基础特征**：直接来自原始数据的特征
- **贷款相关特征**：
  - loan_to_limit: 贷款额/信用额度
  - balance_to_limit: 余额/信用额度
  - loan_to_balance: 贷款额/余额
  
- **时间相关特征**：
  - history_years: 信用历史年限
  - record_to_issue: 记录时间到发放时间的间隔
  - history_to_issue: 历史时间到发放时间的间隔
  
- **账户相关特征**：
  - balance_accounts_ratio: 有余额账户占比
  - avg_balance_per_account: 平均每账户余额
  - avg_limit_per_account: 平均每账户额度
  
- **贷款压力指标**：
  - loan_per_year: 年均贷款额
  - monthly_payment: 月供
  - payment_to_balance: 月供/余额
  - payment_to_limit: 月供/额度
  
- **利率相关特征**：
  - total_interest: 总利息
  - total_repayment: 总还款额
  - monthly_repayment: 月还款额
  
- **交叉特征**：
  - level编码
  - 风险标记（高风险贷款、低余额高贷款等）
  - 分组统计（按career、term、level分组的统计量）
  
- **高阶特征**：
  - 多项式特征（平方项、交互项）

### 2. 动态特征（交易流水特征）

- **基础聚合统计**：
  - 交易次数、总额、均值、标准差、最大值、最小值、中位数
  - 时间跨度、交易频率
  
- **方向特征**（收入/支出）：
  - 收入交易统计
  - 支出交易统计
  - 收支比、收支差、日均净流水
  
- **交易金额波动性**：
  - 变异系数
  - 金额范围
  - 四分位距
  
- **序列特征**：
  - 最近10笔交易统计
  - 最早10笔交易统计
  - 行为变化（最近vs最早）
  
- **周期性特征**：
  - 按星期几的交易频率统计

## 模型配置

使用AutoGluon进行自动建模，包含以下模型：

1. **LightGBM (GBM)** - 2组参数
   - 轻量级梯度提升机，训练速度快
   
2. **XGBoost (XGB)** - 2组参数
   - 经典梯度提升算法，性能稳定
   
3. **CatBoost (CAT)** - 2组参数
   - 处理类别特征能力强
   
4. **Neural Network (NN_TORCH)** - 2组参数
   - 深度学习模型，捕捉复杂非线性关系
   
5. **Random Forest (RF)** - 2组参数
   - 随机森林，鲁棒性好
   
6. **Stacking Ensemble**
   - 自动集成多个模型的预测结果

## 安装依赖

```bash
pip install -r requirements.txt
```

注意：AutoGluon需要Python 3.8-3.11版本。

## 使用方法

### 1. 完整流程（特征工程 + 训练 + 预测）

```bash
python main.py
```

这将执行以下步骤：
1. 生成训练集和测试集特征
2. 划分训练集和验证集（80/20）
3. 使用AutoGluon训练多个模型
4. 在验证集上评估模型（输出AUC）
5. 对测试集进行预测并生成提交文件

### 2. 仅测试特征工程

```bash
python feature_engineering.py
```

### 3. 使用已训练模型快速预测

```python
from main import quick_predict

# 指定模型路径
quick_predict('models/model_20241025_120000')
```

## 评估指标

- **主要指标**：AUC (Area Under the ROC Curve)
- **验证策略**：分层抽样，保持label分布一致

## 输出文件

1. **特征文件**：
   - `data/train_features.csv`: 训练集特征
   - `data/test_features.csv`: 测试集特征

2. **模型文件**：
   - `models/model_YYYYMMDD_HHMMSS/`: 训练好的模型

3. **预测文件**：
   - `submission_YYYYMMDD_HHMMSS.csv`: 测试集预测结果

4. **评估结果**：
   - `models/model_YYYYMMDD_HHMMSS/evaluation_results.txt`: 模型评估详情

## 注意事项

1. **避免数据泄露**：
   - 特征工程中严格避免使用label信息
   - 分组统计使用全量数据但不涉及label
   - 验证集划分使用分层抽样

2. **缺失值处理**：
   - 数值特征：填充-1或0
   - 类别特征：填充'missing'
   - 标记缺失值特征

3. **计算稳定性**：
   - 除法运算添加小常数避免除零
   - 替换无穷值

4. **模型训练**：
   - 默认训练时间限制：1小时
   - 使用5折bagging和1层stacking
   - 自动保存最佳模型

## 性能优化建议

1. 如需更快速度，可调整：
   - 减少`time_limit`参数
   - 使用`presets='medium_quality'`

2. 如需更高性能，可调整：
   - 增加`time_limit`参数
   - 增加`num_bag_folds`和`num_stack_levels`
   - 添加更多特征工程

3. 内存优化：
   - 如遇内存问题，可减少模型数量
   - 或使用`presets='optimize_for_deployment'`

## 作者

金融风控建模项目

## 许可

MIT License

