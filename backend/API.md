# API 定义清单

## 1. 数据上传接口
- **POST** /upload
- **功能**: 上传CSV/TXT数据集，返回列名列表。
- **请求**: multipart/form-data (file)
- **响应**: { "columns": ["age", "income", "is_default", ...] }

## 2. 开始建模接口
- **POST** /model/start
- **功能**: 接收目标变量和配置，开始异步建模流程。
- **请求**: { "target": "is_default", "test_ratio": 0.2, "description": "..." }
- **响应**: { "task_id": "uuid-1234", "status": "started" }

## 3. 进度查询接口
- **GET** /model/status/{task_id}
- **功能**: 获取当前建模进度。
- **响应**: { "task_id": "...", "progress": 45, "step": "特征工程", "message": "..." }

## 4. 获取分析报告
- **GET** /model/report/{task_id}
- **功能**: 获取建模后的详细指标（IV/WOE等）。
- **响应**: { "accuracy": 0.94, "ks": 0.52, "features": [...] }
