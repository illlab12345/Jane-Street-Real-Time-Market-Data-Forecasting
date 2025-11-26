# 分析报告

## 1. 系统概述

该系统是一个基于集成学习的金融市场预测框架，专门用于预测股票市场响应值（responder_6）。系统通过组合多种不同类型的机器学习模型（XGBoost、神经网络、Ridge回归和TabM）来提高预测性能，采用加权融合的方式整合各模型的预测结果。

## 2. 系统架构与工作流程

系统整体采用模块化设计，由以下几个主要部分组成：

1. **数据处理与特征工程**：加载和预处理市场数据，生成特征
2. **基础模型训练**：训练四种不同类型的模型
3. **模型集成**：通过加权融合组合各模型预测结果
4. **推理服务**：部署模型进行实时预测

系统工作流程如下：
```
数据输入 → 特征工程 → 多模型并行训练 → 模型集成 → 预测输出
```

## 3. 数据处理与特征工程

### 3.1 数据来源与加载

系统支持两种数据来源路径：
- 本地路径：`./input_df`
- Kaggle竞赛路径：`/kaggle/input/`

使用Polars库进行高效数据读取和处理，特别适合大规模数据集：

```python
df = pl.scan_parquet(f"{input_path}/training.parquet").collect().to_pandas()
valid = pl.scan_parquet(f"{input_path}/validation.parquet").collect().to_pandas()
```

### 3.2 特征定义

系统使用多种特征组合：

- **基础特征**：`feature_00` 到 `feature_78`（共79个特征）
- **滞后特征**：`responder_0_lag_1` 到 `responder_8_lag_1`（共9个滞后特征）
- **类别特征**：`feature_09`、`feature_10`、`feature_11`

### 3.3 数据预处理

- **数据合并技巧**：将验证集数据添加到训练集中，提高模型性能
  ```python
df = pd.concat([df, valid]).reset_index(drop=True)# A trick to boost LB from 0.0045->0.005
  ```

- **缺失值处理**：使用前向填充和0填充
  ```python
df[feature_names] = df[feature_names].fillna(method = 'ffill').fillna(0)
  ```

- **类别特征编码**：使用预定义映射将类别特征转换为数值
  ```python
category_mappings = {'feature_09': {...}, 'feature_10': {...}, 'feature_11': {...}}
  ```

- **数据标准化**：对连续特征进行标准化处理
  ```python
def standardize(df, feature_cols, means, stds):
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])
  ```

## 4. 基础模型实现

### 4.1 XGBoost模型

XGBoost是一个基于梯度提升的决策树模型，在金融预测任务中表现出色。

**关键参数配置**：
- 学习率：0.05
- 最大深度：6
- 子样本比例：0.9
- 正则化参数：alpha=0.01, lambda=1
- 树的数量：2000

**训练策略**：
- 使用early stopping防止过拟合
- 按symbol_id进行性能分析
- 保存最佳模型供后续集成使用

### 4.2 神经网络模型

使用PyTorch Lightning实现的多层感知机模型，能够捕捉非线性关系。

**网络架构**：
- 输入层：88个特征（79个基础特征 + 9个滞后特征）
- 隐藏层：[512, 512, 256]
- 激活函数：SiLU（Sigmoid Linear Unit）
- Dropout正则化：[0.1, 0.1]
- 输出层：1个神经元，使用Tanh激活函数

**训练配置**：
- 损失函数：加权MSE损失
- 优化器：Adam，学习率1e-3
- 学习率调度：ReduceLROnPlateau
- 批量大小：8192
- 5折交叉验证

```python
class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        # 网络结构定义
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1)) 
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        # ...
```

### 4.3 Ridge回归模型

Ridge回归是一种线性模型，添加了L2正则化，有助于防止过拟合。

**特点**：
- 使用默认参数配置
- 训练数据采样：82%的数据用于训练
- 简单高效，提供基础预测能力

### 4.4 TabM模型

TabM（Table Multimodal）是一种专门为表格数据设计的集成学习模型，能够同时处理连续特征和类别特征。

**核心组件**：
- **连续特征处理**：直接输入MLP
- **类别特征处理**：使用OneHotEncoding0d
- **集成机制**：使用LinearEfficientEnsemble实现高效集成

**架构特点**：
- 主干网络：3层MLP，每层512个神经元
- Dropout率：0.25
- 集成数量（k）：32
- 类别特征基数：[23, 10, 32, 40, 969]

**训练配置**：
- 损失函数：自定义R2Loss
- 优化器：AdamW，学习率1e-4，权重衰减5e-3
- 批量大小：8192
- 训练轮次：4轮

```python
model = Model(
    n_num_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3,
        'd_block': 512,
        'dropout': 0.25,
    },
    arch_type=arch_type,
    k=k,
)
```

## 5. 模型集成策略

系统采用加权平均的方式集成四个模型的预测结果，具体实现如下：

```python
def predict_ensemble(data):
    # 获取各个模型的预测
    pred_nn_xgb = predict_nn_xgb(data)
    pred_ridge = predict_ridge(data)
    pred_tabm = predict_tabm(data)
    
    # 加权融合
    weights = [0.70, 0.10, 0.40]
    final_pred = (pred_nn_xgb * weights[0] + 
                 pred_ridge * weights[1] + 
                 pred_tabm * weights[2]) / sum(weights)
    
    return final_pred
```

**权重分布**：
- NN+XGB组合模型：0.70（最高权重）
- Ridge回归模型：0.10（最低权重）
- TabM模型：0.40

这种权重分配反映了各模型在验证集上的表现，NN+XGB组合模型获得了最高的权重，表明它在预测任务中表现最佳。

## 6. 评估指标与性能

### 6.1 评估指标

系统使用加权R²分数作为主要评估指标：

```python
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2
```

### 6.2 模型性能

各模型在验证集上的性能表现：

- **XGBoost模型**：通过R²分数评估，按symbol_id分析性能差异
- **神经网络模型**：最终验证R²约为0.009+（根据训练日志）
- **Ridge回归模型**：测试加权R²约为0.0024
- **TabM模型**：最终验证R²达到0.009890

集成模型通过组合各模型优势，最终性能超过任何单个模型。

## 7. 推理服务部署

系统提供了与Kaggle推理服务器兼容的接口，用于部署模型进行实时预测：

```python
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )
```

## 8. 技术亮点与创新

### 8.1 多模型集成策略

通过集成不同类型的模型（决策树、神经网络、线性模型），系统能够捕捉数据中的不同模式和关系，提高整体预测稳定性和准确性。

### 8.2 特征工程创新

- **滞后特征利用**：引入responder滞后特征，捕捉时间序列依赖性
- **数据合并技巧**：将验证集数据添加到训练集，提高模型泛化能力
- **特征标准化**：对连续特征进行标准化，提高模型训练效率

### 8.3 高效数据处理

- 使用Polars库进行大规模数据处理，比传统Pandas更高效
- 采用数据缓存策略，避免重复数据加载和预处理
- 实现内存优化，通过del和gc.collect()释放内存

### 8.4 先进的模型架构

- **TabM模型**：结合了表格数据处理和集成学习的最新技术
- **神经网络设计**：使用BatchNorm、Dropout等技术提高模型性能
- **自定义损失函数**：针对金融预测任务优化的损失函数设计
