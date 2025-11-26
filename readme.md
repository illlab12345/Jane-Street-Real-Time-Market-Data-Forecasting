# Jane Street Real-Time Market Data Forecasting
## Project Overview
This project is a solution developed for the Jane Street Real-Time Market Data Forecasting competition, aiming to predict financial market response values (responder_6) using real-world data. The project adopts an ensemble learning framework, combining multiple types of machine learning models to enhance prediction performance.

**Competition Link**: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)

## System Architecture
The system adopts a modular design, consisting of the following main components:

1. **Data Processing & Feature Engineering**: Load and preprocess market data
2. **Multi-Model Training**: Train four different types of base models
3. **Model Ensemble**: Combine prediction results of each model through weighted fusion
4. **Inference Service**: Deploy the model for real-time prediction

### Workflow
```
Data Input → Feature Engineering → Parallel Training of Multi-Models → Model Ensemble → Prediction Output
```

## Feature Description
The system uses three types of features:

- **Base Features**: `feature_00` to `feature_78` (79 features in total)
- **Lag Features**: `responder_0_lag_1` to `responder_8_lag_1` (9 lag features in total)
- **Categorical Features**: `feature_09`, `feature_10`, `feature_11`

## Base Models
### 1. XGBoost Model
**Key Parameters**:
- Learning rate: 0.05
- Max depth: 6
- Subsample ratio: 0.9
- Regularization parameters: alpha=0.01, lambda=1
- Number of trees: 2000

**Features**: Uses early stopping to prevent overfitting, and conducts performance analysis by `symbol_id`.

### 2. Neural Network Model
**Network Architecture**:
- Input layer: 88 features (79 base features + 9 lag features)
- Hidden layers: [512, 512, 256]
- Activation function: SiLU (Sigmoid Linear Unit)
- Dropout regularization: [0.1, 0.1]
- Output layer: 1 neuron, using Tanh activation function

**Training Configuration**:
- Loss function: Weighted MSE loss
- Optimizer: Adam, learning rate = 1e-3
- Learning rate scheduling: ReduceLROnPlateau
- Batch size: 8192
- 5-fold cross-validation

### 3. Ridge Regression Model
- Uses default parameter configuration
- Training data sampling: 82% of the data is used for training
- Simple and efficient, providing basic prediction capabilities

### 4. TabM Model
**Core Components**:
- **Continuous Feature Processing**: Direct input to MLP
- **Categorical Feature Processing**: Uses OneHotEncoding0d
- **Ensemble Mechanism**: Uses LinearEfficientEnsemble for efficient integration

**Architectural Features**:
- Backbone network: 3-layer MLP with 512 neurons per layer
- Dropout rate: 0.25
- Number of ensembles (k): 32

**Training Configuration**:
- Loss function: Custom R2Loss
- Optimizer: AdamW, learning rate = 1e-4, weight decay = 5e-3
- Batch size: 8192
- Training epochs: 4

## Model Ensemble Strategy
The system integrates the prediction results of the four models using weighted averaging:

```python
# Ensemble weight distribution
weights = [0.70, 0.10, 0.40]

# Weighted fusion
final_pred = (pred_nn_xgb * weights[0] + 
             pred_ridge * weights[1] + 
             pred_tabm * weights[2]) / sum(weights)
```

**Weight Distribution**:
- NN+XGB combined model: 0.70 (highest weight)
- Ridge regression model: 0.10 (lowest weight)
- TabM model: 0.40

## Evaluation Metric
The system uses the weighted R² score as the main evaluation metric, which is consistent with the competition's official evaluation metric:

```python
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2
```

Formula:

```
R² = 1 - Σwᵢ(yᵢ - ŷᵢ)² / Σwᵢyᵢ²
```

Where y and ŷ are the ground-truth and predicted value vectors respectively, and w is the sample weight vector.

## Environment Requirements
### Hardware Requirements
- At least 1 GPU (NVIDIA GPU supporting CUDA is recommended)
- Memory: At least 64GB RAM

### Software Dependencies
- Python 3.9+
- PyTorch
- PyTorch Lightning
- XGBoost
- Polars
- Pandas
- NumPy
- Scikit-learn
- TabM-related libraries

## Usage Instructions
### 1. Data Preparation
Download the competition data:
```bash
git clone https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data
```

Ensure the data includes the following files:
- training.parquet
- validation.parquet
- Preprocessed lag feature data

### 2. Model Training
Run the following files in sequence to train each base model:

```bash
# Train the neural network model
jupyter notebook nn_train.ipynb

# Train the Ridge regression model
jupyter notebook ridge_train.ipynb

# Train the XGBoost model
jupyter notebook xgb_train.ipynb

# Train the TabM model
jupyter notebook tabm_train.ipynb
```

### 3. Model Ensemble
Run the ensemble script to merge the prediction results of each model:

```bash
jupyter notebook ensemble.ipynb
```

## Technical Highlights
### 1. Multi-Model Ensemble Strategy
By integrating different types of models (decision trees, neural networks, linear models), the system can capture different patterns and relationships in the data, improving the overall prediction stability and accuracy.

### 2. Innovation in Feature Engineering
- **Utilization of Lag Features**: Introduce responder lag features to capture time-series dependencies
- **Data Merging Techniques**: Add validation set data to the training set to improve model generalization
- **Categorical Feature Processing**: Conduct specialized encoding and processing for special categorical features

### 3. Efficient Data Processing
- Use the Polars library for large-scale data processing, which is more efficient than traditional Pandas
- Implement memory optimization to free up memory through `del` and `gc.collect()`
- Data caching strategy to avoid repeated data loading and preprocessing

### 4. Advanced Model Architecture
- **TabM Model**: Combines the latest technologies in tabular data processing and ensemble learning
- **Neural Network Design**: Uses BatchNorm, Dropout and other technologies to improve model performance
- **Custom Loss Function**: Loss function optimized for financial prediction tasks

## File Structure
```
├── describe.md           # Detailed system analysis report
├── ensemble.ipynb        # Model ensemble and inference code
├── nn_train.ipynb        # Neural network model training code
├── ridge_train.ipynb     # Ridge regression model training code
├── tabm_train.ipynb      # TabM model training code
├── tanm_reference.py     # Reference implementation of the TabM model
└── xgb_train.ipynb       # XGBoost model training code
```

## Notes
- Ensure the system meets the hardware requirements, especially in terms of memory and GPU configuration
- The training process may take a long time; it is recommended to run it in a stable computing environment
- Lag features need to be preprocessed and correctly associated with the main dataset
- Model weight files will occupy a large amount of storage space; ensure sufficient disk space is available

## Competition Background
This competition hosted by Jane Street Group aims to demonstrate the complexity and challenges of financial market forecasting. Participants need to address the following difficulties:
- Heavy-tailed distribution of financial data
- Non-stationary time series
- Sudden changes in market behavior
- The impact of human factors on the market

The competition uses real-world data, which, although anonymized, still retains the core challenges of practical financial forecasting problems.

## License
This project is for learning and competition use only.
