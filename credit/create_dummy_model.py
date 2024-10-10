import xgboost as xgb
import numpy as np

# Create a dummy dataset
X = np.random.rand(100, 30)
y = np.random.randint(0, 2, 100)

# Create a DMatrix
dtrain = xgb.DMatrix(X, label=y, feature_names=[f'feature_{i}' for i in range(30)])

# Set parameters
params = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train the model
num_round = 10
model = xgb.train(params, dtrain, num_round)

# Save the model
model.save_model('xgboost_model.json')
print("Dummy model saved as 'xgboost_model.json'")