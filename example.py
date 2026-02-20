from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import json
import numpy as np
import pandas as pd

# Load curves from JSONL
def load_curves(path):
    curves = []
    with open(path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            curves.append(entry)
    return curves

# Preprocess: Extract X (hyperparameter configurations), y (scores), anchors, dataset-name
def extract_info(curves):
    X, y = [], []
    for c in curves:
        # Hyperparameter configurations
        cfg = c['hyperparam_values']
        x_vec = [ 
            cfg['learning_rate'],
            cfg['n_estimators'],
            cfg['max_depth'],
            cfg['min_samples_split'],
            cfg['min_samples_leaf'],
            cfg['subsample']
        ]
        X.append(x_vec)
        # Scores, options: val_means|test_means
        y.append(c['learning_curve']['val_means'])
    anchors = curves[0]["learning_curve"]["anchors"]
    dataset = curves[0]["dataset"]
    return np.array(X), np.array(y), anchors, dataset

# --- Experiment: Surrogate Modeling with Random Forest ---
def surrogate_modeling(X, y, anchors, dataset):
    
    accum_rmse_fold = 0 # Accumaltive RMSE over all folds
    rmse_anchor = []# List of RMSE scores per anchor
    
    # Cross-validation
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    for f, (train_index, test_index) in enumerate(cv.split(X)):
        # Train set
        X_train = X[train_index] # Train instances
        y_train = y[train_index] # Train labels
        # Test set
        X_test = X[test_index] # Test instances
        y_test = y[test_index] # Test labels
        # Surrogate model: fit & predict
        surrogate = RandomForestRegressor(random_state=8)
        surrogate.fit(X_train, y_train)
        y_pred = surrogate.predict(X_test)

    # RMSE per anchor for fold 'f' 
        for i in range(len(anchors)):
            rmse_anchor = root_mean_squared_error(y_test[:, i], y_pred[:, i])
            # Export
            #df_rmse_a = pd.DataFrame([[dataset,model,r,f,anchors[i],rmse_anchor]],columns=['dataset','cv_run', 'fold', 'anchor', 'RMSE'])
            #df_rmse_a.to_csv('exp_1/rmse_anchors.csv', mode='a', index=False, header=False)

    # RMSE for fold 'f'
        rmse_fold = root_mean_squared_error(y_test, y_pred)
        print("RMSE Fold ", f, ": ", rmse_fold)
        # Export
        #df_rmse_fold = pd.DataFrame([[dataset,model,r,f,rmse_fold]],columns=['dataset', 'model', 'cv_run', 'fold', 'RMSE'])
        #df_rmse_fold.to_csv('exp_1/rmse_fold.csv', mode='a', index=False, header=False)

        accum_rmse_fold += rmse_fold

    # Calculate average RMSE of all folds
    rmse_cv = accum_rmse_fold/cv.get_n_splits()
    # Export
    #df_rmse_cv = pd.DataFrame([[dataset,model,r,rmse_cv]],columns=['dataset', 'model', 'cv_run', 'RMSE'])
    #df_rmse_cv.to_csv('exp_1/rmse_cv.csv', mode='a', index=False, header=False)
    print(f"RMSE average of cross-validation: {rmse_cv:.6f}")


# --- Main ---
if __name__ == '__main__':
    dataset_paths = ["./lc/lc_waveform.jsonl"] # Path to learning curve JSONL file
    for path in dataset_paths:
        curves = load_curves(path)
        X, y, anchors, dataset = extract_info(curves)
    # Experiment 1
        surrogate_modeling(X, y, anchors, dataset)