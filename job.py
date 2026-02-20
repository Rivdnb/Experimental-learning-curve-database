from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import time
import os
import math
import json
import pandas as pd
import numpy as np
import random
import multiprocessing as mp
import argparse

dataset_config = {
    "eeg": {
        "csv": "./datasets/eeg.csv",
        "seed": 0
    },
    "blood": {
        "csv": "./datasets/blood.csv",
        "seed": 1
    },
    "banknote": {
        "csv": "./datasets/banknote.csv",
        "seed": 0
    },
    "steel": {
        "csv": "./datasets/steel.csv",
        "seed": 0
    },
    "mfeat": {
        "csv": "./datasets/mfeat.csv",
        "seed": 0
    },
    "nomao": {
        "csv": "./datasets/nomao.csv",
        "seed": 0
    },
    "texture": {
        "csv": "./datasets/texture.csv",
        "seed": 11
    },
    "waveform": {
        "csv": "./datasets/waveform.csv",
        "seed": 0
    },
    "phoneme": {
        "csv": "./datasets/phoneme.csv",
        "seed": 0
    },
    "churn": {
        "csv": "./datasets/churn.csv",
        "seed": 20
    }
}

def mysim(run ,df, d_config, dataset):# (run-number, data, dataset-configuration, dataset-name)
    pid = os.getpid()
    # Set random-seeds
    seed = (pid + time.time_ns()) % 10**6
    np.random.seed(seed=seed)
    random.seed(seed)
    # Nested CV settings
    outer_folds = 5
    inner_folds = 5 
    outer_test_size = 0.1
    inner_val_size = 0.1
    val_test_cap = 5000 # cap on outer test and inner validation sets
    k = 0 # iterator (anchors)
    anchor = 16 # current anchor (starts with 16)
    anchor_check = False
    # Generate random config
    lr = round(float(np.exp(np.random.uniform(np.log(0.01), np.log(0.2)))), 6) # learning rate
    n_est = random.randint(50, 300) # Number of estimators
    max_dpt = random.randint(3, 20) # Max depth
    min_smpl_splt = random.randint(2, 8) # Minimum sample split
    min_smpl_leaf = random.randint(1, 4) # Minimum sample leaf
    sub_smpl = round(random.uniform(0.5, 1),6) # Subsample
    # Logging dictionary
    hyperparam_values = {
        "learning_rate": lr,
        "n_estimators": n_est,
        "max_depth": max_dpt,
        "min_samples_split": min_smpl_splt,
        "min_samples_leaf": min_smpl_leaf,
        "subsample": sub_smpl
    }

    X = df.iloc[:, :-1] # Feature columns (everything but the last column)
    y = df.iloc[:, -1] # Class column (last column)
    anchors = []# List of anchors
    max_train_size = ((1-outer_test_size) * (1-inner_val_size)) * len(y)

# If saving to CSV
    #lc = []# Learning curve
    #csv_instance = []# Complete instance (config,lc)
# If saving to JSONL
    val_scores_list = []
    val_mean_scores_list = []
    test_scores_list = []
    test_mean_scores_list = []

# Nested CV
    cv_start_time = time.time()
    while anchor <= max_train_size:
        accum_val_score = 0 # Accumalative validation score (sum)
        accum_test_score = 0 # Accumalative test score (sum)
        val_scores = [] # Individual validation scores
        test_scores = [] # Individual test scores
        outer_cv = StratifiedShuffleSplit(n_splits=outer_folds, test_size=outer_test_size, random_state=d_config["seed"])
        for outer_fold, (outer_train, outer_test) in enumerate(outer_cv.split(X, y)):
            # Outer train fold
            X_outer_train = X.iloc[outer_train].reset_index(drop=True)
            y_outer_train = y.iloc[outer_train].reset_index(drop=True)
            # Outer test fold
            X_outer_test = X.iloc[outer_test].reset_index(drop=True)
            y_outer_test = y.iloc[outer_test].reset_index(drop=True)
            # Apply cap to test set
            if len(X_outer_test) > val_test_cap:
                X_outer_test = X_outer_test.iloc[:val_test_cap]
                y_outer_test = y_outer_test.iloc[:val_test_cap]

            # Inner CV on outer_train
            inner_cv = StratifiedShuffleSplit(n_splits=inner_folds, test_size=inner_val_size, random_state=d_config["seed"]+outer_fold)
            for inner_fold, (inner_train, inner_val) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
                # Inner train set
                X_inner_train = X_outer_train.iloc[inner_train].reset_index(drop=True)
                y_inner_train = y_outer_train.iloc[inner_train].reset_index(drop=True)
                if(len(X_inner_train) < anchor):
                    print("Safety check: Not enough training data for anchor")
                    anchor_check = True
                    break

                # Inner validation set
                X_inner_val = X_outer_train.iloc[inner_val].reset_index(drop=True)
                y_inner_val = y_outer_train.iloc[inner_val].reset_index(drop=True)

                # Apply cap to inner validation set
                if len(X_inner_val) > val_test_cap:
                    X_inner_val = X_inner_val.iloc[:val_test_cap]
                    y_inner_val = y_inner_val.iloc[:val_test_cap]

                # Subsample anchors monotonically
                X_anchor = X_inner_train.iloc[:anchor]
                y_anchor = y_inner_train.iloc[:anchor]
                try:
                    # Fit model on subsample
                    model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est, max_depth=max_dpt,
                                                    min_samples_split=min_smpl_splt,min_samples_leaf=min_smpl_leaf,subsample=sub_smpl, random_state=11)
                    model.fit(X_anchor, y_anchor)

                    # Score on inner validation
                    val_score = accuracy_score(y_inner_val, model.predict(X_inner_val))
                    val_scores.append(val_score)
                    accum_val_score += val_score
                    # Score on outer test
                    test_score = accuracy_score(y_outer_test, model.predict(X_outer_test))
                    test_scores.append(test_score)
                    accum_test_score += test_score
                except Exception as e:
                    error_details = {
                        "error": str(e),
                        "pid": pid,
                        "run": run,
                        "anchor": anchor,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "hyperparam_values": hyperparam_values,
                        "dataset": dataset,
                        "seed": d_config["seed"]
                    }

                    print("Error during nested cross-validation:", error_details)
                    return {"error": error_details}
        if anchor_check: # Anchor exceeds limit
            break
        else:
            val_mean_score = accum_val_score/(outer_folds * inner_folds) # Mean val score
            test_mean_score = accum_test_score/(outer_folds * inner_folds) # Mean test score
            anchors.append(anchor) # Add current anchor to list
        # Iterate to next anchor
            k += 1
            anchor = math.ceil(16 * (2 ** (k / 2)))
        # CSV
            #lc_point = (anchor, val_scores, val_mean_score, test_scores, test_mean_score)
            #lc.append(lc_point)
        # JSON
            val_scores_list.append(val_scores)
            val_mean_scores_list.append(val_mean_score)
            test_scores_list.append(test_scores)
            test_mean_scores_list.append(test_mean_score)

    cv_time = time.time() - cv_start_time
# Save learning curve to JSONL
    json_instance = {
            "dataset": dataset,
            "seed": d_config["seed"],
            "hyperparam_values": hyperparam_values,
            "runtime": cv_time,
            "learning_curve": {
                "anchors": anchors,
                "val_scores": val_scores_list,
                "val_means": val_mean_scores_list,
                "test_scores": test_scores_list,
                "test_means": test_mean_scores_list
            }
        }
    os.makedirs("./lc", exist_ok=True)
    filepath = os.path.join("./lc", f"{dataset}_{pid}.jsonl")
    with open(filepath, "a") as f:
        f.write(json.dumps(json_instance) + "\n")
# Save learning curve to CSV
    #csv_instance.append((dataset,d_config["seed"],lr,n_est,max_dpt,min_smpl_splt,min_smpl_leaf,sub_smpl,cv_time,lc))
    #df_lc = pd.DataFrame(csv_instance)
    #df_lc.to_csv('./lc/placeholder.csv', mode='a', index=False, header=False)


def main(): 
# Parsing the command line arguments (dataset & #runs)
    parser = argparse.ArgumentParser(description="Dataset & train sizes")
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_config.keys(),
                        help="Dataset name.")
    parser.add_argument("--runs", type=int, required=True,
                        help="The amount of runs to execute.")

    args = parser.parse_args()
    d_config = dataset_config.get(args.dataset)

# Check for invalid inputs
    if d_config is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.runs <= 0 :
        raise ValueError(f"Amount of runs needs to be above 0, value given: {args.runs}")
# Check for dataset CSV file
    csv_file = d_config["csv"]
    # Load file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found in current directory.")
# Import dataset CSV
    df = pd.read_csv(csv_file, encoding = 'latin-1')

# Multiprocessing (remove/comment out the Local loop)
    # Slurm
    #n_cores = os.environ['SLURM_CPUS_PER_TASK']
    #n_nodes = os.environ['SLURM_JOB_NUM_NODES']
    # Go through the runs in parallel
    #pool = mp.Pool(processes=int(n_cores)*int(n_nodes))
    #res = pool.starmap(mysim, [(run,df,d_config,args.dataset) for run in range(args.runs)])
    #pool.close()
    #pool.join()

# Local loop (remove/comment out Multiprocessing)
    for run in range(args.runs):
        mysim(run, df, d_config, args.dataset)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Python script finished (running time: {0:.1f}s)".format(time.time() - start_time))
