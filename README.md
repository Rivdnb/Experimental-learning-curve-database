Repository includes code and plots for the bachelor thesis titled: "Building an experimental learning curve database from many configurations of the Gradient Boosting algorithm".

General notes
- Python version 3.12.3, see requirements.txt for dependency versions
- The database folder contains the learning curves for ten different (preproccesed) OpenML datasets.
- The mean_val_curves_plots folder contains plots of all mean validation learning curves for each dataset (same datasets as the database folder).
- job.py is the code for producing the learning curves.
- example.py demonstrates basic usage of the database
Instructions job.py
- In order to run job.py, you require a dataset in CSV format. Any utf-8 values need to be decoded so it may work with the Gradient Boosting classifier.
- At the top you can find the dataset configuration, where the file paths and random seeds are tied to the datasets. If you want to run this code with a new dataset, make sure to include it in this configuration.
- Due to the cross-validation setup, some random seeds may not work for smaller datasets. You might have to brute force and find a working seed.
- Commented code is for exporting to CSV, make sure to remove/comment out the JSONL equivalent code. (JSONL is recommened for clear structure)
- In main() you can uncomment the Multiprocessing lines for parallel execution using Slurm, make sure to again remove/comment out the Local loop code.
- Run command: python job.py --dataset "dataset_name" --runs "amount of runs"

Instructions example.py
- This file demonstrates loading and extracting information from a learning curve dataset, and includes an example of an experiment using this dataset.
- In extract_info() you have the option to extract either the mean validation or mean test scores.
- The experiment is a surrogate-modeling task, where a Random Forest regressor is trained on learning curves corresponding to different hyperparameter configurations. The regressor then predicts learning curves for unknown configurations. This is implemented using a 5-fold cross-validation setup. For each fold, the score is calculated per anchor and an average score across anchors for that fold is calculated as well. Lastly, the average is calculated over all the average fold scores.
- In main() you can change the path to the learning curve dataset (JSONL file)
- Run command: python example.py
