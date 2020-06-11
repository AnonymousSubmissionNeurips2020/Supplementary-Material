:: Class for our proposed procedures (BKK,SBKK,ABKK) is the file Closed_form_estimator.py. API is consistent with scikit-learn.
:: To run all experiments, either run all cells of the following jupyter notebooks (in that order):
:: - dataset_preprocessing.ipynb
:: - synthetic_dataset_generation.ipynb
:: - evaluation.ipynb

:: Or the following batch commands
mkdir dataset_folder

bash download_raw_data.sh
python3 dataset_preprocessing.py
bash remove_raw_data.sh

python3 synthetic_dataset_generation.py

python3 evaluation.py

:: Results files are npy matrices indexed as follow (Method, Dataset, Seed, Metric), where
:: - Method is BKK, SBKK, ABKK, RidgeCV, LassoCV or ElasticNetCV
:: - Dataset is either synthetic A,B,C or UCI 1,..,14 or 20news 500,...,2875
:: - Seed is 0,..,99
:: - Metric is Running Time, R2-score on test-set, Loss of ER-G criterion at convergence, Gradient-step iterations, Lambda value


