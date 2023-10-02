# ToxTeller
ToxTeller is a toxicity prediction tool for peptides with 10-50 amino acids. 

It provides four predictors trained using logistic regression (LR), support vector machine (SVM), random forest (RF) and XGBoost, respectively.

We also employ these four predictors in [ToxTeller webserver](https://comics.iis.sinica.edu.tw/ToxTeller).

## Program files
The essential files of ToxTeller are provided in the folder [_program_resource_](program_resource), and they need to be placed in the same folder to run ToxTeller. 

These files include four python files which perform the prediction workflow of the four predictors and four pairs of pickle files which correspond to the four prediction models and respective scalers to scale input features. 

The configuration file, which records the dependencies and version numbers for running ToxTeller, is also provided.


### Python files (source code)
+ toxteller.py => starts the prediction workflow taking the sequence file path as the argument.
+ modelwizard.py
+ accessdata.py
+ extractfeature.py

### Pickle files (models)
+ Logistic regression: model_lr.pkl, scaler_lr.pkl
+ Support vector machine: model_svm.pkl, scaler_svm.pkl
+ Random forest: model_rf.pkl, scaler_rf.pkl
+ XGBoost: model_xgboost.pkl, scaler_xgboost.pkl

### Dependencies 
+ requirements.txt


## Running ToxTeller
ToxTeller takes a string of sequence file path as the argument.

The sequence file must be in FASTA format, and each sequence consists of 10-50 amino acids.

The command to run ToxTeller is as follows:

	```
	python toxteller.py <SEQUENCE_FILE_PATH>
	```


## ToxTeller Output
The output of ToxTeller is a CSV file which contains the prediction results of four prediction models and is located in the same folder of the sequence file. 

Each row of the CSV file represents a peptide in the following format:

    Index, peptide name, peptide sequence, sequence length, prediction by LR, prediction by SVM, prediction by RF, prediction by XGBoost
 
The entry for prediction by each predictor is 1 if predicted as toxic, and 0 if predicted as non-toxic.


## Data
The training dataset and independent test dataset, which were used in the development stage of ToxTeller for fair performance evaluation without over estimation, are provided in the folder [_data_](data).

The independent test dataset consists of 100 toxic and 100 non-toxic peptide sequences, where all the 200 sequences share at most 40% similarity with each other (by CD-HIT) and with the training dataset (by CD-HIT-2D).

The training dataset, despite sharing at most 40% similarity with the independent test dataset, has at most 90% similarity among all the sequences in it (by CD-HIT).

Note that the trained models (pickle files in [_program_resource_](program_resource)), which are also used in ToxTeller webserver, are trained on the whole collected dataset, larger than the combination of independent test dataset and the training dataset with at most 90% sequence similarity. 





