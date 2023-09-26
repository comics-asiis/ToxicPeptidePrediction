# ToxTeller
ToxTeller is a toxicity prediction tool for peptides with length 10-50 amino acids. 

It provides four predictors trained using logistic regression (LR), support vector machine (SVM), random forest (RF), and XGBoost.

We also employ these four predictors in [ToxTeller webserver](https://comics.iis.sinica.edu.tw/ToxTeller).

## Program files
The essential files of ToxTeller are provided in the folder [_program_resource_](program_resource), and they need to be placed in the same folder to run ToxTeller. 

These files include four python files that perform the workflow and four pairs of pickle files that correspond to the four prediction models and respective scalers to scale input features. 


## Usage
ToxTeller takes a string of sequence file path as the argument.

The sequence file must be in FASTA format, and each sequence should be 10-50 amino acids.

The command is as follows:
```
python toxteller.py [sequence file path]
```

## Output
The output of ToxTeller is a CSV file showing the prediction results generated in the same folder of the sequence file. 

Each row represents a peptide with the format:

 Index, peptide name, peptide sequence, sequence length, prediction by LR, prediction by SVM, prediction by RF, prediction by XGBoost
 
The prediction is expressed by 1 (toxic) and 0 (non-toxic).

## Data
The training dataset and independent test dataset, which were used in the development stage of ToxTeller, are provided in the folder [_data_](data).

The independent test set consists of 100 toxic and 100 non-toxic peptide sequences, where all the 200 sequences share at most 40% similarity with each other (by CD-HIT) and with the training dataset (by CD-HIT-2D).

The training dataset, despite sharing at most 40% similarity with the independent test dataset, has at most 90% similarity for all the sequences in itself (by CD-HIT).

Note that the trained models (pickle files in [_program_resource_](program_resource)), also used in ToxTeller webserver, are trained on the whole collected dataset keeping at most 90% sequence similarity. 





