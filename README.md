# ToxTeller
ToxTeller is a toxicity prediction tool for peptides with length 10-50 amino acids. 

It provides four predictors trained using logistic regression (LR), support vector machine (SVM), random forest (RF), and XGBoost.

## Usage
ToxTeller takes a string of sequence file path as the argument.

The sequence file must be in FASTA format, and each sequence should be 10-50 amino acids.

The command is as follows:
```
python toxteller.py [sequence file path]
```

## Output
The output of ToxTeller is a CSV file created in the same folder of the sequence file showing the prediction results. 

Each row represent a peptide sequence with the format:

 Index, peptide name, peptide sequence, sequence length, prediction by LR, prediction by SVM, prediction by RF, prediction by XGBoost
 
The prediction is expressed by 1 (toxic) and 0 (non-toxic).

## Data
The training dataset and independent test dataset are provided in the folder _data_.

The training dataset, where the sequences share similarity smaller than 90%, was used to optimize hyper-parameters and feature combinations in the development stage of ToxTeller.

The independent test set, which consists of sequences sharing similarity smaller than 40% with the training dataset and within itself, was compiled for confirming the effectiveness of the model development process.

The details for compiling the dataset are shown in the figure belows.

![The flowchart ](/assets/images/dataset_flowchart.png)

The trained models (pickle files in _source_code_), also used in ToxTeller webserver, are trained on the whole dataset we collected keeping at most 90% sequence similarity. 





