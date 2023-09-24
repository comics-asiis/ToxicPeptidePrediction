# ToxTeller
ToxTeller is a toxicity prediction tool for peptides with length 10-50 amino acids. 
It provides four predictors trained using logistic regression (LR), support vector machine (SVM), random forest (RF), and XGBoost.

## Usage
ToxTeller takes a string of sequence file name/path as the argument.
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

