import os
import pandas as pd
import sklearn.datasets as datasets

def LoadBreastCancerDataset(splitxy=False):
    datasets.load_breast_cancer(return_X_y=splitxy, as_frame=True)

def ReadMatrixFromCsv(filepath, splitxy=False, label_name='Label'):
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        if splitxy is True:
            df_x = df.drop(columns=[label_name])
            df_y = df[label_name]
            return df_x, df_y
        else:
            return df
    else:
        print('The CSV file is not found')
        return None


def ReadFasta(fastapath, ul_seqlen=50):
    if os.path.isfile(fastapath):
        fp = open(fastapath, 'r')
    else:
        print('The fasta file is not found')
        return None

    seqDict = dict()
    notprocessed_seqDict = dict()
    currentHeader = ''
    currentSeq = ''
    line = fp.readline()
    # input fasta可能都是一行一條peptide，也沒有header，所以多加一個判斷是否有header，沒有的話就當作一行一條peptide
    if line.startswith('>'):
        with_header = True
    else:
        with_header = False

    if with_header is True:
        while(line):
            if line.startswith('>'):
                if currentHeader != '' and currentHeader not in seqDict and len(currentSeq) <= ul_seqlen:
                    seqDict[currentHeader] = currentSeq
                elif currentHeader != '' and currentHeader not in notprocessed_seqDict and len(currentSeq) > ul_seqlen:
                    notprocessed_seqDict[currentHeader] = currentSeq
                currentHeader = line[1:].replace('\n', '')
                currentSeq = ''
            else:
                currentSeq += line.rstrip()
            line = fp.readline()
        if currentHeader != '' and currentHeader not in seqDict and len(currentSeq) <= ul_seqlen:
            seqDict[currentHeader] = currentSeq
        elif currentHeader != '' and currentHeader not in notprocessed_seqDict and len(currentSeq) > ul_seqlen:
            notprocessed_seqDict[currentHeader] = currentSeq

    else:
        seq_count = 0
        while(line):
            seq_count += 1
            sudoheader = 'Sequence_' + str(seq_count)
            seq = line.replace('\n', '').rstrip()
            if len(seq) <= ul_seqlen:
                seqDict[sudoheader] = seq
            else:
                notprocessed_seqDict[sudoheader] = seq
            line = fp.readline()
    fp.close()
    return seqDict, notprocessed_seqDict

def WriteDataframe2Csv(df, csvpath):
    df.to_csv(csvpath, index=False)