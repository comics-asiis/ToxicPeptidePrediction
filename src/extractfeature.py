import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

aaChargeDi = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 1.0,
              'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1.0, 'S': 0, 'T': 0, 'V': 0,
              'W': 0, 'Y': 0}   # positive: R,K  negative: D,E  neutral: others
aaHydroIdxDi = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'W': -0.9,
                'S': -0.8, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9,
                'R': -4.5}
aaHyd1Di = {'A': 0.62, 'C': 0.29, 'D': -0.9, 'E': -0.74, 'F': 1.19, 'G': 0.48, 'H': -0.4, 'I': 1.38, 'K': -1.5,
              'L': 1.06, 'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18, 'T': -0.05, 'V': 1.08,
              'W': 0.81, 'Y': 0.26}  # hydrophobicity
aaHyd2Di = {'A': -0.5, 'C': -1, 'D': 3, 'E': 3, 'F': -2.5, 'G': 0, 'H': -0.5, 'I': -1.8, 'K': 3,
              'L': -1.8, 'M': -1.3, 'N': 2, 'P': 0, 'Q': 0.2, 'R': 3, 'S': 0.3, 'T': -0.4, 'V': -1.5,
              'W': -3.4, 'Y': -2.3}  # hydrophilicity
aaPIDi = {'A': 6, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74,
              'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.3, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.6, 'V': 5.96,
              'W': 5.89, 'Y': 5.66}   # isoelectric point, according to Chapter 27 of course of www.chem.ucalgary.ca
aaPKa1Di = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18,
              'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32,
              'W': 2.83, 'Y': 2.2}   # pKa1: carboxyl group (-COOH), according to Chapter 27 of course of www.chem.ucalgary.ca
aaPKa2Di = {'A': 9.69, 'C': 8.18, 'D': 9.6, 'E': 9.67, 'F': 9.13, 'G': 9.6, 'H': 9.17, 'I': 9.6, 'K': 8.95,
              'L': 9.6, 'M': 9.21, 'N': 8.8, 'P': 10.6, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.1, 'V': 9.62,
              'W': 9.39, 'Y': 9.11}   # pKa2: ammonium group (NH2-), according to Chapter 27 of course of www.chem.ucalgary.ca
aaVolumeDi = {'A': 91.5, 'R': 196.1, 'N': 138.3, 'D': 135.2, 'C': 102.4, 'Q': 156.4, 'E': 154.6, 'G': 67.5, 'H': 163.2,
              'I': 162.6, 'L': 163.4, 'K': 162.5, 'M': 165.9, 'F': 198.8, 'P': 123.4, 'S': 102.0, 'T': 126.0,
              'W': 237.2, 'Y': 209.8, 'V': 138.4}
aaVSCDi = {'A': 27.5, 'C': 44.6, 'D': 40, 'E': 62, 'F': 115.5, 'G': 0, 'H': 79, 'I': 93.5, 'K': 100, 'L': 93.5,
              'M': 94.1, 'N': 58.7, 'P': 41.9, 'Q': 80.7, 'R': 105, 'S': 29.3, 'T': 51.3, 'V': 71.5, 'W': 145.5, 'Y': 117.3}
aaGappDi = {'A': 0.11, 'R': 2.58, 'N': 2.05, 'D': 3.49, 'C': -0.13, 'Q': 2.36, 'E': 2.68, 'G': 0.74, 'H': 2.06, 'I': -0.6, 'L': -0.55, 'K': 2.71, 'M': -0.1, 'F': -0.32, 'P': 2.23, 'S': 0.84, 'T': 0.52, 'W': 0.3, 'Y': 0.68, 'V': -0.31}
aaAmpDi = {'A': 0, 'C': 0, 'D': 0, 'E': 1.27, 'F': 0, 'G': 0, 'H': 1.45, 'I': 0, 'K': 3.67, 'L': 0,
           'M': 0, 'N': 0, 'P': 0, 'Q': 1.25, 'R': 2.45, 'S': 0, 'T': 0, 'V': 0, 'W': 6.93, 'Y': 5.06}

aaPol1Di = {'A': 8.1, 'C': 5.5, 'D': 13, 'E': 12.3, 'F': 5.2, 'G': 9, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
              'M': 5.7, 'N': 11.6, 'P': 8, 'Q': 10.5, 'R': 10.5, 'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2}
aaPol2Di = {'A': 0.046, 'C': 0.128, 'D': 0.105, 'E': 0.151, 'F': 0.29, 'G': 0, 'H': 0.23, 'I': 0.186, 'K': 0.219, 'L': 0.186,
              'M': 0.221, 'N': 0.134, 'P': 0.131, 'Q': 0.18, 'R': 0.291, 'S': 0.062, 'T': 0.108, 'V': 0.14, 'W': 0.409, 'Y': 0.298}
aaNCISDi = {'A': 0.007187, 'C': -0.03661, 'D': -0.02382, 'E': -0.006802, 'F': 0.037552, 'G': 0.179052, 'H': -0.01069,
              'I': 0.021631, 'K': 0.017708, 'L': 0.051672, 'M': 0.002683, 'N': 0.005392, 'P': 0.239531, 'Q': 0.049211,
              'R': 0.043587, 'S': 0.004627,
            'T': 0.003352, 'V': 0.057004, 'W': 0.037977, 'Y': 0.0323599} # net charge index of side chain
aaSASADi = {'A': 1.181, 'C': 1.461, 'D': 1.587, 'E': 1.862, 'F': 2.228, 'G': 0.881, 'H': 2.025, 'I': 1.81, 'K': 2.258,
              'L': 1.931, 'M': 2.034, 'N': 1.655, 'P': 1.468, 'Q': 1.932, 'R': 2.56, 'S': 1.298, 'T': 1.525, 'V': 1.645,
              'W': 2.663, 'Y': 2.368}
aaBlosum62Di = {'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                  'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
                  'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                  'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
                  'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                  'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
                  'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
                  'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
                  'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
                  'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
                  'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
                  'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
                  'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
                  'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
                  'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
                  'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
                  'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
                  'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
                  'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
                  'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
                  'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1],
                  '*': [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]}
aaOHEDi = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'R': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'C': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'Q': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'E': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'G': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'H': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'F': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               '*': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

def GenerateInitialSequenceDataframe(seq_dict, tag_pos ='pos_', tag_neg ='neg_', islabeled=True):
    # print('There are ' + str(len(seq_dict)) + ' sequences')
    if len(seq_dict) == 0:
        return None

    df = pd.DataFrame(list(seq_dict.items()), columns=['Name', 'Sequence'])
    if islabeled is True:
        df['Label'] = df['Name'].map(lambda x: 1 if x.startswith(tag_pos) else 0)  # assign labels according to the tag in the Name column
    df['Length'] = df['Sequence'].map(lambda s: len(s))
    # print('Initial dataframe has been created')
    return df

def GenerateFeature_AAC(df):
    if 'Sequence' not in df.columns:
        return
    aaLi = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for residue in aaLi:
        labelName = 'AAC_' + residue
        df[labelName] = df['Sequence'].map(lambda s: s.count(residue) / len(s))
    #print('AAC feature columns are generated.')

def GenerateFeature_DPC(df):
    if 'Sequence' not in df.columns:
        return
    aaLi = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for residue1 in aaLi:
        for residue2 in aaLi:
            dp = residue1 + residue2
            labelName = 'DP_' + dp
            df[labelName] = df['Sequence'].map(lambda s: s.count(dp) / (len(s) - 1))
    #print('Di-peptide feature columns are generated.')

def GenerateFeature_CBP(df, kmer=8):
    if 'Sequence' not in df.columns:
        return
    aaLi = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(1, kmer + 1):
        pos = i - kmer - 1
        for residue in aaLi:
            labelName = 'C' + str(kmer) + 'mer_' + str(i) + '_' + residue
            df[labelName] = df['Sequence'].map(lambda s: 1 if len(s) > pos * (-1) and s[pos] == residue else 0)
    #print('C-terminal binary profile feature columns are generated.')

def GenerateFeature_NBP(df, kmer=8):
    if 'Sequence' not in df.columns:
        return
    aaLi = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(1, kmer + 1):
        pos = i - 1
        for residue in aaLi:
            labelName = 'N' + str(kmer) + 'mer_' + str(i) + '_' + residue
            df[labelName] = df['Sequence'].map(lambda s: 1 if i <= len(s) and s[pos] == residue else 0)
    #print('N-terminal binary profile feature columns are generated.')

def GenerateFeature_NetCharge(df):
    if 'Sequence' not in df.columns:
        return
    # according to Petr Klein, Minoru Kanehisa and Charles DeLisi, 1984
    df['NetCharge'] = df['Sequence'].map(lambda s: s.count('R') + s.count('K') - s.count('D') - s.count('E '))
    # print('Net charge has been summarized for each sequence.')

def GenerateFeature_AminoAcidCharge(df):
    if 'Sequence' not in df.columns:
        return
    # df['averagedCharge'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaChargeDi))
    df['Charge'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaChargeDi))

def GenerateFeature_ChargeComposition(df):
    if 'Sequence' not in df.columns:
        return
    df['PosChargeResPct'] = df['Sequence'].map(lambda s: (s.count('R') + s.count('K')) / len(s))
    df['NegChargeResPct'] = df['Sequence'].map(lambda s: (s.count('D') + s.count('E')) / len(s))
    df['NeutralResPct'] = 1 - df['PosChargeResPct'] - df['NegChargeResPct']
    # print('Percentages of positively charged, negatively charged, and non-charged residues are generated.')

def GenerateFeature_NetChargeIndexOfSideChain(df):
    if 'Sequence' not in df.columns:
        return
    df['NCIS'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaNCISDi))
    # print('Net charge indices of side chains are generated.')

def GenerateFeature_HydropathyIndex(df):
    # according to the values from Kyte and Doolittle, 1982           This is also the GRAVY index
    if 'Sequence' not in df.columns:
        return
    # df['HydropathyIndex'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaHydroIdxDi))
    df['Hydro'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaHydroIdxDi))
    # print('Hydropathy indices are generated.')

def GenerateFeature_Hydrophobicity(df):
    if 'Sequence' not in df.columns:
        return
    # df['Hydrophobicity'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaHyd1Di))
    df['Hyd1'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaHyd1Di))
    # print('Hydrophobicity indices are generated.')

def GenerateFeature_Hydrophilicity(df):
    if 'Sequence' not in df.columns:
        return
    df['Hydrophilicity'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaHyd2Di))
    # print('Hydrophilicity indices are generated.')

def GenerateFeature_pHatIsoelectricPoint(df):
    if 'Sequence' not in df.columns:
        return
    df['pI'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPIDi))
    # print('Averaged iso-electric points are generated.')

def GenerateFeature_pKaCOOH(df):
    if 'Sequence' not in df.columns:
        return
    df['pKa1'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPKa1Di))
    # print('Averaged pKa values for -COOH group are generated.')

def GenerateFeature_pKaNH2(df):
    if 'Sequence' not in df.columns:
        return
    df['pKa2'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPKa2Di))
    # print('Averaged pKa values for NH2- group are generated.')

def GenerateFeature_Volume(df):
    # according to the values from Joan Pontius, Jean Richelle and Shoshana J. Wodak, 1996
    if 'Sequence' not in df.columns:
        return
    #df['AAVolumeSum'] = df['Sequence'].map(lambda s: ComputeSummedIndex(s, aaVolumeDi))
    df['Vol'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaVolumeDi))

def GenerateFeature_VolumeSideChain(df):
    if 'Sequence' not in df.columns:
        return
    #df['VolSideChainSum'] = df['Sequence'].map(lambda s: ComputeSummedIndex(s, aaVSCDi))
    # df['VolSideChain'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaVSCDi))
    df['VSC'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaVSCDi))

def GenerateFeature_AmiphiphilicityIndex(df):
    # according to the values from Shigeki et al., 2002
    if 'Sequence' not in df.columns:
        return
    df['AmphiphilicityIndex'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaAmpDi))
    # print('Amphiphilicity indices are generated.')

def GenerateFeature_FreeEnergyTMH(df):
    if 'Sequence' not in df.columns:
        return
    # according to the values from Tara Hessa et al., 2005 Nature
    #df['FreeEnergyTMH_Sum'] = df['Sequence'].map(lambda s: ComputeSummedIndex(s, aaGappDi))
    # df['FreeEnergyTMH'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaGappDi))
    df['FEtmh'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaGappDi))

def GenerateFeature_Polarity(df):
    if 'Sequence' not in df.columns:
        return
    # df['Polarity'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPol1Di))
    df['Pol1'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPol1Di))

def GenerateFeature_Polarizability(df):
    if 'Sequence' not in df.columns:
        return
    # df['Polarizability'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPol2Di))
    df['Pol2'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaPol2Di))

def GenerateFeature_SolventAccessibleSurfaceArea(df):
    if 'Sequence' not in df.columns:
        return
    #df['SASASum'] = df['Sequence'].map(lambda s: ComputeSummedIndex(s, aaSASADi))
    df['SA'] = df['Sequence'].map(lambda s: ComputeAveragedIndex(s, aaSASADi))

def ComputeAveragedIndex(seq, aa_index_dict):
    index = 0
    for residue in seq:
        index += aa_index_dict[residue]
    index /= len(seq)
    return index

def ComputeSummedIndex(seq, aa_index_dict):
    index = 0
    for residue in seq:
        index += aa_index_dict[residue]
    return index

def GenerateFeatureTableGivenSeqDiAndFeatureLi(seq_di, feature_list, islabeled=True):
    df = GenerateInitialSequenceDataframe(seq_di, islabeled=islabeled)
    if 'NetCh' in feature_list:
        GenerateFeature_NetCharge(df)
    if 'Chg' in feature_list:
        GenerateFeature_AminoAcidCharge(df)
    if 'ChCmp' in feature_list:
        GenerateFeature_ChargeComposition(df)
    if 'NCIS' in feature_list:
        GenerateFeature_NetChargeIndexOfSideChain(df)
    if 'Hydro' in feature_list:
        GenerateFeature_HydropathyIndex(df)
    if 'Hyd1' in feature_list:
        GenerateFeature_Hydrophobicity(df)
    if 'Hyd2' in feature_list:
        GenerateFeature_Hydrophilicity(df)
    if 'pI' in feature_list:
        GenerateFeature_pHatIsoelectricPoint(df)
    if 'pKa1' in feature_list:
        GenerateFeature_pKaCOOH(df)
    if 'pKa2' in feature_list:
        GenerateFeature_pKaNH2(df)
    if 'Amp' in feature_list:
        GenerateFeature_AmiphiphilicityIndex(df)
    if 'Vol' in feature_list:
        GenerateFeature_Volume(df)
    if 'VSC' in feature_list:
        GenerateFeature_VolumeSideChain(df)
    if 'Pol1' in feature_list:
        GenerateFeature_Polarity(df)
    if 'Pol2' in feature_list:
        GenerateFeature_Polarizability(df)
    if 'SASA' in feature_list or 'SA' in feature_list:
        GenerateFeature_SolventAccessibleSurfaceArea(df)
    if 'FEtmh' in feature_list:
        GenerateFeature_FreeEnergyTMH(df)
    if 'AAC' in feature_list:
        GenerateFeature_AAC(df)
    if 'DPC' in feature_list:
        GenerateFeature_DPC(df)
    if 'CBP' in feature_list:
        GenerateFeature_CBP(df)
    if 'NBP' in feature_list:
        GenerateFeature_NBP(df)
    return df


def GenerateAAfeatureTable_GivenPropertyFeatures(given_features):
    feature_tab = []
    dimension_eachaa = 0
    for feature in given_features:
        if feature == 'Chg' or feature == 'Charge':
            feature_tab.append(aaChargeDi)
            dimension_eachaa += 1
        if feature == 'Hydro' or feature == 'Hydropathy':
            feature_tab.append(aaHydroIdxDi)
            dimension_eachaa += 1
        if feature == 'Hyd1' or feature == 'Hydrophobicity':
            feature_tab.append(aaHyd1Di)
            dimension_eachaa += 1
        if feature == 'Hyd2' or feature == 'Hydrophilicity':
            feature_tab.append(aaHyd2Di)
            dimension_eachaa += 1
        if feature == 'VSC' or feature == 'VolumeSideChain':
            feature_tab.append(aaVSCDi)
            dimension_eachaa += 1
        if feature == 'Vol' or feature == 'Volume':
            feature_tab.append(aaVolumeDi)
            dimension_eachaa += 1
        if feature == 'FEtmh' or feature == 'Gapp' or feature == 'FreeEnergy' or feature == 'FreeEnergyTMH':
            feature_tab.append(aaGappDi)
            dimension_eachaa += 1
        if feature == 'IEP' or feature == 'IsoElectricPoint' or feature == 'pI':
            feature_tab.append(aaPIDi)
            dimension_eachaa += 1
        if feature == 'pKa1':
            feature_tab.append(aaPKa1Di)
            dimension_eachaa += 1
        if feature == 'pKa2':
            feature_tab.append(aaPKa2Di)
            dimension_eachaa += 1
        if feature == 'Amp' or feature == 'Amphiphilicity':
            feature_tab.append(aaAmpDi)
            dimension_eachaa += 1
        if feature == 'Pol1' or feature == 'Polarity':
            feature_tab.append(aaPol1Di)
            dimension_eachaa += 1
        if feature == 'Pol2' or feature == 'Polarizability':
            feature_tab.append(aaPol2Di)
            dimension_eachaa += 1
        if feature == 'SASA' or feature == 'SA' or feature == 'SolventAccessibility' or feature == 'SolventAccessibilityOfSurfaceArea':
            feature_tab.append(aaSASADi)
            dimension_eachaa += 1
        if feature == 'NCIS' or feature == 'NetChargeIndexOfSideChain' or feature == 'NetChargeIndexSideChain':
            feature_tab.append(aaNCISDi)
            dimension_eachaa += 1
    df_aavector = pd.DataFrame(feature_tab)
    return dimension_eachaa, df_aavector

def ConstructSequenceProfile_N8merC8mer(df):
    if not {'Sequence', 'Label'}.issubset(df.columns):
        print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
        return
    GenerateFeature_NBP(df)
    GenerateFeature_CBP(df)
    data = df[df.columns[~df.columns.isin(['Name', 'Label', 'Sequence', 'Length'])]]
    df_data = pd.DataFrame(data)
    num_sample = df.shape[0]
    nparr_data = np.array(df_data)
    nparr_data = nparr_data.reshape((num_sample, 16, 20))
    return df['Label'], nparr_data

def ConstructSequenceProfile_N8mer(df):
    if not {'Sequence', 'Label'}.issubset(df.columns):
        print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
        return
    GenerateFeature_NBP(df)
    data = df[df.columns[~df.columns.isin(['Name', 'Label', 'Sequence', 'Length'])]]
    df_data = pd.DataFrame(data)
    num_sample = df.shape[0]
    nparr_data = np.array(df_data)
    nparr_data = nparr_data.reshape((num_sample, 8, 20))
    return df['Label'], nparr_data

def ConstructSequenceProfile_C8mer(df):
    if not {'Sequence', 'Label'}.issubset(df.columns):
        print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
        return
    GenerateFeature_CBP(df)
    data = df[df.columns[~df.columns.isin(['Name', 'Label', 'Sequence', 'Length'])]]
    df_data = pd.DataFrame(data)
    num_sample = df.shape[0]
    nparr_data = np.array(df_data)
    nparr_data = nparr_data.reshape((num_sample, 8, 20))
    return df['Label'], nparr_data


def ConstructSequenceProfile_GivenPropertyFeatures(df, given_features, padding_length=50, reshape=True, with_label=True):
    if not {'Sequence', 'Label'}.issubset(df.columns) and with_label is True:
        print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
        return None, None
    if len(given_features) == 0:
        print('No features are specified to construct the profile.')
        return None, None
    dimension_eachaa, df_aavector = GenerateAAfeatureTable_GivenPropertyFeatures(given_features)
    # print('Dataframe of aa vectors: ', df_aavector.shape)

    # Normalize feature table s.t. the mean of every feature across 20 amino acids is 0 and std is 1
    df_aavector_z = df_aavector.sub(df_aavector.mean(1), axis=0).div(df_aavector.std(1), axis=0)
    df_aavector_z['*'] = 0  # add the '*' column for padding values
    aavectorDi = {}         # convert dataframe of feature vector to a dictionary with key of a.a. and value of list
    for colname in df_aavector_z.columns:
        aavectorDi[colname] = list(df_aavector_z[colname])
    data = []
    for seq in df['Sequence']:
        vector = []
        padded_seq = pd.Series(list(seq.ljust(padding_length, '*')))
        for residue in padded_seq:
            vector = vector + aavectorDi[residue]
        data.append(vector)
    df_data = pd.DataFrame(data)
    num_sample = df.shape[0]
    nparr_data = np.array(df_data)
    if reshape is True:
        nparr_data = nparr_data.reshape((num_sample, padding_length, dimension_eachaa))

    if with_label is True:
        return np.array(df['Label']), nparr_data
    else:
        return None, nparr_data

# def ConstructSequenceProfile_BLOSUM62(df, padding_length=50, normalize=True, reshape=False):
#     if not {'Sequence', 'Label'}.issubset(df.columns):
#         print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
#         return
#
#     aa_blosum62_std_di = dict(aaBlosum62Di)
#     df_aa_bls62std = pd.DataFrame(aa_blosum62_std_di)
#     df_aa_bls62std = df_aa_bls62std.sub(df_aa_bls62std.mean(1), axis=0).div(df_aa_bls62std.std(1), axis=0)
#     # df_aa_bls62std.columns = ['blsvec_A', 'blsvec_R', 'blsvec_N', 'blsvec_D', 'blsvec_C', 'blsvec_Q', 'blsvec_E',
#     #                           'blsvec_G', 'blsvec_H', 'blsvec_I', 'blsvec_L', 'blsvec_K', 'blsvec_M', 'blsvec_F',
#     #                           'blsvec_P', 'blsvec_S', 'blsvec_T', 'blsvec_W', 'blsvec_Y', 'blsvec_V', 'blsvec_X', 'blsvec_*']
#     aa_blsvec_di = aaBlosum62Di  # if normalize if False, then use the original aaBlosum62Di as the feature vectors
#     if normalize is True:
#         aa_blsvec_di = {}
#         for colname in df_aa_bls62std.columns:
#             aa_blsvec_di[colname] = list(df_aa_bls62std[colname])
#
#     data = []
#     for seq in df['Sequence']:
#         vector = []
#         padded_seq = pd.Series(list(seq.ljust(padding_length, '*')))
#         for residue in padded_seq:
#             vector = vector + aa_blsvec_di[residue]
#         data.append(vector)
#     df_data = pd.DataFrame(data)
#     nparr_data = np.array(df_data)
#     if reshape is True:
#         num_sample = df.shape[0]
#         nparr_data = nparr_data.reshape((num_sample, padding_length, 20))
#     return df['Label'], nparr_data
#
# def ConstructSequenceProfile_OneHotEncoding(df, padding_length=50, reshape=False):
#     if not {'Sequence', 'Label'}.issubset(df.columns):
#         print('Input dataframe needs contains column of \'Sequence\' and column of \'Label\'.')
#         return
#     data = []
#     for seq in df['Sequence']:
#         vector = []
#         padded_seq = pd.Series(list(seq.ljust(padding_length, '*')))
#         for residue in padded_seq:
#             vector = vector + aaOHEDi[residue]
#         data.append(vector)
#     df_data = pd.DataFrame(data)
#     nparr_data = np.array(df_data)
#     if reshape is True:
#         num_sample = df.shape[0]
#         nparr_data = nparr_data.reshape((num_sample, padding_length, 20))
#     return df['Label'], nparr_data