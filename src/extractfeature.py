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
              'W': 5.89, 'Y': 5.66}   # isoelectric point
aaPKa1Di = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18,
              'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32,
              'W': 2.83, 'Y': 2.2}   # pKa1: carboxyl group (-COOH)
aaPKa2Di = {'A': 9.69, 'C': 8.18, 'D': 9.6, 'E': 9.67, 'F': 9.13, 'G': 9.6, 'H': 9.17, 'I': 9.6, 'K': 8.95,
              'L': 9.6, 'M': 9.21, 'N': 8.8, 'P': 10.6, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.1, 'V': 9.62,
              'W': 9.39, 'Y': 9.11}   # pKa2: ammonium group (NH2-)
aaVolumeDi = {'A': 91.5, 'R': 196.1, 'N': 138.3, 'D': 135.2, 'C': 102.4, 'Q': 156.4, 'E': 154.6, 'G': 67.5, 'H': 163.2,
              'I': 162.6, 'L': 163.4, 'K': 162.5, 'M': 165.9, 'F': 198.8, 'P': 123.4, 'S': 102.0, 'T': 126.0,
              'W': 237.2, 'Y': 209.8, 'V': 138.4}
aaVSCDi = {'A': 27.5, 'C': 44.6, 'D': 40, 'E': 62, 'F': 115.5, 'G': 0, 'H': 79, 'I': 93.5, 'K': 100, 'L': 93.5,
              'M': 94.1, 'N': 58.7, 'P': 41.9, 'Q': 80.7, 'R': 105, 'S': 29.3, 'T': 51.3, 'V': 71.5, 'W': 145.5, 'Y': 117.3}
aaGappDi = {'A': 0.11, 'R': 2.58, 'N': 2.05, 'D': 3.49, 'C': -0.13, 'Q': 2.36, 'E': 2.68, 'G': 0.74, 'H': 2.06, 'I': -0.6, 'L': -0.55, 'K': 2.71, 'M': -0.1, 'F': -0.32, 'P': 2.23, 'S': 0.84, 'T': 0.52, 'W': 0.3, 'Y': 0.68, 'V': -0.31}

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


def generate_initial_dataframe(seq_dict):
    if len(seq_dict) == 0:
        return None
    df = pd.DataFrame(list(seq_dict.items()), columns=['Name', 'Sequence'])
    df['Length'] = df['Sequence'].map(lambda s: len(s))
    return df


def generate_feature_aac(df):
    aatype = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for residue in aatype:
        feat = 'AAC_' + residue
        df[feat] = df['Sequence'].map(lambda s: s.count(residue) / len(s))


def generatefeature_dpc(df):
    aatype = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for residue1 in aatype:
        for residue2 in aatype:
            dp = residue1 + residue2
            feat = 'DP_' + dp
            df[feat] = df['Sequence'].map(lambda s: s.count(dp) / (len(s) - 1))


def generatefeature_cbp(df, kmer=8):
    aatype = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(1, kmer + 1):
        pos = i - kmer - 1
        for residue in aatype:
            feat = 'C' + str(kmer) + 'mer_' + str(i) + '_' + residue
            df[feat] = df['Sequence'].map(lambda s: 1 if len(s) > pos * (-1) and s[pos] == residue else 0)


def generatefeature_nbp(df, kmer=8):
    aatype = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(1, kmer + 1):
        pos = i - 1
        for residue in aatype:
            feat = 'N' + str(kmer) + 'mer_' + str(i) + '_' + residue
            df[feat] = df['Sequence'].map(lambda s: 1 if i <= len(s) and s[pos] == residue else 0)


def generatefeature_chg(df):
    df['Charge'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaChargeDi))


def generatefeature_ncis(df):
    df['NCIS'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaNCISDi))


def generatefeature_hydro(df):
    df['Hydro'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaHydroIdxDi))


def generatefeature_hyd1(df):
    df['Hyd1'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaHyd1Di))


def generatefeature_hyd2(df):
    df['Hyd2'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaHyd2Di))


def generatefeature_iep(df):
    df['pI'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaPIDi))


def generatefeature_pka1(df):
    df['pKa1'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaPKa1Di))


def generatefeature_pka2(df):
    df['pKa2'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaPKa2Di))


def generatefeature_vol(df):
    df['Vol'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaVolumeDi))


def generatefeature_vsc(df):
    df['VSC'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaVSCDi))


def generatefeature_fetmh(df):
    df['FEtmh'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaGappDi))


def generatefeature_pol1(df):
    df['Pol1'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaPol1Di))


def generatefeature_pol2(df):
    df['Pol2'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaPol2Di))


def generatefeature_sa(df):
    df['SA'] = df['Sequence'].map(lambda s: cal_avg_index(s, aaSASADi))


def cal_avg_index(seq, aa_index_dict):
    index = 0
    for residue in seq:
        index += aa_index_dict[residue]
    index /= len(seq)
    return index


def generatefeaturetable(seq_di, feature_list):
    df = generate_initial_dataframe(seq_di)
    if 'Chg' in feature_list:
        generatefeature_chg(df)
    if 'NCIS' in feature_list:
        generatefeature_ncis(df)
    if 'Hydro' in feature_list:
        generatefeature_hydro(df)
    if 'Hyd1' in feature_list:
        generatefeature_hyd1(df)
    if 'Hyd2' in feature_list:
        generatefeature_hyd2(df)
    if 'pI' in feature_list:
        generatefeature_iep(df)
    if 'pKa1' in feature_list:
        generatefeature_pka1(df)
    if 'pKa2' in feature_list:
        generatefeature_pka2(df)
    if 'Vol' in feature_list:
        generatefeature_vol(df)
    if 'VSC' in feature_list:
        generatefeature_vsc(df)
    if 'Pol1' in feature_list:
        generatefeature_pol1(df)
    if 'Pol2' in feature_list:
        generatefeature_pol2(df)
    if 'SASA' in feature_list or 'SA' in feature_list:
        generatefeature_sa(df)
    if 'FEtmh' in feature_list:
        generatefeature_fetmh(df)
    if 'AAC' in feature_list:
        generate_feature_aac(df)
    if 'DPC' in feature_list:
        generatefeature_dpc(df)
    if 'CBP' in feature_list:
        generatefeature_cbp(df)
    if 'NBP' in feature_list:
        generatefeature_nbp(df)
    return df


def generate_aapcptable(given_features):
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


def constructprofile_givenpropertyfeatures(df, given_features, padding_length=50, reshape=True):
    if len(given_features) == 0:
        print('No features are specified to construct the profile.')
        return None, None
    dimension_eachaa, df_aavector = generate_aapcptable(given_features)
    df_aavector_z = df_aavector.sub(df_aavector.mean(1), axis=0).div(df_aavector.std(1), axis=0)
    df_aavector_z['*'] = 0
    aavector_di = {}
    for colname in df_aavector_z.columns:
        aavector_di[colname] = list(df_aavector_z[colname])
    data = []
    for seq in df['Sequence']:
        vector = []
        padded_seq = pd.Series(list(seq.ljust(padding_length, '*')))
        for residue in padded_seq:
            vector = vector + aavector_di[residue]
        data.append(vector)
    df_data = pd.DataFrame(data)
    num_sample = df.shape[0]
    nparr_data = np.array(df_data)
    if reshape is True:
        nparr_data = nparr_data.reshape((num_sample, padding_length, dimension_eachaa))
    return None, nparr_data
