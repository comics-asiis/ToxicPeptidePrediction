import os

def readfasta(fastapath, ul_seqlen=50):
    if os.path.isfile(fastapath):
        fp = open(fastapath, 'r')
    else:
        print('The fasta file is not found')
        return None

    seq_dict = dict()
    notprocessed_seq_dict = dict()
    current_header = ''
    current_seq = ''
    line = fp.readline()
    if line.startswith('>'):
        with_header = True
    else:
        with_header = False

    if with_header is True:
        while line:
            if line.startswith('>'):
                if current_header != '' and current_header not in seq_dict and len(current_seq) <= ul_seqlen:
                    seq_dict[current_header] = current_seq
                elif current_header != '' and current_header not in notprocessed_seq_dict and len(current_seq) > ul_seqlen:
                    notprocessed_seq_dict[current_header] = current_seq
                current_header = line[1:].replace('\n', '')
                current_seq = ''
            else:
                current_seq += line.rstrip()
            line = fp.readline()
        if current_header != '' and current_header not in seq_dict and len(current_seq) <= ul_seqlen:
            seq_dict[current_header] = current_seq
        elif current_header != '' and current_header not in notprocessed_seq_dict and len(current_seq) > ul_seqlen:
            notprocessed_seq_dict[current_header] = current_seq

    else:
        seq_count = 0
        while line:
            seq_count += 1
            sudoheader = 'Sequence_' + str(seq_count)
            seq = line.replace('\n', '').rstrip()
            if len(seq) <= ul_seqlen:
                seq_dict[sudoheader] = seq
            else:
                notprocessed_seq_dict[sudoheader] = seq
            line = fp.readline()
    fp.close()
    return seq_dict, notprocessed_seq_dict

def write_df_2_csv(df, csvpath):
    df.to_csv(csvpath, index=False)