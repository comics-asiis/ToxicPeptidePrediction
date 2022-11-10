# SNOP means Silver Needle On Peptides
# This program loads the pickle files of pre-trained toxic peptide classifiers and predicts the input peptide sequences
import sys
import os
import errno
import modelwizard

if len(sys.argv) > 2:
    input_fastapath = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.isfile(input_fastapath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_fastapath)
    if not os.path.isdir(output_directory):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), output_directory)
    print('Fasta path:', input_fastapath)
    print('Output directory:', output_directory)
    output_filename = modelwizard.Load4modelsAndPredict(input_fastapath, output_dir = output_directory)
    print('Result file name:', output_filename)

elif len(sys.argv) > 1:
    # argv[1] should be the input fasta file path
    input_fastapath = sys.argv[1]
    if not os.path.isfile(input_fastapath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_fastapath)
    print('Fasta path:', input_fastapath)
    print('Output directory:', os.path.dirname(os.path.realpath(__file__)))
    output_filename = modelwizard.Load4modelsAndPredict(input_fastapath)
    print('Result file name:', output_filename)

else:
    print('No input detected')
    print('Run ToxinPred dataset')
    print('Output directory:', os.path.dirname(os.path.realpath(__file__)))
    output_filename = modelwizard.Load4modelsAndPredict('data/toxinpred_indepdataset.fasta')
    print('Result file name:', output_filename)
