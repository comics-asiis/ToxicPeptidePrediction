""" This script is where ToxTeller starts to execute the prediction workflow """
import sys
import os
import errno
import modelwizard
from datetime import datetime

if __name__ == '__main__':
    now1 = datetime.now()
    time1 = now1.strftime("%Y-%m-%d@%H%M%S")
    print('Time:')
    print(time1)

    if len(sys.argv) > 1:
        # argv[1] is the path of input fasta
        input_fastapath = sys.argv[1]
        if not os.path.isfile(input_fastapath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), input_fastapath)
        print('Fasta path:', input_fastapath)
        print('Output directory:', os.path.dirname(os.path.realpath(__file__)))
        output_filename = modelwizard.load_4models_predict(input_fastapath)
        print('Result file name:', output_filename)

    else:
        print('No input detected')

    now2 = datetime.now()
    time2 = now2.strftime("%Y-%m-%d@%H%M%S")
    print('\nTime when finished:')
    print(time2)
