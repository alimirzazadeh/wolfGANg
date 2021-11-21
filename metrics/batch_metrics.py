

import pypianoroll
import numpy as np
import os
from metrics import metrics_function


def batch_metrics(folder, verbose=True):
    # N = len(tracks)  # N may not be 4 depending on the mid file
    N = 4

    batch_empty_beat_rate = np.zeros((N,))
    batch_qualified_note_rate = np.zeros((N,))
    batch_n_pitch_class_used = np.zeros((N,))
    batch_TD = np.zeros((N,N))
    total_files = 0

    for (_, _, files) in os.walk(folder):
        total_files = len(files)
        for file in files:

            if file.endswith('.mid'):
                print("file= ", file)
                mid_path = folder + '/' + file
                each = metrics_function(mid_path, verbose=False)
                print("each= ", each)
                batch_empty_beat_rate += each['empty_beat_rate']
                batch_qualified_note_rate += each['qualified_note_rate']
                batch_n_pitch_class_used += each['n_pitch_class_used']
                batch_TD += each['TD']


    if total_files == 0:
        return

    return {'batch_empty_beat_rate': batch_empty_beat_rate / total_files,
            'batch_qualified_note_rate': batch_qualified_note_rate / total_files,
            'batch_n_pitch_class_used': batch_n_pitch_class_used / total_files,
            'batch_TD': batch_TD / total_files}


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
folder = './test_midi/'
print(batch_metrics(folder))