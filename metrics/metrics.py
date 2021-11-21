

import pypianoroll
import numpy as np
import matplotlib

def metrics_function(mid_path, verbose=True):
    multitrack = pypianoroll.read(mid_path)
    tracks = multitrack.tracks
    N = len(tracks) # N may not be 4 depending on the mid file
    N = 4

    # 1. empty_beat_rate
    empty_beat_rate = np.zeros((N,))
    # 2. qualified_note_rate
    qualified_note_rate = np.zeros((N,))
    # 3. n_pitch_class_used
    n_pitch_class_used = np.zeros((N,))
    # 4. TD
    TD = np.zeros((N,N))

    if verbose:
        print("Multitrack information: \n")
        print("========================\n")
        print(multitrack)
        print("========================\n")
        print("Track information: \n")
        print("========================\n")
        print("tracks= ", tracks)

    for i in range(N): # we can also force it to be 4 if we only need to first 4 tracks
        curr_pianoroll = tracks[i].pianoroll
        empty_beat_rate[i] = pypianoroll.metrics.empty_beat_rate(curr_pianoroll, 24)
        qualified_note_rate[i] = pypianoroll.qualified_note_rate(curr_pianoroll) # , threshold=3
        n_pitch_class_used[i] = pypianoroll.n_pitch_classes_used(curr_pianoroll)

        for j in range(N):
            TD[i][j] = pypianoroll.tonal_distance(pianoroll_1=tracks[i].pianoroll, pianoroll_2=tracks[j].pianoroll,
                                                  resolution=24) # resolution=24

    return {'empty_beat_rate': empty_beat_rate, 'qualified_note_rate': qualified_note_rate,
            'n_pitch_class_used':n_pitch_class_used, 'TD':TD}

result = metrics_function("./test_midi/midi_long1.mid", verbose=False)
# result = metrics_function("myexample2.mid", verbose=False)
print(result)
