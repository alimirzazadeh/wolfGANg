

import pypianoroll
import numpy as np
import matplotlib
"""Objective metrics for piano rolls.

Functions
---------

- drum_in_pattern_rate
- empty_beat_rate
- in_scale_rate
- n_pitch_classes_used
- n_pitches_used
- pitch_range
- pitch_range_tuple
- polyphonic_rate
- qualified_note_rate
- tonal_distance

"""
from math import nan
from typing import Sequence, Tuple

import numpy as np
from numpy import ndarray

__all__ = [
    "drum_in_pattern_rate",
    "empty_beat_rate",
    "in_scale_rate",
    "n_pitch_classes_used",
    "n_pitches_used",
    "pitch_range",
    "pitch_range_tuple",
    "polyphonic_rate",
    "qualified_note_rate",
    "tonal_distance",
]


def _to_chroma(pianoroll: ndarray) -> ndarray:
    """Return the unnormalized chroma features."""
    reshaped = pianoroll[:, :120].reshape(-1, 12, 10)
    reshaped[..., :8] += pianoroll[:, 120:].reshape(-1, 1, 8)
    return np.sum(reshaped, -1)








def qnr(pianoroll: ndarray, threshold: float = 2) -> float:
    r"""Return the ratio of the number of the qualified notes.

    The qualified note rate is defined as the ratio of the number of
    qualified notes (notes longer than `threshold`, in time steps) to
    the total number of notes. Return NaN if no note is found.

    .. math::
        qualified\_note\_rate = \frac{
            \#(notes\_longer\_than\_the\_threshold)
        }{
            \#(notes)
        }

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.
    threshold : int
        Threshold of note length to count into the numerator.

    Returns
    -------
    float
        Qualified note rate.

    References
    ----------
    1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
       "MuseGAN: Multi-track sequential generative adversarial networks
       for symbolic music generation and accompaniment," in Proceedings
       of the 32nd AAAI Conference on Artificial Intelligence (AAAI),
       2018.

    """
    # if np.issubdtype(pianoroll.dtype, np.bool_):
    #     pianoroll = pianoroll.astype(np.uint8)
    # padded = np.pad(pianoroll, ((1, 1), (0, 0)), "constant")
    # diff = np.diff(padded, axis=0).reshape(-1)
    # onsets = (diff > 0).nonzero()[0]
    # if len(onsets) < 1:
    #     return nan
    # offsets = (diff < 0).nonzero()[0]
    # n_qualified_notes = np.count_nonzero(offsets - onsets >= threshold)
    # return n_qualified_notes / len(onsets)


    # my code
    padded = np.pad(pianoroll.astype(np.int), ((1, 1), (0, 0)), "constant")
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1,)
    onsets = (flattened > 0).nonzero()[0]
    offsets = (flattened < 0).nonzero()[0]
    #n_qualified_notes = np.count_nonzero(offsets - onsets >= threshold).sum()
    n_qualified_notes = (offsets - onsets >= threshold).sum()
    return n_qualified_notes / len(onsets)



def _get_tonal_matrix(r1, r2, r3) -> ndarray:
    """Return a tonal matrix for computing the tonal distance."""
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7.0 / 6.0) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7.0 / 6.0) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3.0 / 2.0) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3.0 / 2.0) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2.0 / 3.0) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2.0 / 3.0) * np.pi)
    return tonal_matrix


def _to_tonal_space(
        pianoroll: ndarray, resolution: int, tonal_matrix: ndarray
) -> ndarray:
    """Return the tensor in tonal space (chroma normalized per beat)."""
    # # original code
    # beat_chroma = _to_chroma(pianoroll).reshape((-1, resolution, 12))
    # beat_chroma = beat_chroma / beat_chroma.sum(2, keepdims=True)

    # # my added code 1
    # beat_chroma = beat_chroma.reshape(-1, beat_chroma.shape[-1])

    # # my added code 2
    beat_chroma = _to_chroma(pianoroll)
    beat_chroma = beat_chroma / beat_chroma.sum()

    return np.matmul(tonal_matrix, beat_chroma.T).T


def tonal_distance(
        pianoroll_1: ndarray,
        pianoroll_2: ndarray,
        resolution: int,
        radii: Sequence[float] = (1.0, 1.0, 0.5),
) -> float:
    """Return the tonal distance [1] between the two input piano rolls.

    Parameters
    ----------
    pianoroll_1 : ndarray
        First piano roll to evaluate.
    pianoroll_2 : ndarray
        Second piano roll to evaluate.
    resolution : int
        Time steps per beat.
    radii : tuple of float
        Radii of the three tonal circles (see Equation 3 in [1]).

    References
    ----------
    1. Christopher Harte, Mark Sandler, and Martin Gasser, "Detecting
       harmonic change in musical audio," in Proceedings of the 1st ACM
       workshop on Audio and music computing multimedia, 2006.

    """
    assert len(pianoroll_1) == len(
        pianoroll_2
    ), "Input piano rolls must have the same length."

    r1, r2, r3 = radii
    tonal_matrix = _get_tonal_matrix(r1, r2, r3)
    mapped_1 = _to_tonal_space(pianoroll_1, resolution, tonal_matrix)
    mapped_2 = _to_tonal_space(pianoroll_2, resolution, tonal_matrix)
    return np.linalg.norm(mapped_1 - mapped_2)

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
    # 4. TD : AB, AC, AD, BC, BD, CD
    TD = np.zeros((6,))

    if verbose:
        print("Multitrack information: \n")
        print("========================\n")
        print(multitrack)
        print("========================\n")
        print("Track information: \n")
        print("========================\n")
        print("tracks= ", tracks)
    td_i = 0
    for i in range(N): # we can also force it to be 4 if we only need to first 4 tracks
        curr_pianoroll = tracks[i].pianoroll
        empty_beat_rate[i] = pypianoroll.metrics.empty_beat_rate(curr_pianoroll, 24)
        qualified_note_rate[i] = qnr(curr_pianoroll,2) # , threshold=3
        n_pitch_class_used[i] = pypianoroll.n_pitch_classes_used(curr_pianoroll)



        for j in range(i+1,N):
            TD[td_i] = tonal_distance(pianoroll_1=tracks[i].pianoroll, pianoroll_2=tracks[j].pianoroll,
                                      resolution=24) # resolution=24
            td_i +=1

    return {'empty_beat_rate': empty_beat_rate, 'qualified_note_rate': qualified_note_rate,
            'n_pitch_class_used':n_pitch_class_used, 'TD':TD}
#
# result = metrics_function("./test_midi/midi_long1.mid", verbose=False)
# # result = metrics_function("myexample2.mid", verbose=False)
# print(result)
