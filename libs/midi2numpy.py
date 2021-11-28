import music21
import numpy as np
from mido import MidiFile
from ipdb import set_trace as bp

# from pathlib import Path

BPB = 16 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (0, 1000000)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

SAMPLE_FREQ = 4
NOTE_SIZE = 84
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1 # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)

def file2stream(fp):
	
	if isinstance(fp, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(fp)
	return music21.converter.parse(fp)

def stream2chordarr(s, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    "Converts music21.Stream to 1-hot numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    highest_time = max(s.flat.getElementsByClass('Note').highestTime, s.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq)
    score_arr = np.zeros((maxTimeStep, len(s.parts), NOTE_SIZE))

    def note_data(pitch, note):
        return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))

    for idx,part in enumerate(s.parts):
        notes=[]
        for elem in part.flat:
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))
                
        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
        for n in notes_sorted:
            if n is None: continue
            pitch,offset,duration = n
            if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
            score_arr[offset, idx, pitch] = duration
            score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
    
    score_arr_reshaped = np.reshape(score_arr,(4,2,16,84))
    final_np = np.expand_dims(score_arr_reshaped, axis=0)
    return final_np
def midiToNumpy(filepath):
    arr = stream2chordarr(file2stream(filepath))
    # arr[arr > 0] = 1
    # arr[arr < 0] = 0
    return arr


# from music21 import midi
# from music21 import converter
# from music21 import note, stream, duration, tempo
# def binarise_output(output):
#     max_pitches=np.argmax(output, axis=-1)
#     return max_pitches
# def postProcess(max_pitches, n_tracks=4, n_bars=2, n_steps_per_bar=16,):
#     parts = stream.Score()
#     parts.append(tempo.MetronomeMark(number= 66))
#     # bp()
#     # print(max_pitches[0].shape)
#     midi_note_score = [max_pitches[i].reshape([n_bars * n_steps_per_bar, n_tracks]) for i in range(max_pitches.shape[0])]
#     midi_note_score = np.vstack(midi_note_score)
#     for i in range(n_tracks):
#         last_x = int(midi_note_score[:,i][0])
#         s= stream.Part()
#         dur = 0
#         for idx, x in enumerate(midi_note_score[:, i]):
#             x = int(x)
#             if (x != last_x or idx % 4 == 0) and idx > 0:
#                 n = note.Note(last_x)
#                 n.duration = duration.Duration(dur)
#                 s.append(n)
#                 dur = 0
#             last_x = x
#             dur = dur + 0.25
#         n = note.Note(last_x)
#         n.duration = duration.Duration(dur)
#         s.append(n)
#         parts.append(s)
#     return parts

# print(res.shape)
# music_data = postProcess(binarise_output(res))
# filename = 'test1.midi'
# music_data.write('midi', fp=filename)