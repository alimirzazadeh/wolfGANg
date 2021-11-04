wolfGANg
=========

to run the training: python train.py
saves a MIDI file that is generated from the generator at the end
shape of each dataset element and the produced data output: [4, 2, 16, 84]
postProcess then converts the output into a MIDI file

MIDI file then can be listened to.

In order to run our metrics, we must use pypianoroll to run the evaluation metrics directly on the MIDI file

--currently trying to setup server with this github repo so we can all run it
