

import pypianoroll
import numpy as np
import os
from metrics import metrics_function
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import json


def batch_metrics(folder, verbose=True):
    # N = len(tracks)  # N may not be 4 depending on the mid file
    num_midi = 4

    # batch_empty_beat_rate = np.zeros((N,))
    # batch_qualified_note_rate = np.zeros((N,))
    # batch_n_pitch_class_used = np.zeros((N,))
    # batch_TD = np.zeros((N,N))

    batch_empty_beat_rate = []
    batch_qualified_note_rate = []
    batch_n_pitch_class_used = []
    batch_TD = []

    total_files = 0

    for (_, _, files) in os.walk(folder):
        total_files = len(files)
        if total_files > 2:
            files = files[:2]
        for file in files:

            if file.endswith('.mid'):
                mid_path = folder + '/' + file
                each = metrics_function(mid_path, verbose=verbose)

                # batch_empty_beat_rate += each['empty_beat_rate']
                # batch_qualified_note_rate += each['qualified_note_rate']
                # batch_n_pitch_class_used += each['n_pitch_class_used']
                # batch_TD += each['TD']

                batch_empty_beat_rate.append(list(each['empty_beat_rate']))
                batch_qualified_note_rate.append(list(each['qualified_note_rate']))
                batch_n_pitch_class_used.append(list(each['n_pitch_class_used']))
                batch_TD.append(list(each['TD']))


    if total_files == 0:
        return

    return {'batch_empty_beat_rate': batch_empty_beat_rate,
            'batch_qualified_note_rate': batch_qualified_note_rate,
            'batch_n_pitch_class_used': batch_n_pitch_class_used,
            'batch_TD': batch_TD,
            }
def jsontoDict(res):
    with open(res,'r') as j:
        contents = json.loads(j.read())
        # print(contents)
    # jsonkeys = list(contents)
    # print(jsonkeys[:4])
    # res = {r:jsonf[r] for r in jsonkeys[:4]}
    return contents
def avg_permidi(res):
    #average across midi files
    avg_res = {}
    for r in res:
        data = res[r]
        avg_res[r] = np.mean(np.array(data), axis = 1 )
    return avg_res
def avg_pertrack(res):
    #average across  each track
    avg_res = {}
    for r in res:
        data = res[r]
        mean =  np.mean(np.array(data), axis = 0 )
        ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=np.std(data))
        mean =  np.mean(np.array(data), axis = 0)
        std = np.std(data)
        avg_res[r] = (ci, mean, std)
    return avg_res


def prob_metrics(res):# takes in json file
    if type(res) is dict:#input is dictionary
        res = res
        avg_midi = avg_permidi(res) #create graphs
        avg_track = avg_pertrack(res) #create table
    else: #input is json file
        res = jsontoDict(res)
        avg_midi = avg_permidi(res) #create graphs
        avg_track = avg_pertrack(res) #create table


    print()
    print("avg_midi", avg_midi)
    print()
    print("avg_track", avg_track)
    print()
    # track_metrics = {}
    #
    # #metrics per track
    # for t in avg_track:
    #     track_metrics[t] =

    fig, axs = plt.subplots(1, 4)
    prob = {}
    bars1 = []
    yer1 = []
    #get confidence interval, mean, standard deviation
    i = 0
    print(len(avg_midi))
    for r in avg_midi:
        axN = axs[i]
        data = avg_midi[r]
        #print("d" , data)
        bars1.append(np.mean(data))
        yer1.append(2*np.std(data))
        ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=np.std(data))
        prob[r] = (ci, np.mean(data), np.std(data))
        #plot pdf of statistic
        sns.histplot(data, kde = True, label='samples', ax = axN, bins = 10)
        # calculate the pdf
        x0, x1 = axN.get_xlim()  # extract the endpoints for the x-axis
        x_pdf = np.linspace(x0, x1, 100)
        y_pdf = norm.pdf(x_pdf, loc=np.mean(data), scale = np.std(data))

        axN.plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')
        axN.legend()
        i+=1

    # print("bars1", bars1)
    print(prob)
    #plot confidence intervals on bar plot
    bars = ['ebr', 'qnr', 'np', 'td']
    plt.figure()
    y_bars = np.arange(len(bars))
    plt.bar(y_bars, bars1, yerr = yer1, edgecolor = 'black', capsize=7)
    plt.xticks([r for r in range(len(bars1))], bars)
    plt.show()
    return prob
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
# folder = './test_midi/'
# # print(batch_metrics(folder))
# #res = batch_metrics(folder)
# jsonfile = 'batch_metrics_npz.json'
# #f = open('batch_metrics_npz.json')
# # jsonf = json.load(f)
# # jsonkeys = list(jsonf)
# # print(jsonkeys[-4:])
# # res = {r:jsonf[r] for r in jsonkeys[-4:]}
# # print("json",res)
# print(prob_metrics(jsonfile))
