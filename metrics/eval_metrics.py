import metrics
from batch_metrics import batch_metrics, prob_metrics
import numpy as np
import json

####################
# eval the npz
####################
def batch_real_data():
    # result = metrics.metrics_function("../output_midi/jsb-npz", verbose=False)
    # print(result)
    folder_npz = './output_midi/jsb-npz/jsb-npz/'
    res_npz = batch_metrics(folder_npz)
    print('===========================================')
    print('================= npz =====================')
    print('===========================================')
    # print(res_npz)

    # json_npz = open("./batch_metrics_npz.json", "w")
    # json_npz = json.dumps(res_npz, indent = 4)

    with open('batch_metrics_npz.json', 'w') as file1:
        # file1.write(json.dumps(res_npz))
        json.dump(res_npz, file1, separators=(',', ':'))

################################
# eval the randomized generator
################################
def batch_random_data():
    folder_random = './output_midi/randomized-midi/randomized-midi/'
    res_rand = batch_metrics(folder_random)
    print('===========================================')
    print('================= random ==================')
    print('===========================================')
    print(res_rand)

    # json_rand = open("./batch_metrics_rand.json", "w")
    # json_rand = json.dumps(res_rand, indent = 4)

    with open('batch_metrics_rand.json', 'w') as file2:
        # file2.write(json.dumps(res_rand))
        json.dump(res_rand, file2, separators=(',', ':'))

def eval_data(res):
    r = prob_metrics(res)

if __name__ == "__main__":
    batch_real_data()
    batch_random_data()
    # f = open('batch_metrics_npz.json')
    # jsonf = json.load(f)
    # jsonkeys = list(jsonf)
    # print(jsonkeys[:4])
    # res = {r:jsonf[r] for r in jsonkeys[:4]}

    eval_data('batch_metrics_npz.json')
    eval_data('batch_metrics_rand.json')
    #print(prob_metrics(res_npz))
    ################################
    # eval the randomized generator
    ################################
    #print(prob_metrics(res_rand))
    ################################
    # eval our generated outputs
    ################################

