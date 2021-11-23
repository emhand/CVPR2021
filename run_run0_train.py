

from Train import Train, Evaluate
'''

def run_01_0():
    Train(0, "datasets_combined_separated", "revised_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "datasets_combined_separated", "enlarged_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "ourcar", "bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)

def run_01_1():
    Train(0, "ourcar", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True)

    Train(0, "datasets_combined_separated", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)


def run_02_0():
    Train(0, "datasets_combined_separated", "enlarged_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "webcaricature", "enlarged_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)

def run_02_1():
    Train(0, "webcaricature", "revised_bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)
    Train(0, "ourcar", "enlarged_bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "webcaricature", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)

def run_03_0():
    Train(0, "webcaricature", "enlarged_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)
    Train(0, "ourcar", "enlarged_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=0,
          debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "datasets_combined_separated", "bounding_box_based", lr=0.000001, device="cuda:0", finetune=True,
          foldNum=0,debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)

def run_03_1():
    Train(0, "datasets_combined_separated", "revised_bounding_box_based", lr=0.000001, device="cuda:1", finetune=True,
          foldNum=0, debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
    Train(0, "webcaricature", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True,
          foldNum=0,debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)
def run_07_0():
    Train(0, "webcaricature", "revised_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True,
          foldNum=0,debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
          pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=False)
    Train(0, "ourcar", "revised_bounding_box_based", lr=0.000001, device="cuda:0", finetune=True,
          foldNum=0, debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
          pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=False)
'''
def run_01_0():
    '''
    #revised combined restricted all -> finished
    for i in range (1, 11):
        Train(i, "datasets_combined_separated", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=139)
    #bbox combined restricted mmo -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd = 93)
    #revised webcar_clean unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd = 15)
    '''
    #eye webcar restricted MMO -> through 5
    for i in range(6, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd = 77)
    # eye ourcar unrestricted MMO -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd = 42)
    #bbox ourcar unrestricted all -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd = 110)

def run_01_1():
    '''
    for i in range(1, 11):
        #revised combined unrestricted all -> finished
        Train(i, "datasets_combined_separated", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
             debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
             pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=10)
    #bbox combined restricted all -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=122)
    '''
    #eye webcar restricted all -> through 2
    for i in range(3, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=145)
    #enlarged ourcar restricted all -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=118)
    #bbox ourcar restricted mmo-> not started
    for i in range(1, 11):
        Train(i, "ourcar", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd = 134)
    #bbox ourcar restricted all-> not started
    for i in range(1, 11):
        Train(i, "ourcar", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd = 72)

def run_02_0():
    '''
    #bbox combined unrestricted MMO -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=12)
    #enlarged combined restricted all -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=139)
    '''
    #eye webcar unrestricted mmo -> through 8
    for i in range(9, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=8)
    #enlarged webcar_clean restricted all -> not finished
    for i in range(1, 11):
        Train(i, "webcaricature", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=133)
    #bbox ourcar unrestricted mmo -> not finished
    for i in range(1, 11):
        Train(i, "ourcar", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=41)

def run_02_1():
    '''
    #bbox combined unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=3)
    #Eye webcar unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=7)
    #enlarged webcar_clean unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=10)
    #bbox webcar_clean restricted MMO -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=True, eval_pairs=True, vgg=True, epochEnd=124)
    '''
    #bbox webcar_clean restricted all -> through 8
    for i in range(9, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=125)
    #bbox webcar restricted all -> not finished
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=133)
    #enlarged ourcar unrestricted all -> not finished
    for i in range(1, 11):
        Train(i, "ourcar", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=28)


def run_03_0():
    '''
    #eye combined unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=18)
    '''
    #eye combined restricted all -> through 7
    for i in range(8, 11):
        Train(i, "datasets_combined_separated", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=133)
    #bbox webcar unrestricted MMO  -> not started
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=133)
    #eye webcar_clean restricted all -> not started
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=149)
    #eye ourcar unrestricted all -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=21)
def run_03_1():
    '''
    #eye combined unrestricted mmo -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=9)
    #bbox webcar_clean unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=6)
    #bbox webcar_clean unrestricted mmo -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=True, eval_pairs=True, vgg=True, epochEnd=10)
    #eye webcar_clean unrestricted mmo -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=True, isClean=True, eval_pairs=True, vgg=True, epochEnd=11)
    '''
    #eye ourcar restricted all -> through 9
    for i in range(10, 11):
        Train(i, "ourcar", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=94)
    #bbox webcar unrestricted all -> not started
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=6)
    #eye ourcar restricted mmo -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "eye_location_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=16)

def run_07_0():
    '''
    #enlarged combined unrestricted all -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "enlarged_bounding_box_based_rotated", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=6)
    #eye combined restricted mmo -> finished
    for i in range(1, 11):
        Train(i, "datasets_combined_separated", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True,  epochEnd=76)
    #bbox webcar restricted mmo -> finished
    for i in range(1, 11):
        Train(i, "webcaricature", "bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=False, eval_pairs=True, vgg=True, epochEnd=138)
    '''
    #eye webcar_clean unrestricted all -> through 2
    for i in range(3, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=85)
    #revised webcar_clean restricted all -> not started
    for i in range(1, 11):
        Train(i, "webcaricature", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=True, eval_pairs=True, vgg=True, epochEnd=145)
    #eye webcar_clean restricted mmo -> not started
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=True, eval_pairs=True, vgg=True, epochEnd=7)
    #revised ourcar restricted all -> not started
    for i in range(1, 11):
        Train(i, "ourcar", "revised_bounding_box_based_rotated", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=False, isClean=False, eval_pairs=True, vgg=True, epochEnd=147)

#run_01_0()
#run_01_1()
#run_02_0()
#run_02_1()
#run_03_0()ssh s
#run_03_1()
#run_07_0()
