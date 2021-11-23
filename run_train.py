from Train import Train, Evaluate
def run_02_0():
    #webcar unrestricted all clean eye
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based", lr= 0.000001, device="cuda:0", finetune=True, foldNum=i, debug=False,
              doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False, MMO=False, isClean=True, eval_pairs=True)
def run_02_1():
    for i in range(1, 11):
        #webcar  unrestricted, MMO bbox
        Train(i, "webcaricature", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=True, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        #Combo Unrestricted MMO eye
        Train(i, "datasets_combined_separated", "eye_location_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=True, isClean=False, eval_pairs=True)
def run_03_0():
    for i in range(1, 11):
        #webcar unrestricted all eye
        Train(i, "webcaricature", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        #ourcar restricted all bbox
        Train(i, "ourcar", "bounding_box_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)
def run_03_1():
    for i in range(1, 11):
        #combo unrestricted all bbox
        Train(i, "datasets_combined_separated", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        #wevcar restricted all bbox
        Train(i, "webcaricature", "bounding_box_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)

def run_01_0():
    '''
    for i in range(1, 11):
        #combo restricted all eye
        Train(i, "datasets_combined_separated", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        #combo restricted mmo
        Train(i, "datasets_combined_separated", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=True, out=64, pickupOld=False,
              MMO=True, isClean=False, eval_pairs=True)
    '''
    #webcaricature_clean restricted all eye
    Train(i, "webcaricature", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True, foldNum= i, debug=False,
          doTrainEval=True, eval_gender=True, run_thresholding=True, restricted=True, out=64, pickupOld=False,
          MMO=False, isClean=True, eval_pairs=True)
def run_01_1():
    for i in range(1, 11):
        #combo unrestricted all eye
        Train(i, "datasets_combined_separated", "eye_location_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=False, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        #ourcar unrestricted mmo eye
        Train(i, "ourcar", "eye_location_based", lr=0.000001, device="cuda:1", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=True, isClean=False, eval_pairs=True)


def run_07_0():
    '''
    for i in range(1, 11):
        # webcar unrestricted mmo eye
        Train(i, "webcaricature", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True,
              foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=True, isClean=False, eval_pairs=True)
    for i in range(1, 11):
        # webcar unrestricted mmo_clean eye
        Train(i, "webcaricature", "eye_location_based", lr=0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False
              , doTrainEval=False, eval_gender=False, run_thresholding=True, restricted=False, out=64, pickupOld=False,
              MMO=True, isClean=True, eval_pairs=True)
    '''
    #Webcar restricted mmo eye
    for i in range(1, 11):
        Train(i, "webcaricature", "eye_location_based", lr= 0.000001, device="cuda:0", finetune=True, foldNum=i,
              debug=False, doTrainEval=True, eval_gender=True, run_thresholding=True, restricted=True, out=64,
              pickupOld=False, MMO=True, isClean=True, eval_pairs=True)

#run_01_0()
#run_01_1()
#run_02_1()
#run_03_0()
#run_03_1()
#run_07_0()