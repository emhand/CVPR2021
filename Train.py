# https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a
from ContrastiveLoss import ContrastiveLoss
from SiameseNetworkDatasetRevised_AllMatchImage import SiameseNetworkDataset
from SiameseNetwork import SiameseNetwork
from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from sklearn import metrics
from pathlib import Path

def imshow(img1, img2, label, text=None, should_save=False):


    #image1 = img1.cpu()
    #npimg1 = image1.detach().numpy()
    #image2 = img2.cpu()
    #npimg2 = image2.detach().numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    print(label)
    concatenated = torch.cat((img1, img2), 0)
    # print(concatenated.shape)
    gridded = torchvision.utils.make_grid(concatenated)
    #print(example_batch[2].numpy())  # print the labels
    #imshow(gridded)  # show the images
    npimg = gridded.cpu().detach().numpy()
    print(npimg)
    plt.show(npimg)
    plt.close()


def show_plot(iteration, trainData, valData, title, loss=True):
    epochs = []
    #print(len(trainData), len(valData))

    if len(trainData) > 0:
        if len(trainData) < len(valData):
            diff_len = len(valData) - len(trainData)
            val_to_add = trainData[-1]
            for i in range(len(valData)):
                epochs.append(i + 1)
            for i in range(diff_len):
                trainData.append(val_to_add)
        elif len(valData) < len(trainData):
            diff_len = len(trainData) - len(valData)
            val_to_add = valData[-1]
            for i in range(len(trainData)):
                epochs.append(i + 1)
            for i in range(diff_len):
                valData.append(val_to_add)
        else:
            for i in range(len(valData)):
                epochs.append(i+1)
        print(len(epochs), len(valData), len(trainData))
        plt.plot(epochs, trainData, label="Training")
        if loss:
            plt.title("Loss over epochs, train and val")
        else:
            plt.title("F1 over epochs, train and val")

    else:
        for i in range(len(valData)):
            epochs.append(i+1)
        if loss:
            plt.title("Loss over epochs, val")
        else:
            plt.title("F1 over epochs, val")


    plt.plot(epochs, valData, label="Validation")

    plt.xlabel("Epochs")
    if loss:
        plt.ylabel("Loss")
    else:
        plt.ylabel("F1")
    plt.legend()
    plt.savefig(title)
    plt.close()


def load_vgg_weights(net):
    preloadnet = models.vgg16(pretrained=True)
    preloadnet.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True)
    )
    net.load_state_dict(preloadnet.state_dict())
    return net


def load_data(root, imagepairs, type, fold=0, celebA=True, transform=None, invert=False, shuffle=True, restricted=False, bsize=64, MMO=False, isClean=False):
    # print(imagepairs)

    data = SiameseNetworkDataset(root=root,
                                 imageVerificationPairs=imagepairs, type=type, fold=fold, celebA=celebA,
                                 transform=transform, invert=invert, restricted=restricted, MMO=MMO,  isClean=isClean)
    loader = DataLoader(data, shuffle=shuffle, num_workers=8, batch_size=bsize)
    return loader, len(data)


def interpolate(tprs, fprs, rate):
    '''
    zipped_lists = zip(fprs, tprs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    fprs, tprs = [list(tuple) for tuple in tuples]
    '''

    tmpFP = 0
    for i, fpr in enumerate(fprs):
        if fpr >= tmpFP and fpr <= rate:
            tmpFP = fpr
            ind = i
    FPVal = tmpFP
    nextFPVal = fprs[ind + 1]
    distance = nextFPVal - FPVal
    ldiff = rate - FPVal
    leftDist = 1 - (ldiff / distance)
    rdiff = nextFPVal - rate
    rightDist = 1 - (rdiff / distance)

    TPValCont = tprs[ind] * leftDist
    TPValNextCont = tprs[ind + 1] * rightDist
    return TPValCont + TPValNextCont


def plot_roc(dataset, tprs, fprs, epoch, RunNum, cropType, Test=False, restricted=False, MMO=False, isClean =False, vgg=True):
    if restricted:
        restriction = "Restricted"
    else:
        restriction = "Unrestricted"
    if MMO:
        pairType = "MMO"
    else:
        pairType = "All"
    if dataset == "ourcar":
        data = "ourcar"
        endDir = ""
    else:
        data = "webcar"
        if isClean:
            endDir="_cleaned"
        else:
            endDir = ""
    if not vgg:
        endDir += "Resnet"

    if Test:
        saveName = "figures/" + cropType +"/" + data +restriction + pairType + endDir +"/" + "ROC_epoch" + str(epoch) + "run" + str(RunNum) + "_Test.png"
    else:
        saveName = "figures/" + cropType +"/" + data + restriction + pairType + endDir + "/" + "ROC_epoch" + str(epoch) + "run" + str(RunNum) + "_Train.png"
    plt.plot(fprs, tprs)
    # plt.plot(tprs)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(saveName)
    plt.close()


def calculate_ROC_metrics(dataSet, tprs, fprs, epoch, RunNum, cropType, Test=False, Restricted=False, MMO = False, isClean =False, vgg=False):
    zipped_lists = zip(fprs, tprs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    fprs, tprs = [list(tuple) for tuple in tuples]
    plot_roc(dataset=dataSet, tprs=tprs, fprs=fprs, epoch=epoch, RunNum=RunNum, Test=Test, cropType=cropType, restricted=Restricted, MMO=MMO, isClean=isClean, vgg=vgg)
    auc = metrics.auc(fprs, tprs)  # gives area under the curve
    tpr1 = interpolate(tprs, fprs, .1)
    tpr01 = interpolate(tprs, fprs, .01)
    return auc, tpr1, tpr01




def calculate_metrics(preds, sortedLabels):
    acc = np.mean(preds == sortedLabels)
    tp = np.sum(preds * sortedLabels)

    tn = np.sum((1 - sortedLabels) * (1 - preds))
    fp = np.sum((1 - sortedLabels) * preds)
    fn = np.sum(sortedLabels * (1 - preds))
    #print(tn, tp, fp, fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if (prec + recall) > 0:
        F1 = (2 * prec * recall) / (prec + recall)
    else:
        F1 = 0
    return acc, tp, tn, fp, fn, tpr, fpr, prec, recall, F1

def evaluate_pair_composition(EvalList, preds, labels, epoch, RunNum, cropType, webCar = True, restricted=False, MMO = False, isClean = False):
    if MMO:
        pairingType = 'MMO'
    else:
        pairingType = 'All'
    if restricted:
        restriction = "Restricted"
    else:
        restriction = "Unrestricted"
    if webCar:
        if isClean:
            ending = "_cleaned"
        else:
            ending = ""
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType + "/webcar" + restriction + pairingType + ending + "/Run" + str(
                RunNum) + "_Test_Counts_By_Pair.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType + "/webcar" + restriction + pairingType + ending + "/Run" + str(
                RunNum) + "_Eval_Counts_By_Pair.txt", "a")
        print("PAIR PATH: model_metrics/" + cropType +"/webcar" + restriction + pairingType + "/Run" + str(RunNum) + "_Test_Counts_By_Pair_webcar.txt")

    elif "combined" in EvalList:
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType +"/combined" + restriction + pairingType + "/Run" + str(RunNum) + "_Test_Counts_By_Pair_combined.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType +"/combined" + restriction + pairingType + "/Run" + str(RunNum) + "_Eval_Counts_By_Pair_combined.txt", "a")
        print("PAIR PATH: model_metrics/" + cropType +"/combined" + restriction + pairingType + "/Run" + str(RunNum) + "_Test_Counts_By_Pair_combined.txt")
    else:
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType + "/ourcar" + restriction + pairingType + "/Run" + str(
                RunNum) + "_Test_Counts_By_Pair_OURCAR.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType + "/ourcar" + restriction + pairingType + "/Run" + str(
                RunNum) + "_Eval_Counts_By_Pair_OURCAR.txt", "a")
        print("PAIR PATH: model_metrics/" + cropType + "/ourcar" + restriction + pairingType + "/Run" + str(
                RunNum) + "_Test_Counts_By_Pair_OURCAR.txt")





    ppTP, ppTN, ppFP, ppFN, pcTP, pcTN, pcFP, pcFN, ccTP, ccTN, ccFP, ccFN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    MixppTP, MixppTN, MixppFP, MixppFN, MixpcTP, MixpcTN, MixpcFP, MixpcFN, MixccTP, MixccTN, MixccFP, MixccFN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    print(EvalList)
    dataFile = open(EvalList, "r")
    pairs = dataFile.readlines()
    evalUntil = len(preds)
    pairs = pairs[:evalUntil]

    # print(maleVal)
    for i, pair in enumerate(pairs):
        id0, im0, id1, im1 = pair.split()
        id0, id1 = int(id0), int(id1)
        id0IsCar, id1IsCar = False, False
        if id0 %2 == 0:
            checkid0 = id0 -1
            id0IsCar = True
        else:
            checkid0 = id0
        if id1 %2 == 0:
            checkid1 = id1 -1
            id1IsCar = True
        else:
            checkid1 = id1
        if checkid0 == checkid1: #groundTruth Match

            if labels[i] == preds[i] and not id0IsCar and not id1IsCar:
                ppTP +=1
            elif labels[i] == preds[i] and ((not id0IsCar and id1IsCar) or id0IsCar and not id1IsCar):
                pcTP +=1
            elif labels[i] == preds[i] and id0IsCar and id1IsCar:
                ccTP += 1
            elif labels[i] != preds[i] and not id0IsCar and not id1IsCar:
                ppFN += 1
            elif labels[i] != preds[i] and ((not id0IsCar and id1IsCar) or id0IsCar and not id1IsCar):
                pcFN +=1
            elif labels[i] != preds[i] and id0IsCar and id1IsCar:
                ccFN +=1
        else:
            if labels[i] == preds[i] and not id0IsCar and not id1IsCar:
                MixppTN +=1
            elif labels[i] == preds[i] and ((not id0IsCar and id1IsCar) or id0IsCar and not id1IsCar):
                MixpcTN +=1
            elif labels[i] == preds[i] and id0IsCar and id1IsCar:
                MixccTN += 1
            elif labels[i] != preds[i] and not id0IsCar and not id1IsCar:
                MixppFP += 1
            elif labels[i] != preds[i] and ((not id0IsCar and id1IsCar) or id0IsCar and not id1IsCar):
                MixpcFP +=1
            elif labels[i] != preds[i] and id0IsCar and id1IsCar:
                MixccFP +=1



    # print(ffTP, ffTN, ffFN, ffFP)
    if epoch == 0:
        fout.write(
            "Match p FN, Match p FP, Match p TP, Match p TN, Match c FN, Match c FP, Match c TP, Match c TN, Match pc FN, Match pc FP, Match pc TP, Match pc TN\n")
        fout.write(
            "-atch p FN, -atch p FP, -atch p TP, -atch p TN, -atch c FN, -atch c FP, -atch c TP, -atch c TN, -atch pc FN, -atch pc FP, -atch pc TP, -atch pc TN\n")
    fout.write(
        str(ppFN) + " " + str(ppFP) + " " + str(ppTP) + " " + str(ppTN) + " " + str(ccFN) + " " + str(ccFP) + " " + str(
            ccTP) + " " + str(ccTN) + " " + str(pcFN) + " " + str(pcFP) + " " + str(pcTP) + " " + str(pcTN))
    fout.write("\n")
    fout.write(
        str(MixppFN) + " " + str(MixppFP) + " " + str(MixppTP) + " " + str(MixppTN) + " " + str(MixccFN) + " " + str(
            MixccFP) + " " + str(MixccTP) + " " + str(MixccTN) + " " + str(MixpcFN)
        + " " + str(MixpcFP) + " " + str(MixpcTP) + " " + str(MixpcTN))
    fout.write("\n")

def evaluate_gender(EvalList, preds, labels, epoch, RunNum, cropType, webCar = True, restricted=False, MMO = False, isClean = False):
    #print("PREDS", preds)
    #print("LABELS", labels)
    if MMO:
        pairingType = 'MMO'
    else:
        pairingType = 'All'
    if restricted:
        restriction = "Restricted"
    else:
        restriction = "Unrestricted"
    if webCar:
        fMale = open("namesLists/WebCaricature_Male.txt", "r")
        fFemale = open("namesLists/WebCaricature_Female.txt", "r")
        if isClean:
            ending = "_cleaned"
        else:
            ending = ""
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType +"/webcar" + restriction + pairingType +ending +"/Run" + str(RunNum) + "_Test_Counts_By_Gender.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType +"/webcar" + restriction + pairingType + ending +  "/Run" + str(RunNum) + "_Eval_Counts_By_Gender.txt", "a")
        fall = open("verification_pairs-DONOTPUSH/map_person_to_number_webcaricature.txt", "r")
    elif "combined" in EvalList:
        fMale = open("namesLists/Combined_Male.txt", "r")
        fFemale = open("namesLists/Combined_Female.txt", "r")
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType +"/combined" + restriction + pairingType + "/Run" + str(RunNum) + "_Test_Counts_By_Gender_combined.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType +"/combined" + restriction + pairingType + "/Run" + str(RunNum) + "_Eval_Counts_By_Gender_combined.txt", "a")
        fall = open("verification_pairs-DONOTPUSH/map_person_to_number_combined.txt", "r")
    else:
        fMale = open("namesLists/OurCar_Male.txt", "r")
        fFemale = open("namesLists/OurCar_Female.txt", "r")
        if "test" in EvalList:
            fout = open("model_metrics/" + cropType +"/ourcar" + restriction + pairingType + "/Run" + str(RunNum) + "_Test_Counts_By_Gender_OURCAR.txt", 'a')
        else:
            fout = open("model_metrics/" + cropType +"/ourcar" + restriction + pairingType + "/Run" + str(RunNum) + "_Eval_Counts_By_Gender_OURCAR.txt", "a")
        fall = open("verification_pairs-DONOTPUSH/map_person_to_number_ourcar.txt", "r")
    maleNames = fMale.readlines()
    femaleNames = fFemale.readlines()

    femaleDict = {}
    maleDict = {}

    for femaleName in femaleNames:
        femaleDict[femaleName.strip()] = 0
    for maleName in maleNames:
        maleDict[maleName.strip()] = 0
    fMale.close()
    fFemale.close()
    del maleNames
    del femaleNames


    peopleNums = fall.readlines()

    peopleNumDict = {}
    for peopleNum in peopleNums:
        nameNumList = peopleNum.split()
        num = nameNumList[-1].strip()
        rest = nameNumList[:-1]
        space = " "
        rest = space.join(rest)
        rest = rest.lower()
        rest = rest.replace(" ", "_")
        #print(num)
        if "_caricature" in rest:
            rest = rest.replace("_caricature", "")
        if rest not in peopleNumDict.keys():
            peopleNumDict[rest] = [int(num)]
        else:
            peopleNumDict[rest].append(int(num))
    #print(peopleNumDict)
    for name in femaleDict.keys():
        nums = peopleNumDict[name]
        femaleDict[name] = nums
    for name in maleDict.keys():
        nums = peopleNumDict[name]
        maleDict[name] = nums
    #print(maleDict)
    del peopleNumDict

    mmTP, mmTN, mmFP, mmFN, ffTP, ffTN, ffFP, ffFN, mfTP, mfTN, mfFP, mfFN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    MixmmTP, MixmmTN, MixmmFP, MixmmFN, MixffTP, MixffTN, MixffFP, MixffFN, MixmfTP, MixmfTN, MixmfFP, MixmfFN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    print(EvalList)
    dataFile = open(EvalList, "r")
    pairs = dataFile.readlines()
    evalUntil = len(preds)
    pairs = pairs[:evalUntil]
    femaleVal = femaleDict.values()
    femaleVal = [j for sub in femaleVal for j in sub]
    maleVal = maleDict.values()
    maleVal = [j for sub in maleVal for j in sub]
    #print(maleVal)
    for i, pair in enumerate(pairs):
        id0, im0, id1, im1 = pair.split()
        id0, id1 = int(id0), int(id1)
        #print(id0, id1)
        if id0%2 == 0:
            id0 -= 1
        elif id1%2 == 0:
            id1 -=1
        #print(id0)
        label = labels[i]
        pred = preds[i]
        if id0 == id1:
            if id0 in femaleVal:
                #print("FEMALE ID0", str(id0))
                if label != pred:
                    ffFN += 1
                else:
                    ffTP += 1
            elif id0 in maleVal:
                if label != pred:

                    mmFN += 1
                else:
                    mmTP += 1
        else:
            fn, fp, tp, tn = 0, 0, 0, 0
            if label != pred:
                fp += 1
            else:
                tn += 1
            if id0 in femaleVal and id1 in femaleVal:
                MixffFN, MixffFP, MixffTP, MixffTN = MixffFN+fn, MixffFP+fp, MixffTP+tp, MixffTN+tn
            elif id0 in femaleVal and id1 in maleVal:
                MixmfFN, MixmfFP, MixmfTP, MixmfTN = MixmfFN+fn, MixmfFP+fp, MixmfTP+tp, MixmfTN+tn
            else:
                MixmmFN, MixmmFP, MixmmTP, MixmmTN = MixmmFN+fn, MixmmFP+fp,MixmmTP+tp, MixmmTN+tn

    #print(ffTP, ffTN, ffFN, ffFP)
    if epoch == 0:
        fout.write("Match F FN, Match F FP, Match F TP, Match F TN, Match M FN, Match M FP, Match M TP, Match M TN, Match FM FN, Match FM FP, Match FM TP, Match FM TN\n")
        fout.write(
            "-atch F FN, -atch F FP, -atch F TP, -atch F TN, -atch M FN, -atch M FP, -atch M TP, -atch M TN, -atch FM FN, -atch FM FP, -atch FM TP, -atch FM TN\n")
    fout.write(str(ffFN) + " " + str(ffFP) + " " +str(ffTP)+ " " + str(ffTN)+ " " + str(mmFN)+ " " + str(mmFP)+ " " + str(mmTP)+ " " + str(mmTN)+ " " +str(mfFN)+ " " + str(mfFP)+ " " + str(mfTP)+ " " + str(mfTN))
    fout.write("\n")
    fout.write(str(MixffFN)+ " " + str(MixffFP)+ " " + str(MixffTP)+ " " + str(MixffTN)+ " " +str(MixmmFN)+ " " + str(MixmmFP)+ " " + str(MixmmTP)+ " " + str(MixmmTN)+ " " +str(MixmfFN)
               + " " + str(MixmfFP)+ " " + str(MixmfTP)+ " " +str(MixmfTN))
    fout.write("\n")


def Train(RunNum, dataset, alignmentType, lr= .0001, device="cuda:0", finetune=False,pickupOld = False, foldNum=0,
          debug=False, doTrainEval = False, eval_gender = False, run_thresholding = False, restricted=False, out=64,
          MMO=False, isClean = False, eval_pairs = False, vgg=True, epochStart=0, epochEnd=0):
    print(debug)
    #print(doTrainEval)
    bsize = 64
    output_size = out
    if debug:
        print("WORKING IN DEBUG MODE...")
    if os.path.exists("models/") == False:
        os.mkdir("models/")
    if os.path.exists("figures/") == False:
        os.mkdir("figures/")
    if os.path.exists("model_metrics/") == False:
        os.mkdir("model_metrics/")
    if os.path.exists("numpy_data/") == False:
        os.mkdir("numpy_data/")
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),  # default
        transforms.RandomResizedCrop((178, 218), scale=(0.95, 1.05), ratio=(0.75, 1.25), interpolation=2),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if vgg:
        netType = ""
        modelType = "vgg-16"
    else:
        netType = "Resnet"
        modelType = "resnet-50"

    if not isClean:
        dirEnd = ""
    else:
        dirEnd = "_cleaned"
    if os.path.isdir("model_metrics/" + alignmentType)  == False:
        os.mkdir("model_metrics/" + alignmentType )
    if os.path.isdir("model_metrics/" + alignmentType + "/ourcarRestrictedAll" + dirEnd + netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/ourcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("model_metrics/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("model_metrics/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType)

    print(alignmentType)

    if os.path.isdir("figures/" + alignmentType ) == False:
        os.mkdir("figures/" + alignmentType )
    if os.path.isdir("figures/" + alignmentType + "/ourcarRestrictedAll" + dirEnd + netType)== False:
        os.mkdir("figures/" + alignmentType + "/ourcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("figures/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("figures/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType)

    if os.path.isdir("models/" + alignmentType ) == False:
        os.mkdir("models/" + alignmentType )
    if os.path.isdir("models/" + alignmentType + "/ourcarRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/ourcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/ourcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/ourcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/ourcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/webcarRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/webcarRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/combinedRestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/combinedRestrictedMMO" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/combinedUnrestrictedAll" + dirEnd+ netType)
    if os.path.isdir("models/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType) == False:
        os.mkdir("models/" + alignmentType + "/combinedUnrestrictedMMO" + dirEnd+ netType)

    if dataset == "celebA":
        root = str(Path.home()) + "/label_checking/celebA-id/CelebA/"
        verificationPair = "namesLists/acceptableidentity_CelebA.txt"
        celeba = True
        if vgg:
            modelpath = "models/run" + str(RunNum) + "_vgg-16_celebAPretrain_" + str(out)
            fname_train = "model_metrics/CelebA_PretrainEval_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/CelebA_PretrainTrain_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/run" + str(RunNum) + "_LOSS.png"
            titleF1 = "figures/run" + str(RunNum) + "_F1.png"
        else:
            modelpath = "models/run" + str(RunNum) + "_resnet-50_celebAPretrain_" + str(out)
            fname_train = "model_metrics/CelebA_Resnet50PretrainEval_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/CelebA_Resnet50PretrainTrain_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/Resnet50run" + str(RunNum) + "_LOSS.png"
            titleF1 = "figures/Resnet50run" + str(RunNum) + "_F1.png"
    elif dataset == "webcaricature":
        if not isClean:
            root = "image_alignment/" + alignmentType + "/" + "webcaricature_separated/"
        else:
            root = "image_alignment/" + alignmentType + "/" + "webcaricature_cleaned_separated/"
            # verificationPair = "verification_pairs-DONOTPUSH/webcaricature_verification_pairs_all.txt"
        verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_webcaricature.txt"
        celeba = False
        if not restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd + netType +"/webcaricature_finetuneTrain_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd +  netType +"/webcaricature_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd +  netType +"/run" + str(RunNum) + "_webcaricature_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_F1.png"
            modelpath = "models/" + alignmentType + "/webcarUnrestrictedAll" + dirEnd +  netType +"/run" + str(RunNum) + "_"+ modelType + "_webcaricatureFinetune_" + str(out)
        elif restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/webcarRestrictedAll" + dirEnd +  netType +"/webcaricature_finetuneTrain_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/webcarRestrictedAll" + dirEnd +  netType +"/webcaricature_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/webcarRestrictedAll" + dirEnd +  netType +"/run" + str(RunNum) + "_webcaricature_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/webcarRestrictedAll" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_F1.png"
            modelpath = "models/" + alignmentType + "/webcarRestrictedAll" + dirEnd + netType + "/run" + str(RunNum) + "_"+ modelType + "_webcaricatureFinetune_" + str(out)
        elif not restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd + netType + "/webcaricature_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd + netType + "/webcaricature_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_F1.png"
            modelpath = "models/" + alignmentType + "/webcarUnrestrictedMMO" + dirEnd + netType + "/run" + str(RunNum) + "_"+ modelType + "_webcaricatureFinetune_" + str(out)
        elif restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/webcarRestrictedMMO" + dirEnd + netType + "/webcaricature_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/webcarRestrictedMMO" + dirEnd + netType + "/webcaricature_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/webcarRestrictedMMO" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/webcarRestrictedMMO" + dirEnd + netType + "/run" + str(RunNum) + "_webcaricature_F1.png"
            modelpath = "models/" + alignmentType + "/webcarRestrictedMMO" + dirEnd +  netType +"/run" + str(RunNum) + "_"+ modelType + "_webcaricatureFinetune_" + str(out)
    elif "combined" in dataset:
        root = "image_alignment/" + alignmentType + "/" + dataset + "/"
        # verificationPair = "verification_pairs-DONOTPUSH/ourcar_verification_pairs_all.txt"
        verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_combined.txt"
        celeba = False
        if restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/combinedRestrictedAll" + netType + "/combined_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/combinedRestrictedAll" + netType + "/combined_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/combinedRestrictedAll" + netType + "/run" + str(RunNum) + "_combinedcar_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/combinedRestrictedAll" + netType + "/run" + str(RunNum) + "_combined_F1.png"
            modelpath = "models/" + alignmentType + "/combinedRestrictedAll" + netType + "/run" + str(
                RunNum) + "_"+ modelType + "_ourcarFinetune_" + str(out)
        elif not restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/combinedUnrestrictedAll" + netType + "/combined_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/combinedUnrestrictedAll" + netType + "/combined_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/combinedUnrestrictedAll" + netType + "/run" + str(RunNum) + "_combined_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/combinedUnrestrictedAll" + netType + "/run" + str(RunNum) + "_combined_F1.png"
            modelpath = "models/" + alignmentType + "/combinedUnrestrictedAll" + netType + "/run" + str(
                RunNum) + "_"+ modelType + "_combinedFinetune_" + str(out)
        elif restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/combinedRestrictedMMO" + netType + "/combined_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/combinedRestrictedMMO" + netType + "/combined_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/combinedRestrictedMMO" + netType + "/run" + str(RunNum) + "_combined_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/combinedRestrictedMMO" + netType + "/run" + str(RunNum) + "_combined_F1.png"
            modelpath = "models/" + alignmentType + "/combinedRestrictedMMO" + netType + "/run" + str(
                RunNum) + "_"+ modelType + "_combinedFinetune_" + str(out)
        elif not restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/combinedUnrestrictedMMO" + netType + "/combined_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/combinedUnrestrictedMMO" + netType + "/combined_finetuneEval_metrics_run" + str(
                RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/combinedUnrestrictedMMO" + netType + "/run" + str(RunNum) + "_combined_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/combinedUnrestrictedMMO" + netType + "/run" + str(RunNum) + "_combined_F1.png"
            modelpath = "models/" + alignmentType + "/combinedUnrestrictedMMO" + netType + "/run" + str(
                RunNum) + "_"+ modelType + "_combinedFinetune_" + str(out)
    else:
        root = "image_alignment/" + alignmentType + "/" + dataset + "/"
        # verificationPair = "verification_pairs-DONOTPUSH/ourcar_verification_pairs_all.txt"
        verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_ourcar.txt"
        celeba = False
        if restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/ourcarRestrictedAll" + netType + "/ourcar_finetuneTrain_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/ourcarRestrictedAll" + netType + "/ourcar_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/ourcarRestrictedAll" + netType + "/run" + str(RunNum) + "_ourcar_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/ourcarRestrictedAll" + netType + "/run" + str(RunNum) + "_ourcar_F1.png"
            modelpath = "models/" + alignmentType + "/ourcarRestrictedAll" + netType + "/run" + str(RunNum) + "_"+ modelType + "_ourcarFinetune_" + str(out)
        elif not restricted and not MMO:
            fname_train = "model_metrics/" + alignmentType + "/ourcarUnrestrictedAll" + netType + "/ourcar_finetuneTrain_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/ourcarUnrestrictedAll" + netType + "/ourcar_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/ourcarUnrestrictedAll" + netType + "/run" + str(RunNum) + "_ourcar_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/ourcarUnrestrictedAll" + netType + "/run" + str(RunNum) + "_ourcar_F1.png"
            modelpath = "models/" + alignmentType + "/ourcarUnrestrictedAll" + netType + "/run" + str(RunNum) + "_"+ modelType + "_ourcarFinetune_" + str(out)
        elif restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/ourcarRestrictedMMO" + netType + "/ourcar_finetuneTrain_metrics_run" + str(RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/ourcarRestrictedMMO" + netType + "/ourcar_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/ourcarRestrictedMMO" + netType + "/run" + str(RunNum) + "_ourcar_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/ourcarRestrictedMMO" + netType + "/run" + str(RunNum) + "_ourcar_F1.png"
            modelpath = "models/" + alignmentType + "/ourcarRestrictedMMO" + netType + "/run" + str(RunNum) + "_"+ modelType + "_ourcarFinetune_" + str(out)
        elif not restricted and MMO:
            fname_train = "model_metrics/" + alignmentType + "/ourcarUnrestrictedMMO" + netType + "/ourcar_finetuneTrain_metrics_run" + str(
                RunNum) + ".txt"
            fname_eval = "model_metrics/" + alignmentType + "/ourcarUnrestrictedMMO" + netType + "/ourcar_finetuneEval_metrics_run" + str(RunNum) + ".txt"
            titleLoss = "figures/" + alignmentType + "/ourcarUnrestrictedMMO" + netType + "/run" + str(RunNum) + "_ourcar_LOSS.png"
            titleF1 = "figures/" + alignmentType + "/ourcarUnrestrictedMMO" + netType + "/run" + str(RunNum) + "_ourcar_F1.png"
            modelpath = "models/" + alignmentType + "/ourcarUnrestrictedMMO" + netType + "/run" + str(RunNum) + "_"+ modelType + "_ourcarFinetune_" + str(out)

    train_dataloader, numTrain = load_data(root, verificationPair, "train", fold=foldNum, celebA=celeba,
                                           transform=train_transform, invert=False, restricted=restricted, MMO=MMO, isClean=isClean)
    val_dataloader, numVal = load_data(root, verificationPair, "val", fold=foldNum, celebA=celeba, transform=val_transform,
                                       invert=False, shuffle=False, restricted=restricted, MMO=MMO, isClean=isClean)
    # test_dataloader, numTest = load_data(root, verificationPair,  "test", celebA=celeba, transform= transform, invert=False)
    print(len(train_dataloader))
    print(len(val_dataloader))
    if not finetune and vgg:
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=out, bias=True),

        )
        criterion = torch.nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(net.parameters(), lr=.0001)  # dropping learning rate
        epochStart = 0
        epochEnd = 10
    elif not finetune and not vgg:
        print("LOADING RESNET50")
        net = models.resnet50(pretrained=True)
        net.fc = nn.Linear(in_features=2048, out_features=out, bias=True)
        criterion = torch.nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(net.parameters(), lr=.0001)  # dropping learning rate
        epochStart = 0
        epochEnd = 22
    elif not pickupOld:
        epochStart = epochStart
        epochEnd = epochEnd
        if vgg:
            print("Loading VGG model...\n")
            net = models.vgg16(pretrained=False)
            net.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=out, bias=True),
            )
            criterion = torch.nn.CosineEmbeddingLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            checkpoint = torch.load("models/run4_vgg-16_celebAPretrain", map_location=torch.device(device))
            net.load_state_dict(checkpoint)

        else:
            print("Loading RESNET model...\n")
            net = models.resnet50(pretrained=False)
            net.fc = nn.Linear(in_features=2048, out_features=out, bias=True)
            criterion = torch.nn.CosineEmbeddingLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            checkpoint = torch.load("models/run4_resnet-50_celebAPretrain_64", map_location=torch.device(device))
            net.load_state_dict(checkpoint)
        if epochEnd == 0:
            epochEnd = 150
    else:
        print("Loading old webcar model...\n")
        net = models.vgg16(pretrained=False)
        net.classifier = nn.Sequential(
           nn.Linear(in_features=25088, out_features=out, bias = True),
        )
        criterion = torch.nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        model_to_load = "models/run" + str(foldNum) +"_vgg-16_webcaricatureFinetune"
        checkpoint = torch.load(model_to_load, map_location=torch.device(device))
        net.load_state_dict(checkpoint)
        epochStart = 6
        epochEnd = 7
    net.to(device)
    iteration_number = 0
    counter = []
    loss_history = []
    train_losses = []
    val_losses = []
    train_F1s=[]
    val_F1s = []
    #print(doTrainEval)
    print(epochStart, epochEnd)
    if doTrainEval and dataset != "celebA" :
        train_eval_dataloader, numTrain = load_data(root, verificationPair, "train_eval", fold=foldNum, celebA=celeba,
                                                    transform=val_transform, invert=False, shuffle=False, restricted=restricted, bsize=bsize, MMO=MMO, isClean=isClean)
        if not debug:
            print("Doing initial TRAIN EVAL...", fname_train)
            average_train_loss, train_F1 = Evaluate(RunNum, fname_train, "train", dataset, alignmentType, net, train_eval_dataloader,
                                      epochStart, device=device, foldNum=foldNum, run_thresholding=run_thresholding,
                                    debug=debug, eval_gender = False, restricted=restricted, bsize=bsize, out=out, MMO=MMO,  isClean=isClean, eval_pairs=False, vgg=vgg)
            train_losses.append(average_train_loss)
            train_F1s.append(train_F1)
    elif doTrainEval and dataset== "celebA":
        train_eval_dataloader, numTrain = load_data(root, verificationPair, "train", fold=foldNum, celebA=celeba,
                                                    transform=val_transform, invert=False, shuffle=False, restricted=restricted, MMO=MMO, isClean=isClean)
        if not debug:
            print("Doing initial TRAIN EVAL...", fname_train)
            average_train_loss, train_F1 = Evaluate(RunNum, fname_train, "train", dataset, alignmentType, net, train_eval_dataloader,
                                      epochStart, device=device, foldNum=foldNum, run_thresholding=run_thresholding,
                                    debug=debug, eval_gender = False, restricted=restricted, bsize=bsize, out=out, MMO=MMO,  isClean=isClean, eval_pairs = False, vgg=vgg)
            train_losses.append(average_train_loss)
            train_F1s.append(train_F1)
    if not debug:
        print("Doing initial VAL EVAL...", fname_eval)
        average_val_loss, val_F1 =  Evaluate(RunNum, fname_eval, "test", dataset, alignmentType, net, val_dataloader,
                                      epochStart, device=device, foldNum=foldNum, run_thresholding=run_thresholding,
                                    debug=debug, eval_gender = False, restricted=restricted,  bsize=bsize, out=out, MMO=MMO,  isClean=isClean, eval_pairs = False, vgg=vgg)
        val_F1s.append(val_F1)
        val_losses.append(average_val_loss)
    #val_losses.append(0)

    for epoch in range(epochStart, epochEnd):
        #print(epochStart, epochEnd)
        net.train().to(device)
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            optimizer.zero_grad()
            output1 = net(img0)
            output2 = net(img1)
            #print("FT LABELS IM 1 ", output1.shape)
           #imshow(output1, output2, label)
            loss_contrastive = criterion(output1.view(-1, output_size), output2.view(-1, output_size), label.view(-1)) #batch size by output
            loss_contrastive.backward()
            optimizer.step()
            loss_item= loss_contrastive.item()

            running_loss += loss_item
            if i % 10 == 0:
                print("Run{}_Epoch number {}, {:.2f}% Remaining\nCurrent loss {}\n".format(RunNum, epoch,(100-((i/len(train_dataloader)) *100)), loss_contrastive.item()))
                if os.path.isdir("figures/") == False:
                    os.mkdir("figures/")

                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
            #if i %500 == 0 and i != 0:
            #   print("Evaluating Val Set...", fname_eval)
            #   average_val_loss, val_F1 = Evaluate(RunNum, fname_eval, "test", dataset, alignmentType, net, val_dataloader, epoch, device=device, foldNum=foldNum, run_thresholding=run_thresholding, debug=debug, eval_gender=False, restricted=restricted,  bsize=bsize, out=out,  isClean=isClean, eval_pairs=False)
            #   val_losses.append(average_val_loss)
            #   val_F1s.append(val_F1)
            if debug:
                if i == 50:
                    break

        print("Doing final Val eval for epoch...", fname_eval)
        average_val_loss, val_F1 = Evaluate(RunNum, fname_eval, "val", dataset, alignmentType, net, val_dataloader,
                                    epoch, device=device, foldNum=foldNum, run_thresholding=run_thresholding,
                                    debug=debug, eval_gender=eval_gender, restricted=restricted, bsize=bsize, out=out,  isClean=isClean, eval_pairs=eval_pairs, vgg=vgg)
        val_losses.append(average_val_loss)
        val_F1s.append(val_F1)
        if doTrainEval and epoch%10 == 0:
            print("Evaluating Train Set...")
            average_train_loss, train_F1 = Evaluate(RunNum, fname_train, "train", dataset, alignmentType, net, train_eval_dataloader,
                                            epoch, device=device,foldNum=foldNum, run_thresholding=run_thresholding,
                                            debug=debug, eval_gender=eval_gender, restricted=restricted, bsize=bsize, out=out,  isClean=isClean, eval_pairs=eval_pairs, vgg=vgg)
            train_losses.append(average_train_loss)
            train_F1s.append(train_F1)
        if len(val_losses) <len(train_losses):
            val_losses.append(0)
        #train_losses.append(average_train_loss)
        #print(len(val_losses), len(train_losses))
        #val_losses.append(0)
        #print(len(train_losses), len(val_losses))
        show_plot(counter, train_losses, val_losses, titleLoss)
        show_plot(counter, train_F1s, val_F1s, titleF1, loss=False)

        if os.path.isdir("models/") == False:
            os.mkdir("models/")

        torch.save(net.state_dict(), modelpath)

    #housekeeping to remove net from memory to run sequential train
    del net #get rid of net when done training
    #with torch.cude.device(device):
    #    torch.cuda.empty_cache() #clear the cuda cache


def Evaluate(RunNum, fname, eval_type, datatype, cropType, net, dataloader, epoch, device="cuda:0", foldNum=0, run_thresholding=False, debug=False, eval_gender = False, restricted=False, bsize=64, out=4096, MMO=False, isClean = False, eval_pairs=False, vgg=True):
    #print(run_thresholding)
    if vgg:
        netType = ""
        modelType = "vgg-16"
    else:
        netType = "Resnet"
        modelType = "resnet-50"

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        #transforms.RandomHorizontalFlip(p=0.5),  # default
        #transforms.RandomResizedCrop((178, 218), scale=(0.95, 1.05), ratio=(0.75, 1.25), interpolation=2),
        #transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    if isClean:
        dirEnd = "_cleaned"
    else:
        dirEnd = ""
    if MMO:
        pairingType = 'MMO'
    else:
        pairingType = 'All'
    if restricted:
        restriction = "restricted"
    else:
        restriction = "unrestricted"
    if (eval_gender or eval_pairs) and datatype == "webcaricature":
        webcar = True
        ####WILL NEED TO ALTER FOR ADDITIONAL FOLDS
        if eval_type == "train" and foldNum == 0:
            EvalList = "webcaricatureFolds_"+ restriction + pairingType + dirEnd + "/fold0evaluatetrainPairs.txt"
        elif foldNum == 0:
            EvalList = "webcaricatureFolds_"+ restriction + pairingType +  dirEnd + "/fold0testPairs.txt"
        else:
            EvalList = "webcaricatureFolds_"+ restriction + pairingType +  dirEnd + "/fold" + str(foldNum) +"testPairs.txt"
        print("EVAL LIST: ", EvalList)
    elif (eval_gender or eval_pairs) and "combined" in datatype:
        webcar = False
        if eval_type == "train" and foldNum == 0:
            EvalList = "combinedFolds_" + restriction + pairingType + "/fold0evaluatetrainPairs.txt"
        elif foldNum == 0:
            EvalList = "combinedFolds_" + restriction + pairingType + "/fold0testPairs.txt"
        else:
            EvalList = "combinedFolds_" + restriction + pairingType + "/fold" + str(foldNum) + "testPairs.txt"

    elif (eval_gender or eval_pairs) and datatype == "ourcar":
        webcar = False
        if eval_type == "train" and foldNum == 0:
            EvalList = "ourcarFolds_"+ restriction + pairingType + "/fold0evaluatetrainPairs.txt"
        elif foldNum == 0:
            EvalList = "ourcarFolds_"+ restriction + pairingType + "/fold0testPairs.txt"
        else:
            EvalList = "ourcarFolds_"+ restriction + pairingType + "/fold" + str(foldNum) +"testPairs.txt"

    test = False
    if os.path.isdir("numpy_data/" + cropType ) == False:
        os.mkdir("numpy_data/" + cropType )
    if os.path.isdir("numpy_data/" + cropType +"/webcaricatureUnrestrictedAll" +dirEnd +netType) == False:
        os.mkdir("numpy_data/" + cropType +"/webcaricatureUnrestrictedAll" +dirEnd + netType)
    if os.path.isdir("numpy_data/" + cropType +"/webcaricatureRestrictedAll" +dirEnd + netType) == False:
        os.mkdir("numpy_data/" + cropType +"/webcaricatureRestrictedAll" +dirEnd + netType)
    if os.path.isdir("numpy_data/" + cropType +"/webcaricatureUnrestrictedMMO" +dirEnd + netType) == False:
        os.mkdir("numpy_data/" + cropType +"/webcaricatureUnrestrictedMMO" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/webcaricatureRestrictedMMO" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/webcaricatureRestrictedMMO" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/ourcarUnrestrictedAll" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/ourcarUnrestrictedAll" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/ourcarRestrictedAll" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/ourcarRestrictedAll" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/ourcarUnrestrictedMMO" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/ourcarUnrestrictedMMO" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/ourcarRestrictedMMO"+dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/ourcarRestrictedMMO"+dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/combinedUnrestrictedAll" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/combinedUnrestrictedAll" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/combinedRestrictedAll" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/combinedRestrictedAll" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/combinedUnrestrictedMMO" +dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/combinedUnrestrictedMMO" +dirEnd+ netType)
    if os.path.isdir("numpy_data/" + cropType +"/combinedRestrictedMMO"+dirEnd+ netType) == False:
        os.mkdir("numpy_data/" + cropType +"/combinedRestrictedMMO"+dirEnd+ netType)

    if eval_type == "test" or eval_type == "val":
        test = True
    if datatype == "celebA":
        numpyName = "numpy_data/Epoch" + str(epoch) + "_run" + str(RunNum) + "_thresholdingData.npy"
    elif datatype == "webcaricature" and not restricted and not MMO:
        numpyName = "numpy_data/" + cropType +"/webcaricatureUnrestrictedAll"+dirEnd + netType+"/Epoch" + str(epoch) + "_run" + str(RunNum) + "_thresholdingData_webcaricature.npy"
    elif datatype == "webcaricature" and restricted and not MMO:
        numpyName = "numpy_data/" + cropType +"/webcaricatureRestrictedAll"+dirEnd + netType+"/Epoch" + str(epoch) + "_run" + str(RunNum) + "_thresholdingData_webcaricature.npy"
    elif datatype == "webcaricature" and not restricted and MMO:
        numpyName = "numpy_data/" + cropType +"/webcaricatureUnrestrictedMMO"+dirEnd+ netType +"/Epoch" + str(epoch) + "_run" + str(
            RunNum) + "_thresholdingData_webcaricature.npy"
    elif datatype == "webcaricature" and restricted and MMO:
        numpyName = "numpy_data/" + cropType +"/webcaricatureRestrictedMMO/Epoch"+dirEnd + netType+"" + str(epoch) + "_run" + str(
            RunNum) + "_thresholdingData_webcaricature.npy"
    elif datatype == "ourcar" and not restricted and not MMO:
        numpyName = "numpy_data/" + cropType +"/ourcarUnrestrictedAll" + netType +"/Epoch" + str(epoch) + "_run" + str(RunNum) + "_thresholdingData_ourcar.npy"
    elif datatype == "ourcar" and restricted and not MMO:
        numpyName = "numpy_data/" + cropType +"/ourcarRestrictedAll" + netType +"/Epoch" + str(epoch) + "_run" + str(RunNum) + "_thresholdingData_ourcar.npy"
    elif datatype == "ourcar" and not restricted and MMO:
        numpyName = "numpy_data/" + cropType +"/ourcarUnrestrictedMMO" + netType + "/Epoch" + str(epoch) + "_run" + str(
            RunNum) + "_thresholdingData_ourcar.npy"
    elif datatype == "ourcar" and restricted and MMO:
        numpyName = "numpy_data/" + cropType + "/ourcarRestrictedMMO" + netType + "/Epoch" + dirEnd + "" + str(
            epoch) + "_run" + str(
            RunNum) + "_thresholdingData_ourcar.npy"
    elif "combined" in datatype and not restricted and not MMO:
        numpyName = "numpy_data/" + cropType + "/combinedUnrestrictedAll" + netType +"/Epoch" + str(epoch) + "_run" + str(
            RunNum) + "_thresholdingData_combined.npy"
    elif "combined" in datatype and restricted and not MMO:
        numpyName = "numpy_data/" + cropType + "/combinedRestrictedAll" + netType +"/Epoch" + str(epoch) + "_run" + str(
        RunNum) + "_thresholdingData_combined.npy"
    elif "combined" in datatype and not restricted and MMO:
        numpyName = "numpy_data/" + cropType + "/combinedUnrestrictedMMO" + netType + "/Epoch" + str(epoch) + "_run" + str(
        RunNum) + "_thresholdingData_combined.npy"
    elif "combined" in datatype and restricted and MMO:
        numpyName = "numpy_data/" + cropType + "/combinedRestrictedMMO/Epoch" + dirEnd + netType + "" + str(
            epoch) + "_run" + str(
            RunNum) + "_thresholdingData_combined.npy"

    if dataloader == None:
        if datatype == "celebA":
            modelName = "models/run" + str(RunNum) + "_vgg-16_celebAPretrain"
            root = str(Path.home()) + "/label_checking/celebA-id/CelebA/"
            verificationPair = "namesLists/acceptableidentity_CelebA.txt"
            celeba = True
            epoch = 10

        elif datatype == "webcaricature":
            if not isClean:
                root = "image_alignment/" + cropType +"/" + "webcaricature_separated/"
            else:
                root = "image_alignment/" + cropType + "/" + "webcaricature_cleaned_separated/"
                # verificationPair = "verification_pairs-DONOTPUSH/webcaricature_verification_pairs_all.txt"
            verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_webcaricature.txt"
            celeba = False
            epoch = 100

            if not restricted and not MMO:
                modelName = "models/" + cropType +"/webcarUnrestrictedAll"+ dirEnd + "/run" + str(RunNum) + "_vgg-16_webcaricatureFinetune_" + str(
                    out)
            elif restricted and not MMO:

                modelName = "models/" + cropType +"/webcarRestrictedAll"+ dirEnd + "/run" + str(RunNum) + "_vgg-16_webcaricatureFinetune_" + str(
                    out)
            elif not restricted and MMO:

                modelName = "models/" + cropType +"/webcarUnrestrictedMMO"+ dirEnd + "/run" + str(RunNum) + "_vgg-16_webcaricatureFinetune_" + str(
                    out)
            elif restricted and MMO:

                modelName = "models/" + cropType +"/webcarRestrictedMMO"+ dirEnd + "/run" + str(RunNum) + "_vgg-16_webcaricatureFinetune_" + str(
                    out)
        elif "combined" in datatype:
            root = "image_alignment/" + cropType + "/" + datatype + "/"
            # verificationPair = "verification_pairs-DONOTPUSH/ourcar_verification_pairs_all.txt"
            verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_combined.txt"
            celeba = False

            if restricted and not MMO:
                modelName = "models/" + cropType +"/combinedRestrictedAll/run" + str(RunNum) + "_vgg-16_combinedFinetune_" + str(out)
            elif not restricted and not MMO:

                modelName = "models/" + cropType +"/combinedUnrestrictedAll/run" + str(RunNum) + "_vgg-16_combinedFinetune_" + str(out)
            elif restricted and MMO:

                modelName = "models/" + cropType +"/combinedRestrictedMMO/run" + str(RunNum) + "_vgg-16_combinedFinetune_" + str(out)
            elif not restricted and MMO:
                modelName = "models/" + cropType +"/combinedUnrestrictedMMO/run" + str(RunNum) + "_vgg-16_combinedFinetune_" + str(out)
        else:

            root = "image_alignment/" + cropType + "/" + datatype + "/"
            # verificationPair = "verification_pairs-DONOTPUSH/ourcar_verification_pairs_all.txt"
            verificationPair = "verification_pairs-DONOTPUSH/map_image_to_person_number_ourcar.txt"
            celeba = False

            if restricted and not MMO:
                modelName = "models/" + cropType +"/ourcarRestrictedAll/run" + str(RunNum) + "_vgg-16_ourcarFinetune_" + str(out)
            elif not restricted and not MMO:

                modelName = "models/" + cropType +"/ourcarUnrestrictedAll/run" + str(RunNum) + "_vgg-16_ourcarFinetune_" + str(out)
            elif restricted and MMO:

                modelName = "models/" + cropType +"/ourcarRestrictedMMO/run" + str(RunNum) + "_vgg-16_ourcarFinetune_" + str(out)
            elif not restricted and MMO:
                modelName = "models/" + cropType +"/ourcarUnrestrictedMMO/run" + str(RunNum) + "_vgg-16_ourcarFinetune_" + str(out)

        if eval_type == "val":
            dataloader, num = load_data(root, verificationPair, "val", fold=foldNum, celebA=celeba, transform=transform,
                                        invert=False, restricted=restricted, MMO=MMO, isClean=isClean)
            test = False
        elif eval_type == "train":
            dataloader, num = load_data(root, verificationPair, "train", fold=foldNum, celebA=celeba, transform=transform,
                                        invert=False, restricted=restricted, MMO=MMO,  isClean=isClean)
            test = False
        elif eval_type == "train_eval":
            dataloader, num = load_data(root, verificationPair, "train_eval", fold=foldNum, celebA=celeba, transform=transform,
                                        invert=False, restricted=restricted,MMO=MMO,  isClean=isClean)
            test = False
        else:
            print("dataloader", str(foldNum))
            dataloader, num = load_data(root, verificationPair, "test", fold=foldNum, celebA=celeba, transform=transform,
                                        invert=False, restricted=restricted, MMO=MMO,  isClean=isClean)

            test = True
    device = torch.device(device)
    if net == None:
        net = models.vgg16(pretrained=False)
        net.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=out, bias=True),
        )
        checkpoint = torch.load(modelName,  map_location=torch.device(device))  # may need to change after run
        net.load_state_dict(checkpoint)


    total_val_loss = 0
    criterion = torch.nn.CosineEmbeddingLoss()
    #eucDist = np.zeros(64)
    cosDist = np.zeros(bsize)
    NPlabels = np.zeros(bsize)
    with torch.no_grad():
        net.eval().to(device)
        for j, data in enumerate(dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1 = net(img0)
            output2 = net(img1)
            loss = criterion(output1.view(-1, out), output2.view(-1, out), label.view(-1))
            #euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            cosine_distance = F.cosine_similarity(output1.view(-1, out), output2.view(-1,out), dim=1)
            if j == 0:
                #eucDist = euclidean_distance.cpu().numpy()
                cosDist = cosine_distance.cpu().numpy()
                NPlabels = label.cpu().numpy()
            else:
                #eucDist = np.append(eucDist, euclidean_distance.cpu().numpy())
                cosDist = np.append(cosDist, cosine_distance.cpu().numpy())
                NPlabels = np.append(NPlabels, label.cpu().numpy())
            total_val_loss += loss.item()
            if debug:
                if j == 50:
                    break
        maxF1, bestDistance, bestTP, bestFP, bestTN, bestFN, bestPrec, bestRecall, bestAcc = 0, 0, 0, 0, 0, 0, 0, 0, 0

    #print(cosDist[:10])
    #print(NPlabels[:10])
    if run_thresholding:
        f1s = []
        indicesSort = cosDist.argsort(axis=None)
        indicesSort = np.flip(indicesSort, axis=None) #go smallest to largest for plot
        sortedCos = cosDist[indicesSort]
        #print(sortedCos[:10])
        val_tprs = []
        val_fprs = []
        outputNumpy = np.zeros((len(sortedCos), 6))
        sortedLabels = NPlabels[indicesSort]
        #print(sortedLabels[:10])
        for i, distance in enumerate(cosDist):
            # print(distance[0])
            preds = (sortedCos > distance).astype(int)
            #print(preds, sortedLabels)
            # print(preds)
            sortedLabels[sortedLabels == -1] = 0
            # print(preds)
            current_val_acc,current_val_tp, current_val_tn, current_val_fp, current_val_fn, current_val_tpr, current_val_fpr,\
            current_val_prec, current_val_recall,current_val_F1 = calculate_metrics (preds, sortedLabels)
            f1s.append(current_val_F1)
            val_tprs.append(current_val_tpr)
            val_fprs.append(current_val_fpr)

            if current_val_F1 > maxF1:
                maxF1, bestDistance, bestTP, bestTN, bestFP, bestFN, bestPrec, bestRecall, bestAcc, bestPreds = current_val_F1, \
                distance, current_val_tp, current_val_tn, current_val_fp, current_val_fn, current_val_prec, current_val_recall, current_val_acc, preds
            if "train" not in eval_type:
                outputNumpy[i, 0] = distance
                outputNumpy[i, 1] = current_val_prec
                outputNumpy[i, 2] = current_val_recall
                outputNumpy[i, 3] = current_val_F1
                outputNumpy[i, 4] = current_val_tpr
                outputNumpy[i, 5] = current_val_fpr
        f1s = sorted(f1s, reverse=True)
        #print(f1s[:10])

        auc, tpr1, tpr01 = calculate_ROC_metrics(dataSet=datatype, tprs =val_tprs, fprs = val_fprs, epoch=epoch, RunNum=RunNum, cropType=cropType, Test=test, Restricted=restricted, MMO=MMO, isClean=isClean, vgg = vgg)

        np.save(numpyName, outputNumpy)

    else:
        preds = (cosDist > 0).astype(int)
        NPlabels[NPlabels == -1] = 0
        bestAcc, bestTP, bestTN, bestFP, bestFN, tpr, fpr, bestPrec, bestRecall, maxF1 = calculate_metrics(preds, NPlabels)
        bestPreds, sortedLabels = preds, NPlabels
        auc, tpr1, tpr01 = 0, 0, 0
    if eval_gender:
        evaluate_gender(EvalList, bestPreds, sortedLabels, epoch, RunNum, cropType, webCar=webcar, restricted=restricted, MMO=MMO, isClean=isClean)
    if eval_pairs:
        evaluate_pair_composition(EvalList, bestPreds, sortedLabels, epoch, RunNum, cropType, webCar=webcar,
                        restricted=restricted, MMO=MMO, isClean=isClean)
    avg_val_loss = total_val_loss / len(dataloader)
    fout = open(fname, "a")
    fout.write("EPOCH {} \n".format(epoch+1))
    if eval_type == "val":
        fout.write("Avg_val_loss: " + str(avg_val_loss) + "\n")

        print(
            'Epoch [{}/{}], Valid Loss: {:.4f}, ThreshVal: {:.4f}, Valid Accuracy: {:.4f}, Valid F1: {:.4f}, AUC: {:.4f}, VR@1%: {:.4f}, VR@.01%: {:.4f}\n'
                .format(epoch + 1, 11, avg_val_loss, bestDistance, bestAcc, maxF1, auc, tpr1, tpr01))
    elif eval_type == "train":
        fout.write("Avg_train_loss: " + str(avg_val_loss) + "\n")

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, ThreshVal: {:.4f}, Train Accuracy: {:.4f}, Train F1: {:.4f}, AUC: {:.4f}, VR@1%: {:.4f}, VR@.01%: {:.4f}\n'
                .format(epoch + 1, 11, avg_val_loss, bestDistance, bestAcc, maxF1, auc, tpr1, tpr01))
    else:
        fout.write("Avg_test_loss: " + str(avg_val_loss) + "\n")

        print(
            'Epoch [{}/{}], Test Loss: {:.4f}, ThreshVal: {:.4f}, Test Accuracy: {:.4f}, Test F1: {:.4f}, AUC: {:.4f}, VR@1%: {:.4f}, VR@.01%: {:.4f}\n'
                .format(epoch + 1, 11, avg_val_loss, bestDistance, bestAcc, maxF1, auc, tpr1, tpr01))
    fout.write("Best Distance: " + str(bestDistance) + " TP: " + str(bestTP) + " TN: " + str(bestTN) + " FP: " + str(
        bestFP) + " FN: " + str(bestFN) + " Recall: " + str(bestRecall) +
               " Precision: " + str(bestPrec) + " F1: " + str(maxF1) + " Accuracy: " + str(bestAcc) + " AUC " + str(
        auc) + " VR@1% " + str(tpr1) + " VR@01% " + str(tpr01) + "\n")

    fout.close()
    return avg_val_loss, maxF1

#model_metrics_path =  "model_metrics/trashEval.txt"
#foldnums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""
datasets = ["datasets_combined_separated",  "ourcar", "webcaricature"]
alignments = ["bounding_box_based", "eye_location_based"]
MMOs = [True, False]
restricteds = [True, False]
cleans = [True, False]
for dataset in datasets:
    for alignment in alignments:
        for MMO in MMOs:
            for restricted in restricteds:
                if dataset == "webcaricature":
                    for clean in cleans:
                        for foldnum in foldnums:
                            Evaluate(foldnum, model_metrics_path, "test", dataset, alignment, None, None, 0, device="cuda:0", foldNum=foldnum, run_thresholding=True, debug=False, eval_gender = False, restricted=restricted, bsize=64, out=64, MMO=MMO, isClean = clean, eval_pairs=True)

                elif dataset != "datasets_combined_separated" and restricted != True and MMO != False:
                    for foldnum in foldnums:
                        Evaluate(foldnum, model_metrics_path, "test", dataset, alignment, None, None, 0, device="cuda:0",
                                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=restricted,
                                 bsize=64, out=64, MMO=MMO, isClean=False, eval_pairs=True)"""
'''
for foldnum in foldnums:r
    if foldnum != 1:
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "bounding_box_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "bounding_box_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "bounding_box_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "bounding_box_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "eye_location_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "eye_location_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "eye_location_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "ourcar", "eye_location_based", None, None, 0,
                 device="cuda:0", foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "bounding_box_based", None, None, 0, device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "bounding_box_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    #Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "bounding_box_based", None, None, 0,
    #         device="cuda:0",
    #         foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
    #         bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    if foldnum!= 1:
        Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "eye_location_based", None, None, 0, device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
        Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "eye_location_based", None, None, 0,
                 device="cuda:0",
                 foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
                 bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    #Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "eye_location_based", None, None, 0,
    #         device="cuda:0",
    #         foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
    #         bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "datasets_combined_separated", "eye_location_based", None, None, 0,
             device="cuda:0", foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=True, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0", foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=False, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=True, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=True, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=False, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "bounding_box_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=True, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=True, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0",
             foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=True,
             bsize=64, out=64, MMO=False, isClean=True, eval_pairs=True)
    Evaluate(foldnum, model_metrics_path, "test", "webcaricature", "eye_location_based", None, None, 0,
             device="cuda:0", foldNum=foldnum, run_thresholding=True, debug=False, eval_gender=False, restricted=False,
             bsize=64, out=64, MMO=False, isClean=True, eval_pairs=True)
'''


