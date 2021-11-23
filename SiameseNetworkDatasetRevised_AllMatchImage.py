from torch.utils.data import DataLoader, Dataset
import PIL
from PIL import Image
import random
import torch
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from functools import reduce
#https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#https://stackoverflow.com/questions/58834338/how-does-the-getitem-s-idx-work-within-pytorchs-dataloader

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    plt.close()


def test_dataloader():
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),  # default
        #transforms.RandomResizedCrop((178, 218), scale=(0.95, 1.05), ratio=(0.75, 1.25), interpolation=2),
        #transforms.RandomRotation((-5, 5)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    siamese_dataset = SiameseNetworkDataset(root="/home/sarad/BMVC2021/image_alignment/bounding_box_based/webcaricature_separated/",
                                            imageVerificationPairs="./verification_pairs-DONOTPUSH/map_image_to_person_number_webcaricature.txt", fold = 0, type="val", celebA = False,
                                            transform=transform, invert=False)

    vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)
    #dataiter = iter(vis_dataloader)
    #example_batch = next(dataiter)
    #print(len(dataiter), len(dataiter[0]))
    #print(example_batch[0].shape)
    for example_batch in vis_dataloader:
        concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
        #print(concatenated.shape)
        gridded = torchvision.utils.make_grid(concatenated)
        print(example_batch[2].numpy()) #print the labels
        imshow(gridded) #show the images


class SiameseNetworkDataset(Dataset):
    def __init__(self, root, imageVerificationPairs, type, fold = 0,celebA = False, transform=None, invert=True, restricted=False, MMO = False, isClean = False):
        self.doTest = False
        #print(root, imageVerificationPairs, type, celebA)
        if not celebA:
            #f = open(imageVerificationPairs, "r")
            #data = f.readlines()
            self.idImageDict = {}
            self.type = type
            if "ourcar" in imageVerificationPairs:
                print("LOADING OURCAR")
                self.datatype = 'O'
                beginning_append = "ourcar"
            elif not isClean and "webcaricature" in imageVerificationPairs:
                self.datatype = 'W'
                beginning_append="webcaricature"
                print("LOADING WEBCAR")
            elif "combined" in imageVerificationPairs:
                print("LOADING COMBINED")
                self.datatype = "C"
                beginning_append = "combined"
            else:
                self.datatype = "W"
                beginning_append = "webcaricature"
                print("LOADING CLEANED WEBCAR")


            self.doTest = True
            filesToOpen = []
            if not isClean:
                endDir = ""
            else:
                endDir = "_cleaned"
            if restricted and MMO:
                foldDir = "./" + beginning_append + "Folds_restrictedMMO" + endDir + "/fold"
            elif restricted and not MMO:
                foldDir = "./" + beginning_append + "Folds_restrictedAll" + endDir + "/fold"

            elif not restricted and not MMO:
                foldDir = "./" + beginning_append + "Folds_unrestrictedAll" + endDir + "/fold"
            elif  not restricted and MMO:
                foldDir = "./" + beginning_append + "Folds_unrestrictedMMO" + endDir + "/fold"
            if type != "train" and type != "train_eval":
                availablePairFileName = foldDir + str(fold) + "testPairs.txt"
                filesToOpen.append(availablePairFileName)
                print(availablePairFileName)
            elif fold == 0 and type != "train_eval":
                availablePairFileName = foldDir + "0trainPairs.txt"
                print(availablePairFileName)
                filesToOpen.append(availablePairFileName)
            elif fold == 0 and type == "train_eval" :
                availablePairFileName = foldDir + "0evaluatetrainPairs.txt"
                filesToOpen.append(availablePairFileName)
                print(availablePairFileName)
            elif (type == "train" or type ==  "train_eval") and fold != 0:
                possibleFolds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                possibleFolds.remove(fold)
                for val in possibleFolds:
                    file_to_open =foldDir + str(val) + "testPairs.txt"
                    print(file_to_open)
                    filesToOpen.append(file_to_open)

            keyImKeyImComposite = []
            for file in filesToOpen:
                availablePairFile = open(file, "r")
                keyImKeyIm = availablePairFile.readlines()
                if type == "train_eval" and fold != 0:
                    keyImKeyIm = keyImKeyIm[:7000] #create subset
                if len(filesToOpen) > 1:
                    keyImKeyImComposite += keyImKeyIm
                else:
                    keyImKeyImComposite = keyImKeyIm
            self.images =[]
            self.ids = []
            for pair in keyImKeyImComposite:
                person1, im1, person2, im2 = pair.split()
                im1Num = im1.replace(".jpg", "")
                im2Num = im2.replace(".jpg", "")
                imPair = [int(im1Num), int(im2Num)]
                self.images.append(imPair)
                #self.images.append(int(im2))
                idPair = [int(person1), int(person2)]
                self.ids.append(idPair)
        else:
            print("LOADING CELEBA")

            f = open(imageVerificationPairs, "r")
            #print(imageVerificationPairs)
            data = f.readlines()
            #print(data)
            self.idImageDict = {}
            self.type = type
            if self.type == "train_eval":
                self.type = "train"
            if type == "train":
                image_min = 0
                image_max = 162771
            elif type == "val":
                image_min = 162771
                image_max = 182638
            elif type == "test":
                image_min = 182638
                image_max = 100000000
            for imageID in data:
                image, id = imageID.split()
                imagenum = image.replace(".jpg", "")
                imagenum = int(imagenum)
                if int(imagenum) >= image_min and int(imagenum) < image_max:
                    if id not in self.idImageDict:
                        self.idImageDict[id] = [image]
                    else:
                        self.idImageDict[id].append(image) #still need a dictionary if ID's:images to pair later
            #print(self.idImageDict)
            images = list(self.idImageDict.values()) #Instead of list of ID's to iterate, we need ims
            self.images = reduce(lambda z, y : z + y, images)
            self.ids = list(self.idImageDict.keys())



        self.celebA = celebA
        self.transform = transform
        self.invert = invert
        self.root = root
    def find_id(self, img):
        for id, images in self.idImageDict.items():
            #print(img)
            #print(id, images)
            if img in images:
                return id

    def find_id_other(self, index):
        return self.ids[index]

    def __getitem__(self, index):
        #print(self.datatype, self.doTest)
        if not self.doTest:
            is_same = random.randint(0, 1)
            #dprint(self.images)
            img0 = self.images[index] #get the image at that index

            if self.celebA:
                id0 = self.find_id(img0) #find the id of the image in the dict
            else:
                id0 = self.find_id_other(index)
            #print("FIRST IMAGE", img0, id0)
            if is_same and not self.celebA:
                is_same_opp = random.randint(0, 1)
                if is_same_opp == 1:
                    id1 = id0
                elif id0 %2 == 0:
                    id1 = id0 - 1 #swap label become real
                else:
                    id1 = id0 + 1 #swap label become car
                label = 1
            elif not self.celebA:
                available_ids = list(self.ids)
                #available_ids.remove(id0)
                available_ids = [x for x in available_ids if x != id0]

                id1 = available_ids[torch.randint(len(available_ids),(1,)).item()]
                label = -1
            elif is_same and self.celebA:
                id1 = id0
                label = 1
            elif not is_same and self.celebA:
                available_ids = list(self.ids)
                available_ids = [x for x in available_ids if x != id0]
                id1 = available_ids[torch.randint(len(available_ids), (1,)).item()]
                label = -1
            possibleImages2 = self.idImageDict[id1]
            #print(possibleImages2)
            img1 = possibleImages2[torch.randint(len(possibleImages2),(1,)).item()]
            #print("SECOND IMAGE", img1, id1)

            if not self.celebA:
                img0 = str(img0)
                img1 = str(img1)
                #print(self.datatype)
                if self.datatype == "W":
                    lengthLacking0 = 5 - len(str(img0))
                    lengthLacking1 = 5 - len(str(img1))
                    toAdd0 = "0" * lengthLacking0
                    toAdd1 = "0" * lengthLacking1
                    img0 = toAdd0 + img0 + ".jpg"
                    img1 = toAdd1 + img1 + ".jpg"
                else:
                    if len(img0) < 2:
                        img0 = "0" + img0 + ".jpg"
                    else:
                        img0 = img0 + ".jpg"
                    if len(img1) < 2:
                        img1 = "0" + img1 + ".jpg"
                    else:
                        img1 = img1 + ".jpg"


                if self.datatype == "O":
                    f = open("./verification_pairs-DONOTPUSH/map_person_to_number_ourcar.txt", "r")
                else:
                    f = open("./verification_pairs-DONOTPUSH/map_person_to_number_webcaricature.txt", "r")
                data = f.readlines()
                path0 = None
                path1 = None
                for line in data:

                    personNumber = line.split()
                    id = personNumber[-1]
                    person = personNumber[:-1]
                    name = " "
                    nameDir = name.join(person)
                    if int(id) == int(id0):
                        #print(id, id0)
                        path0 = nameDir + "/" + img0
                    if int(id) == int(id1):
                        path1 = nameDir + "/" + img1
                        #print(id, id1)
                    if path0 != None and path1 != None:
                        break
                pathimg0 = os.path.join(self.root, path0)
                pathimg1 = os.path.join(self.root, path1)
                #print(pathimg0)
                #print(pathimg1)
            else:
                id0 = "id-" + id0
                id1 = "id-" + id1
                #print(self.root, id0, img0)
                pathimg0 = os.path.join(self.root,id0, img0)
                pathimg1 = os.path.join(self.root,id1, img1)
            #print(pathimg0, pathimg1, label)
            img0 = Image.open(pathimg0)
            img1 = Image.open(pathimg1)
        else:

             pairID = self.ids[index]
             #print(pairID)
             id0, id1 = pairID[0], pairID[1]
             pairIm = self.images[index]
             #print(len(self.ids), len(self.images))
             #print(pairID, pairIm)
             im0, im1 = pairIm[0], pairIm[1]
             im0, im1 = str(im0), str(im1)
             #print(im0, im1)
             if self.datatype != "O":
                 im0ToAdd = 5-len(im0)
                 im1ToAdd = 5-len(im1)
                 im0 = "0" * im0ToAdd + im0 + ".jpg"
                 im1 = "0" * im1ToAdd + im1 + ".jpg"
             else:
                 #print(im0, im1)
                 if len(im0) < 2:
                     im0 = "0" + im0 +".jpg"
                 else:
                     im0 = im0 + ".jpg"
                 if len(im1) < 2:
                     im1 = "0" + im1 + ".jpg"
                 else:
                     im1 = im1 + ".jpg"
             #print(im0, im1)
             if self.datatype == "O":
                 f = open("./verification_pairs-DONOTPUSH/map_person_to_number_ourcar.txt", "r")
             elif self.datatype == "C":
                 f = open("./verification_pairs-DONOTPUSH/map_person_to_number_combined.txt", "r")
             else:
                 f = open("./verification_pairs-DONOTPUSH/map_person_to_number_webcaricature.txt", "r")
             data = f.readlines()
             path0 = None
             path1 = None
             for line in data:
                 personNumber = line.split()
                 id = personNumber[-1]
                 person = personNumber[:-1]
                 name = " "
                 nameDir = name.join(person)
                 if int(id) == id0:
                     path0 = self.root + nameDir + "/" + im0
                 if int(id) == id1:
                     path1 = self.root + nameDir + "/"+  im1
                 if path0 != None and path1 != None:
                     break
             #modulo makes sure we subtract 1 from caricature vals, so that caricature ids are considered same as real
             #print(path0, path1)
             if pairID[0] %2 ==0:
                 id0 = pairID[0] -1
             if pairID[1] %2 ==0:
                 id1 = pairID[1] -1
             if id0 == id1:
                  label = 1
             else:
                  label = -1
             #print(path0, path1)
             img0 = Image.open(path0)
             img1 = Image.open(path1)

        if self.invert:
            img0 = PIL.ImageOps.invert(img0) #invert colors
            img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            typeIm0 = img0.mode
            typeIm1 = img1.mode
            if typeIm0 != "RGB":
                img0 = img0.convert("RGB")
            if typeIm1 != "RGB":
                img1 = img1.convert("RGB")

            img0 = self.transform(img0) #calls transform fxn
            img1 = self.transform(img1)

        if not self.celebA:
            #return torch.from_numpy(np.array(img0)), torch.from_numpy(np.array(img1)), torch.from_numpy(np.array([label], dtype=np.float32))
            return img0, img1, torch.from_numpy(np.array([label], dtype=np.int16))
        else:
            #print(pathimg0, pathimg1, label)
            #return torch.from_numpy(np.array(img0,  dtype=np.float32)), torch.from_numpy(np.array(img1, dtype=np.float32)), torch.from_numpy(np.array([label], dtype=np.float32))
            return  img0, img1, torch.from_numpy(np.array([label], dtype=np.int16))
    def __len__(self):
        if not self.celebA:
            #return len(self.imagepairs)
            return (len(self.images))
        else:
            return(len(self.images))


#test_dataloader()
