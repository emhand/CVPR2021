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
    siamese_dataset = SiameseNetworkDataset(root="/home/sarad/label_checking/celebA-id/CelebA/",
                                            imageVerificationPairs="./namesLists/acceptableidentity_CelebA.txt", type="train", celebA = True,
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
    def __init__(self, root, imageVerificationPairs, type, fold = 0,celebA = False, transform=None, invert=True):
        if not celebA:
            f = open(imageVerificationPairs, "r")
            data = f.readlines()
            self.idImageDict = {}
            self.type = type
            if "ourcar" in imageVerificationPairs:
                self.datatype = 'O'
                if type == "train":
                    image_min = 0
                    image_max = 367
                elif type == "val":
                    image_min = 367
                    image_max = 412
                else:
                    image_min = 412
                    image_max = 10000000000
            else:
                #print(type)
                self.datatype = 'W'
                if fold == 0:
                    if type == "train":
                        image_min = 0
                        image_max = 455
                    elif type == "val":
                        image_min = 455
                        image_max = 10000000000
                    else:
                        image_min = 455
                        image_max = 10000000000
                else:
                    f = open("./webcaricatureFolds/foldpairs.txt", "r")
                    foldNum = fold-1
                    foldData = f.readlines()
                    folds = foldData[foldNum]
                    foldOrder = folds.split()
                    if type == "test":
                        foldinfo = [foldOrder[0]]
                    elif type == "val":
                        foldinfo = [foldOrder[0]]
                    else:
                        foldinfo = foldOrder[1:]
                    numbers = []
                    if type == "train":
                        for foldline in foldinfo:
                            fnew = open("./webcaricatureFolds/fold"+ foldline + ".txt")
                            data = fnew.readlines()
                            for line in data:
                                personNum = line.split()
                                num = personNum[-1]
                                numbers.append(num)
            if self.datatype == "W" and type != "train":
                availablePairFileName = "./webcaricatureFolds/fold" + str(fold) + "testPairs.txt"
                availablePairFile = open(availablePairFileName, "r")
                keysImages = availablePairFile.readlines()
                data = keysImages.split()
                im1 = data[1].replace(".jpg", "")
                im2 = data[3].replace(".jpg", "")
                if int(data[0]) not in self.idImageDict:
                    self.idImageDict[int(data[0])] = [int(im1), int(data[2]), int(im2)]
                else:
                    self.idImageDict[int(data[0])].append(int(im1), int(data[2]), int(im2))

                self.numPairs = len(keyImages)
            else:
                for imageID in data:
                    print(imageID)
                    image, id = imageID.split(".jpg")
                    person, imnum = image.split("/")
                    imnum = int(imnum)
                    #print(person, imnum)
                    id = int(id)
                    if fold == 0:
                        if int(id) >= image_min and int(id) < image_max:
                            if id not in self.idImageDict:
                                self.idImageDict[id] = [imnum]
                            else:
                                self.idImageDict[id].append(imnum)
                    else:
                        if int(id) in numbers:
                            if id not in self.idImageDict:
                                self.idImageDict[id] = [imnum]
                            else:
                                self.idImageDict[id].append(imnum)
                #print(id, image_min)
                imagesIDS = []
                for key, val in self.idImageDict.items():
                    imagesIDS.append([key] + val)
                self.images = []
                self.ids = []
                for idImage in imagesIDS:
                    allImages = idImage[1:]
                    for i in range(len(allImages)):
                        self.ids.append(int(idImage[0]))
                        self.images.append(allImages[i])
        else:
            f = open(imageVerificationPairs, "r")
            #print(imageVerificationPairs)
            data = f.readlines()
            self.idImageDict = {}
            self.type = type
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

        is_same = random.randint(0, 1)
        #dprint(self.images)
        img0 = self.images[index] #get the image at that index

        if self.celebA:
            id0 = self.find_id(img0) #find the id of the image in the dict
        else:
            id0 = self.find_id_other(index)
        #print("FIRST IMAGE", img0, id0)
        if is_same:
            id1 = id0
            label = 1
        else:
            available_ids = list(self.ids)
            #available_ids.remove(id0)
            available_ids = [x for x in available_ids if x != id0]

            id1 = available_ids[torch.randint(len(available_ids),(1,)).item()]
            label = -1
        possibleImages2 = self.idImageDict[id1]
        #print(possibleImages2)
        img1 = possibleImages2[torch.randint(len(possibleImages2),(1,)).item()]
        #print("SECOND IMAGE", img1, id1)

        if not self.celebA:
            img0 = str(img0)
            img1 = str(img1)

            lengthLacking0 = 5 - len(img0)
            lengthLacking1 = 5 - len(img1)
            toAdd0 = "0" * lengthLacking0
            toAdd1 = "0" * lengthLacking1
            img0 = toAdd0 + img0 + ".jpg"
            img1 = toAdd1 + img1 + ".jpg"


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



            #print(self.root)
            #print(im0)
            #print(im1)
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
            return img0, img1, torch.from_numpy(np.array([label], dtype=np.float32))
        else:
            #print(pathimg0, pathimg1, label)
            #return torch.from_numpy(np.array(img0,  dtype=np.float32)), torch.from_numpy(np.array(img1, dtype=np.float32)), torch.from_numpy(np.array([label], dtype=np.float32))
            return  img0, img1, torch.from_numpy(np.array([label], dtype=np.float32))
    def __len__(self):
        if not self.celebA:
            #return len(self.imagepairs)
            return (len(self.images))
        else:
            return(len(self.images))


#test_dataloader()
