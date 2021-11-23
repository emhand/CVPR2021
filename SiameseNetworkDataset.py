from torch.utils.data import DataLoader, Dataset
import PIL
from PIL import Image
import random
import torch
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt
#https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#https://stackoverflow.com/questions/58834338/how-does-the-getitem-s-idx-work-within-pytorchs-dataloader



def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def test_dataloader():
    #siamese_dataset = SiameseNetworkDataset(root="./our_car/new-cropped/",
    #                                        imageVerificationPairs="./verification_pairs/ourcar_verification_pairs_all.txt", celebA = False,
    #                                        transform=None, invert=False)
    siamese_dataset = SiameseNetworkDataset(root="/home/sarad/label_checking/celebA-id/CelebA/",
                                            imageVerificationPairs="./namesLists/acceptableidentity_CelebA.txt", celebA = True,
                                            transform=None, invert=False)
    #data1, data2, label, ids = siamese_dataset[0]
    #print(data1, data2, label, ids)
    vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 1)
    concatenated = concatenated.permute(0, 3, 1, 2)
    gridded = torchvision.utils.make_grid(concatenated)
    print(example_batch[2].numpy()) #print the labels
    print(example_batch[3])
    imshow(gridded) #show the images


class SiameseNetworkDataset(Dataset):
    def __init__(self, root, imageVerificationPairs, identities, type, celebA = False, transform=None, invert=True):
        if not celebA:
            f = open(imageVerificationPairs, "r")
            self.imagepairs = f.readlines() #open the txt file, and get all pairs im1 im2 label
            self.imageID = [] #not used
        else:
            f = open(imageVerificationPairs, "r")
            self.imagepairs = [] #not used
            self.imageID = f.readlines()
        self.identities = identities
        self.celebA = celebA
        self.transform = transform
        self.invert = invert
        self.root = root

    def __getitem__(self,index):
        if self.celebA:
            numData = len(self.imageID)
            is_same = random.randint(0, 1)
            imageID0 = self.imageID[random.randint(0, numData - 1)]
            img0, id0 = imageID0.split()
            if is_same:
                while True:
                    imageID1 = self.imageID[random.randint(0, numData-1)]
                    img1, id1 = imageID1.split()
                    if id0 == id1:
                        self.identities[id0] = 0
                        self.identities[id1] = 0
                        id0 = "id-" + str(id0)
                        id1 = "id-" + id1
                        label = "1"
                        break
            else:
                while True:
                    imageID1 = self.imageID[random.randint(0, numData - 1)]
                    img1, id1 = imageID1.split()
                    if id0 != id1:

                        self.identities[id0] = 0
                        self.identities[id1] = 0
                        id0 = "id-" + str(id0)
                        id1 = "id-" + id1
                        label = "0"
                        break
            #print(self.identities)
            #print(img0, img1, label)
        else:
            #pick a random image pair
            img0_tuple = random.choice(self.imagepairs)
            #print(index)
            #tries to do a 50/50 balance of 0/1 pairs
            #print(img0_tuple)
            img0, img1, label = img0_tuple.split()
        if not self.celebA:
            pathimg0 = os.path.join(self.root, img0)
            pathimg1 = os.path.join(self.root, img1)
        else:
            pathimg0 = os.path.join(self.root,id0, img0)
            pathimg1 = os.path.join(self.root,id1, img1)
        #print(pathimg0, pathimg1)
        #print(pathimg0)
        img0 = Image.open(pathimg0)
        img1 = Image.open(pathimg1)
        #img0 = img0.convert("L") #convert to greyscale
        #img1 = img1.convert("L")

        if self.invert:
            img0 = PIL.ImageOps.invert(img0) #invert colors
            img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0) #calls transform fxn
            img1 = self.transform(img1)
        #return image vectors and labels
        #torch_numpy = torch.from_numpy(np.array(img0))
        #print(torch_numpy.size())
        #print(label)
        num_vals = [el for el in self.identities if self.identities.values() == 0]
        startNewEpoch = False
        if len(num_vals) == len(self.identities):
            self.identities = dict.fromkeys(self.identities, 1)
            startNewEpoch = True
        #print(self.celebA)
        if not self.celebA:
            return torch.from_numpy(np.array(img0)), torch.from_numpy(np.array(img1)), torch.from_numpy(np.array([label], dtype=np.float32))
        else:
            #print(self.identities)
            return torch.from_numpy(np.array(img0,  dtype=np.float32)), torch.from_numpy(np.array(img1, dtype=np.float32)), torch.from_numpy(np.array([label], dtype=np.float32)), torch.from_numpy(np.array([startNewEpoch]))
    def __len__(self):
        if not self.celebA:
            return len(self.imagepairs)
        else:
            return len(self.imageID)


#test_dataloader()