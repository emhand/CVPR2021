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
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    print(example_batch[0].shape)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    print(concatenated.shape)
    gridded = torchvision.utils.make_grid(concatenated)
    print(example_batch[2].numpy()) #print the labels
    imshow(gridded) #show the images


class SiameseNetworkDataset(Dataset):
    def __init__(self, root, imageVerificationPairs, type, celebA = False, transform=None, invert=True):
        if not celebA:
            f = open(imageVerificationPairs, "r")
            self.imagepairs = f.readlines() #open the txt file, and get all pairs im1 im2 label
            #self.imageID = [] #not used
        else:
            f = open(imageVerificationPairs, "r")
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
            #print(image_min, image_max)
            for imageID in data:
                image, id = imageID.split()
                imagenum = image.replace(".jpg", "")
                imagenum = int(imagenum)
                #print(int(imagenum))
                if int(imagenum) >= image_min and int(imagenum) < image_max:
                    if id not in self.idImageDict:
                        self.idImageDict[id] = [image]
                    else:
                        self.idImageDict[id].append(image)
            self.ids = list(self.idImageDict.keys())

        self.celebA = celebA
        self.transform = transform
        self.invert = invert
        self.root = root

    def __getitem__(self, index):
        if self.celebA:
            is_same = random.randint(0, 1)
            id0 = self.ids[index]
            possibleImages1 = self.idImageDict[id0]
            img0 = possibleImages1[torch.randint(len(possibleImages1), (1,)).item()]
            if is_same:
                id1 = id0
                label = 1
            else:
                available_ids = list(self.ids)
                available_ids.remove(id0)
                id1 = available_ids[torch.randint(len(available_ids),(1,)).item()]
                label = -1
            possibleImages2 = self.idImageDict[id1]
            img1 = possibleImages2[torch.randint(len(possibleImages2),(1,)).item()]

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
            id0 = "id-" + id0
            id1 = "id-" + id1
            pathimg0 = os.path.join(self.root,id0, img0)
            pathimg1 = os.path.join(self.root,id1, img1)

        #print(pathimg0)
        img0 = Image.open(pathimg0)
        img1 = Image.open(pathimg1)
        #img0 = img0.convert("L") #convert to greyscale
        #img1 = img1.convert("L")

        if self.invert:
            img0 = PIL.ImageOps.invert(img0) #invert colors
            img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            #print(pathimg0)
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
            return len(self.imagepairs)
        else:
            return len(self.ids)


#test_dataloader()
