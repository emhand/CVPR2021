#for vggface dataset: https://github.com/ndaidong/vgg-faces-utils
#verfication example: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
#pytorch imp of vgg net: https://pytorch.org/hub/pytorch_vision_vgg/
#original vgg: https://www.robots.ox.ac.uk/~vgg/research/very_deep/
import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision import transforms
from PIL import Image
import math

#pretrain of webcar should be verification as well
#can subsample to balance yes and no (20/80 and save to the file of pairs with labels

def pretrain_model():
    model = vgg16(pretrained=True)
    ###TRAIN ON VGGFACE-WEBCARICATURE DATA TO MAP TO SAME SPACE###

    return model

def modify_model(model):
    ###RIP OFF LAST LAYER OF NET###
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias = True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True)
    )
    return model
#print(model)
####FOR TRIPLET LOSS, IGNORING FOR NOW####
###FREEZE ALL LAYERS EXCEPT LAST###
#for name, layer in model.named_parameters():
#    if "classifier.3" not in name:
#        layer.requires_grad = False
#        print("freezing")
#    else:
#        print("not frozen")

def preprocess(samples):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(samples)
    return input_tensor
def extract_face(face):
    image = Image.open(face)
    image = np.asarray(image)
    return image

def get_embeddings(filenames, model):
    #####for every image in webcaricature batch, get embeddings#####
    faces = [extract_face(f) for f in filenames]
    samples = np.asarray(faces, 'float32')
    #do some preprocessing here
    samples = preprocess(samples)
    model = modify_model(model)
    model.eval()
    with torch.no_grad():
        yhat = model(samples)
    return yhat
def is_match(embedding1, embedding2, thresh=0.5):
    score = math.cosine(embedding1, embedding2)
    if score <= thresh:
        #is match
        return 1
    else:
        #is not match
        return 0

######for every pair of images in webcaricature, check if embeddings are match####


