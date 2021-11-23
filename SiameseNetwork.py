import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),#3 is num im channels in, 64 is num channels out, 3*3 is filter
            nn.ReLU(inplace=True), #inplace has it do it directly within the tensor
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #64 channels in from last layer, 64 out, still 3*3 filter
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #64 channels in from max, 128 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #128 channels in from conv, 128 out now
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #128 from max, 256 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #256 from conv, 256 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),  #256 from conv, 256 out now
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #256 in from max, 512 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)), #512 in from conv, 512 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 512 in from conv, 512 out now
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=64, bias=True),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5, inplace=False),
            #nn.Linear(in_features=4096, out_features=4096, bias=True)#,
            #I don't think we need these, because they do classification. Instead output a nice linear pred layer

            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5, inplace=False),
            #nn.Linear(in_features=4096, out_features=1000, bias=True)
        )

    def forward_once(self, image):
        #print(image)
        output = self.features(image) #run the data through, get the output
        output = self.avgpool(output)
        #output = output.view(output.size()[1], -1) #reshape it
        output = torch.flatten(output, 1)
        output = self.classifier(output) #run the data through linear transform, get output
        return output

    def forward(self, image1, image2):
        output1 = self.forward_once(image1)
        output2 = self.forward_once(image2)
        #print(output1.shape, output2.shape)
        return output1, output2

