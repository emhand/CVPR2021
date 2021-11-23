import torch
import torch.nn
import torch.nn.functional as F
#http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
    def forward(self, output1, output2, label):
        euclidean = F.pairwise_distance(output1, output2, keepdim=True)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean, 2) + (label) * torch.pow(torch.clamp(self.margin- euclidean, min=0.0), 2))

        return contrastive_loss