# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# https://github.com/S-aiueo32/lpips-pytorch

import torch
import torchvision
from torch import nn
from kornia.filters.sobel import spatial_gradient



from lpips import LPIPS


def gram_matrix(x):
    a, b, c, d = x.size()
    act_x = x.reshape(x.shape[0], x.shape[1], -1)
    gram_x = 1. / (a * b * c * d) * act_x @ act_x.permute(0, 2, 1)
    return gram_x


class PerceptualLoss(nn.Module):

    def __init__(self, resize=True, content=True, style=True):
        super(PerceptualLoss, self).__init__()
        self.resize = resize
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.content = content
        self.style = style

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[0, 1, 2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if self.content and i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if self.style and i in style_layers:
                gram_x = gram_matrix(x)
                gram_y = gram_matrix(y)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class LPIPSWrapper(nn.Module):
    def __init__(self, net='alex'):
        super(LPIPSWrapper, self).__init__()
        self.criterion = LPIPS(net=net)

    def forward(self, x, y):
        # LPIPS expects inputs in range [-1, 1]
        return self.criterion(x*2. - 1., y*2. - 1.)

class AnimeLoss(nn.Module):

    def __init__(self, perceptual=True, lpips=True):
        super(AnimeLoss, self).__init__()
        self.base_loss = nn.SmoothL1Loss()
        if perceptual:
            if lpips:
                self.perceptual_loss = LPIPSWrapper(net='squeeze')
            else:
                self.perceptual_loss = PerceptualLoss(content=False)
        else:
            self.perceptual_loss = lambda x, y: torch.tensor(0.)

    def forward(self, prediction, target):
        from kornia.filters.sobel import spatial_gradient
        base_loss_value = self.base_loss(prediction, target)
        sobel_loss_value = (spatial_gradient(prediction) - spatial_gradient(target)).abs().mean()
        perceptual_loss_value = self.perceptual_loss(prediction, target).sum().item()
        # Rescaling losses to be roughly the same order of magnitude for each component
        loss_value = (base_loss_value * 1000) + (sobel_loss_value * 1000) + perceptual_loss_value
        # print("Loss: {:.2f}, Sobel Loss: {:.2f}, Perceptual Loss: {:.2f}".format(base_loss_value * 1000, sobel_loss_value * 1000, perceptual_loss_value))
        return loss_value
