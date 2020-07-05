import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader

from itertools import cycle
from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor

# compose a transform configuration
transform_config = Compose([ToTensor()])


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """
    Plot images in a grid of a given shape.
    Initial code from: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    """
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()


class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Encoder, self).__init__()

        self.conv_model = nn.Sequential(OrderedDict([
            ('convolution_1',
             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_1_in', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('ReLU_1', nn.ReLU(inplace=True)),

            ('convolution_2',
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_2_in', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('ReLU_2', nn.ReLU(inplace=True)),

            ('convolution_3',
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_3_in', nn.InstanceNorm2d(num_features=64, track_running_stats=True)),
            ('ReLU_3', nn.ReLU(inplace=True))
        ]))

        # Style embeddings
        self.style_mu = nn.Linear(in_features=256, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=256, out_features=style_dim, bias=True)

        # Class embeddings
        self.class_output = nn.Linear(in_features=256, out_features=class_dim, bias=True)

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        style_embeddings_mu = self.style_mu(x)
        style_embeddings_logvar = self.style_logvar(x)
        class_embeddings = self.class_output(x)

        return style_embeddings_mu, style_embeddings_logvar, class_embeddings


class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        # Style embeddings input
        self.style_input = nn.Linear(in_features=style_dim, out_features=256, bias=True)

        # Class embeddings input
        self.class_input = nn.Linear(in_features=class_dim, out_features=256, bias=True)

        self.deconv_model = nn.Sequential(OrderedDict([
            ('deconvolution_1',
             nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True)),
            ('deconvolution_1_in', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('deconvolution_2',
             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)),
            ('deconvolution_2_in', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('LeakyReLU_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('deconvolution_3',
             nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True)),
            ('sigmoid_final', nn.Sigmoid())
        ]))

    def forward(self, style_embeddings, class_embeddings):
        style_embeddings = F.leaky_relu_(self.style_input(style_embeddings), negative_slope=0.2)
        class_embeddings = F.leaky_relu_(self.class_input(class_embeddings), negative_slope=0.2)

        x = torch.cat((style_embeddings, class_embeddings), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self.deconv_model(x)

        return x
from AnimalPose.models.keypoint_predictors import ResPoseNet
class CycleVAE(nn.Module):
    def __init__(self, config):
        super(ResPoseNet, self).__init__()
        # Loading weight etc done in ResPosenet here we add the style embeddings and class embeddings

        self.style_dim = self.config["style_dim"] # TODO
        self.class_dim = self.config["class_dim"]

        #BACKBONE

        # Style embeddings
        self.backbone_style_mu = nn.Linear(in_features=256, out_features=self.style_dim, bias=True)
        self.backbone_style_logvar = nn.Linear(in_features=256, out_features=self.style_dim, bias=True)

        # Class embeddings
        self.backbone_class_output = nn.Linear(in_features=256, out_features=self.class_dim, bias=True)


        # HEAD
        # Style embeddings input
        self.head_style_input = nn.Linear(in_features=self.style_dim, out_features=256, bias=True)

        # Class embeddings input
        self.head_class_input = nn.Linear(in_features=self.class_dim, out_features=256, bias=True)


    def forward(self, x):
        #HEAD
        x = self.conv_model(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        style_embeddings_mu = self.style_mu(x)
        style_embeddings_logvar = self.style_logvar(x)
        class_embeddings = self.class_output(x)

        return style_embeddings_mu, style_embeddings_logvar, class_embeddings
        style_latent_space = reparameterize(training=True, mu=mu, logvar=logvar)
        # HEAD
        style_embeddings = F.leaky_relu_(self.style_input(style_embeddings), negative_slope=0.2)
        class_embeddings = F.leaky_relu_(self.class_input(class_embeddings), negative_slope=0.2)

        x = torch.cat((style_embeddings, class_embeddings), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self.head(x)
        return


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=256, bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=256, out_features=256, bias=True)),
            ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=256, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)

        return x


if __name__ == '__main__':
    """
    test for network outputs
    """
    encoder = Encoder(16, 16)
    decoder = Decoder(16, 16)

    classifier = Classifier(z_dim=16, num_classes=10)

    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0, drop_last=True))

    image_batch, labels_batch = next(loader)

    mu, logvar, class_latent_space = encoder(Variable(image_batch))
    style_latent_space = reparameterize(training=True, mu=mu, logvar=logvar)

    reconstructed_image = decoder(style_latent_space, class_latent_space)
    classifier_pred = classifier(style_latent_space)

    print(reconstructed_image.size())
    print(classifier_pred.size())
