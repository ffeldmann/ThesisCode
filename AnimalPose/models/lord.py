from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from edflow import get_logger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm


class LatentModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.content_embedding = RegularizedEmbedding(config['n_imgs'], config['content_dim'], config['content_std'])
        self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
        self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
        self.generator = Generator(config['content_dim'], config['n_adain_layers'], config['adain_dim'],
                                   config['img_shape'])

    def forward(self, img_id, class_id):
        content_code = self.content_embedding(img_id)
        class_code = self.class_embedding(class_id)
        class_adain_params = self.modulation(class_code)
        generated_img = self.generator(content_code, class_adain_params)

        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }

    def init(self):
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class AmortizedModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.content_encoder = Encoder(config['img_shape'], config['content_dim'])
        self.class_encoder = Encoder(config['img_shape'], config['class_dim'])
        self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
        self.generator = Generator(config['content_dim'], config['n_adain_layers'], config['adain_dim'],
                                   config['img_shape'])

    def forward(self, img):
        return self.convert(img, img)

    def convert(self, content_img, class_img):
        content_code = self.content_encoder(content_img)
        class_code = self.class_encoder(class_img)
        class_adain_params = self.modulation(class_code)
        generated_img = self.generator(content_code, class_adain_params)

        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }


class RegularizedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, stddev):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.stddev = stddev

    def forward(self, x):
        x = self.embedding(x)

        if self.training and self.stddev != 0:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.stddev)

            x = x + noise

        return x


class Modulation(nn.Module):

    def __init__(self, code_dim, n_adain_layers, adain_dim):
        super().__init__()

        self.__n_adain_layers = n_adain_layers
        self.__adain_dim = adain_dim

        self.adain_per_layer = nn.ModuleList([
            nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
            for _ in range(n_adain_layers)
        ])

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

        return adain_params


class Generator(nn.Module):

    def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
        super().__init__()

        self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
                out_features=self.__initial_height * self.__initial_width * adain_dim
            ),

            nn.LeakyReLU()
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(n_adain_layers):
            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
                nn.LeakyReLU(),
                AdaptiveInstanceNorm2d(adain_layer_idx=i)
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

        self.last_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
            nn.Sigmoid()
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_adain_params):
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x


class Encoder(nn.Module):

    def __init__(self, img_shape, code_dim):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_shape[-1], out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(256, code_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv_layers(x)
        x = x.view((batch_size, -1))

        x = self.fc_layers(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, adain_layer_idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.adain_layer_idx = adain_layer_idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True
        )

        out = out.view(b, c, *x.shape[2:])
        return out


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids)
        self.layer_ids = layer_ids

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


class Lord:

    def __init__(self, config=None):
        super().__init__()
        self.logger = get_logger(self)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_model = None
        self.amortized_model = None

    def load(self, model_dir, latent=True, amortized=True):
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
            self.config = pickle.load(config_fd)

        if latent:
            self.latent_model = LatentModel(self.config)
            self.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

        if amortized:
            self.amortized_model = AmortizedModel(self.config)
            self.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))

    def save(self, model_dir, latent=True, amortized=True):
        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
            pickle.dump(self.config, config_fd)

        if latent:
            torch.save(self.latent_model.state_dict(), os.path.join(model_dir, 'latent.pth'))

        if amortized:
            torch.save(self.amortized_model.state_dict(), os.path.join(model_dir, 'amortized.pth'))

    def train_latent(self, imgs, classes, model_dir, tensorboard_dir):
        self.latent_model = LatentModel(self.config)

        data = dict(
            img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs.shape[0])),
            class_id=torch.from_numpy(classes.astype(np.int64))
        )

        dataset = NamedTensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=self.config['train']['batch_size'],
            shuffle=True, sampler=None, batch_sampler=None,
            num_workers=1, pin_memory=True, drop_last=True
        )

        self.latent_model.init()
        self.latent_model.to(self.device)

        criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

        optimizer = Adam([
            {
                'params': itertools.chain(self.latent_model.modulation.parameters(),
                                          self.latent_model.generator.parameters()),
                'lr': self.config['train']['learning_rate']['generator']
            },
            {
                'params': itertools.chain(self.latent_model.content_embedding.parameters(),
                                          self.latent_model.class_embedding.parameters()),
                'lr': self.config['train']['learning_rate']['latent']
            }
        ], betas=(0.5, 0.999))

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['train']['n_epochs'] * len(data_loader),
            eta_min=self.config['train']['learning_rate']['min']
        )

        summary = SummaryWriter(log_dir=tensorboard_dir)

        train_loss = AverageMeter()
        for epoch in range(self.config['train']['n_epochs']):
            self.latent_model.train()
            train_loss.reset()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                optimizer.zero_grad()
                out = self.latent_model(batch['img_id'], batch['class_id'])

                content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
                loss = criterion(out['img'], batch['img']) + self.config['content_decay'] * content_penalty

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss.update(loss.item())
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(loss=train_loss.avg)

            pbar.close()
            self.save(model_dir, latent=True, amortized=False)

            summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)

            fixed_sample_img = self.generate_samples(dataset, randomized=False)
            random_sample_img = self.generate_samples(dataset, randomized=True)

            summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
            summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

        summary.close()

    def train_amortized(self, imgs, classes, model_dir, tensorboard_dir):
        self.amortized_model = AmortizedModel(self.config)
        self.amortized_model.modulation.load_state_dict(self.latent_model.modulation.state_dict())
        self.amortized_model.generator.load_state_dict(self.latent_model.generator.state_dict())

        data = dict(
            img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs.shape[0])),
            class_id=torch.from_numpy(classes.astype(np.int64))
        )

        dataset = NamedTensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=self.config['train']['batch_size'],
            shuffle=True, sampler=None, batch_sampler=None,
            num_workers=1, pin_memory=True, drop_last=True
        )

        self.latent_model.to(self.device)
        self.amortized_model.to(self.device)

        reconstruction_criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
        embedding_criterion = nn.MSELoss()

        optimizer = Adam(
            params=self.amortized_model.parameters(),
            lr=self.config['train_encoders']['learning_rate']['max'],
            betas=(0.5, 0.999)
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['train_encoders']['n_epochs'] * len(data_loader),
            eta_min=self.config['train_encoders']['learning_rate']['min']
        )

        summary = SummaryWriter(log_dir=tensorboard_dir)

        train_loss = AverageMeter()
        for epoch in range(self.config['train_encoders']['n_epochs']):
            self.latent_model.eval()
            self.amortized_model.train()

            train_loss.reset()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                optimizer.zero_grad()

                target_content_code = self.latent_model.content_embedding(batch['img_id'])
                target_class_code = self.latent_model.class_embedding(batch['class_id'])

                out = self.amortized_model(batch['img'])

                loss_reconstruction = reconstruction_criterion(out['img'], batch['img'])
                loss_content = embedding_criterion(out['content_code'], target_content_code)
                loss_class = embedding_criterion(out['class_code'], target_class_code)

                loss = loss_reconstruction + 10 * loss_content + 10 * loss_class

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss.update(loss.item())
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(loss=train_loss.avg)

            pbar.close()
            self.save(model_dir, latent=False, amortized=True)

            summary.add_scalar(tag='loss-amortized', scalar_value=loss.item(), global_step=epoch)
            summary.add_scalar(tag='rec-loss-amortized', scalar_value=loss_reconstruction.item(), global_step=epoch)
            summary.add_scalar(tag='content-loss-amortized', scalar_value=loss_content.item(), global_step=epoch)
            summary.add_scalar(tag='class-loss-amortized', scalar_value=loss_class.item(), global_step=epoch)

            fixed_sample_img = self.generate_samples_amortized(dataset, randomized=False)
            random_sample_img = self.generate_samples_amortized(dataset, randomized=True)

            summary.add_image(tag='sample-fixed-amortized', img_tensor=fixed_sample_img, global_step=epoch)
            summary.add_image(tag='sample-random-amortized', img_tensor=random_sample_img, global_step=epoch)

        summary.close()

    def generate_samples(self, dataset, n_samples=5, randomized=False):
        self.latent_model.eval()

        if randomized:
            random = np.random
        else:
            random = np.random.RandomState(seed=1234)

        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

        samples = dataset[img_idx]
        samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

        blank = torch.ones_like(samples['img'][0])
        output = [torch.cat([blank] + list(samples['img']), dim=2)]
        for i in range(n_samples):
            converted_imgs = [samples['img'][i]]

            for j in range(n_samples):
                out = self.latent_model(samples['img_id'][[j]], samples['class_id'][[i]])
                converted_imgs.append(out['img'][0])

            output.append(torch.cat(converted_imgs, dim=2))

        return torch.cat(output, dim=1)

    def generate_samples_amortized(self, dataset, n_samples=5, randomized=False):
        self.amortized_model.eval()

        if randomized:
            random = np.random
        else:
            random = np.random.RandomState(seed=1234)

        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

        samples = dataset[img_idx]
        samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

        blank = torch.ones_like(samples['img'][0])
        output = [torch.cat([blank] + list(samples['img']), dim=2)]
        for i in range(n_samples):
            converted_imgs = [samples['img'][i]]

            for j in range(n_samples):
                out = self.amortized_model.convert(samples['img'][[j]], samples['img'][[i]])
                converted_imgs.append(out['img'][0])

            output.append(torch.cat(converted_imgs, dim=2))

        return torch.cat(output, dim=1)
