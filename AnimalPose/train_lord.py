import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torchvision.utils
from edflow import TemplateIterator
from edflow import get_logger
from edflow.data.util import adjust_support

from AnimalPose.data.util import make_stickanimal
from AnimalPose.utils.image_utils import heatmaps_to_coords, heatmaps_to_image
from AnimalPose.utils.log_utils import plot_input_target_keypoints
from AnimalPose.utils.loss_utils import percentage_correct_keypoints, VGGLossWithL1
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy
from AnimalPose.utils.perceptual_loss.models import PerceptualLoss

import torchvision.transforms as transforms


class LordIteratorAmortized(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # Initialize Loss functions
        self.mse_loss = torch.nn.MSELoss()
        # self.mse_instance = MSELossInstances()
        # self.l1_instance = L1LossInstances()
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        # Imagenet Mean
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        if self.cuda:
            self.model.cuda()

        self.freeze_encoder()

        # vgg loss
        if self.config["losses"]["vgg"]:
            self.vggL1 = VGGLossWithL1(gpu_ids=[0],
                                       l1_alpha=self.config["losses"]["vgg_l1_alpha"],
                                       vgg_alpha=self.config["losses"]["vgg_alpha"]).to(self.device)

        # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)

        # hooks
        # if self.config["adjust_learning_rate"]:
        #    self.hooks.append(
        #        AdjustLearningRate(self.config, self.optimizer)
        #    )

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def criterion(self, targets, predictions):
        # make sure everything is a torch tensor
        targets = sure_to_torch(targets)
        predictions = sure_to_torch(predictions)

        batch_losses = {}
        if self.config["losses"]["L2"]:
            batch_losses["L2"] = self.mse_loss(targets, predictions.cpu())
            batch_losses["L2"] += self.mse_loss(
                torch.from_numpy(heatmaps_to_image(targets.numpy())),
                torch.from_numpy(heatmaps_to_image(predictions.detach().cpu().numpy())))
        if self.config["losses"]["perceptual"]:
            batch_losses["perceptual"] = torch.mean(
                self.perceptual_loss(torch.from_numpy(heatmaps_to_image(targets.numpy())),
                                     torch.from_numpy(heatmaps_to_image(predictions.detach().cpu().numpy())),
                                     True)).cpu()
        batch_losses["total"] = sum(
            [
                batch_losses[key]
                for key in batch_losses.keys()
            ]
        )
        return batch_losses

    def freeze_encoder(self):
        self.logger.info("Encoder freezed!")
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.logger.info("Encoder unfreezed!")
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)
        if self.get_global_step() == self.config["freeze_encoder_for"] and is_train:
            self.unfreeze_encoder()

        # kwargs["inp"]
        # (batch_size, width, height, channel)
        inputs = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2))
        # (batch_size, channel, height, width)
        batch_size, channel, height, width = inputs.shape
        if self.config["pretrained"]:
            normalized = torch.zeros((batch_size, channel, height, width), dtype=torch.float32)
            for idx, element in enumerate(inputs):
                normalized[idx, :, :, :] = self.normalize(element)
            # compute model
            predictions = model(normalized.to(self.device))
        else:
            predictions = model(inputs.to(self.device))

        # compute loss
        losses = self.criterion(kwargs["targets"], predictions)

        def train_op():
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

        def log_op():

            PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]
            HM_THRESH = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if self.config['pck_alpha'] not in PCK_THRESH: PCK_THRESH.append(self.config["pck_alpha"])

            coords, _ = heatmaps_to_coords(predictions.clone(), thresh=self.config["hm"]["thresh"])
            pck = {t: percentage_correct_keypoints(kwargs["kps"], coords, t, self.config["pck"]["type"]) for t in
                   PCK_THRESH}

            gridded_outputs = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
                predictions[0], nrow=10)), 0).transpose(1, 2, 3, 0)
            gridded_targets = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
                sure_to_torch(kwargs["targets"])[0], nrow=10)), 0).transpose(1, 2, 3, 0)

            logs = {
                "images": {
                    # Image input not needed, because stick animal is printed on input image
                    # "image_input": adjust_support(torch2numpy(inputs).transpose(0, 2, 3, 1), "-1->1"),
                    "first_pred": adjust_support(gridded_outputs, "-1->1", "0->1"),
                    "first_targets": adjust_support(gridded_targets, "-1->1", "0->1"),
                    "outputs": adjust_support(heatmaps_to_image(torch2numpy(predictions)).transpose(0, 2, 3, 1),
                                              "-1->1", "0->1"),
                    "targets": adjust_support(heatmaps_to_image(kwargs["targets"]).transpose(0, 2, 3, 1), "-1->1"),
                    "inputs_with_stick": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), kwargs["kps"]),
                    "stickanimal": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), predictions.clone(),
                                                    thresh=self.config["hm"]["thresh"]),
                },
                "scalars": {
                    "loss": losses["total"],
                    "learning_rate": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    f"PCK@{self.config['pck_alpha']}": np.around(pck[self.config['pck_alpha']][0], 5),
                },
                "figures": {
                    "Keypoint Mapping": plot_input_target_keypoints(torch2numpy(inputs).transpose(0, 2, 3, 1),
                                                                    # get BHWC
                                                                    torch2numpy(predictions),  # stay BCHW
                                                                    kwargs["kps"], coords),
                }
            }
            if self.config["losses"]["L2"]:
                logs["scalars"]["L2"] = losses["L2"]
            if self.config["losses"]["L1"]:
                logs["scalars"]["L1"] = losses["L1"]
            if self.config["losses"]["perceptual"]:
                logs["scalars"]["perceptual"] = losses["perceptual"]

            if self.config["pck"]["pck_multi"]:
                for key, val in pck.items():
                    # get mean value for pck at given threshold
                    logs["scalars"][f"PCK@{key}"] = np.around(val[0], 5)
                    for idx, part in enumerate(val[1]):
                        logs["scalars"][f"PCK@{key}_{self.dataset.get_idx_parts(idx)}"] = np.around(part, 5)
            return logs

        def eval_op():
            return #{
                #    "gt_kps": np.array(kwargs["kps"]),
                #    "predictions": np.array(predictions.cpu().detach().numpy()),
            #}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

class LordIteratorLatent(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # Initialize Loss functions
        self.mse_loss = torch.nn.MSELoss()
        # self.mse_instance = MSELossInstances()
        # self.l1_instance = L1LossInstances()
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        # Imagenet Mean
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        if self.cuda:
            self.model.cuda()

        self.freeze_encoder()

        # vgg loss
        if self.config["losses"]["vgg"]:
            self.vggL1 = VGGLossWithL1(gpu_ids=[0],
                                       l1_alpha=self.config["losses"]["vgg_l1_alpha"],
                                       vgg_alpha=self.config["losses"]["vgg_alpha"]).to(self.device)

        # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)

        # hooks
        # if self.config["adjust_learning_rate"]:
        #    self.hooks.append(
        #        AdjustLearningRate(self.config, self.optimizer)
        #    )

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def criterion(self, targets, predictions):
        # make sure everything is a torch tensor
        targets = sure_to_torch(targets)
        predictions = sure_to_torch(predictions)

        batch_losses = {}
        if self.config["losses"]["L2"]:
            batch_losses["L2"] = self.mse_loss(targets, predictions.cpu())
            batch_losses["L2"] += self.mse_loss(
                torch.from_numpy(heatmaps_to_image(targets.numpy())),
                torch.from_numpy(heatmaps_to_image(predictions.detach().cpu().numpy())))
        if self.config["losses"]["perceptual"]:
            batch_losses["perceptual"] = torch.mean(
                self.perceptual_loss(torch.from_numpy(heatmaps_to_image(targets.numpy())),
                                     torch.from_numpy(heatmaps_to_image(predictions.detach().cpu().numpy())),
                                     True)).cpu()
        batch_losses["total"] = sum(
            [
                batch_losses[key]
                for key in batch_losses.keys()
            ]
        )
        return batch_losses

    def freeze_encoder(self):
        self.logger.info("Encoder freezed!")
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.logger.info("Encoder unfreezed!")
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)
        if self.get_global_step() == self.config["freeze_encoder_for"] and is_train:
            self.unfreeze_encoder()

        # kwargs["inp"]
        # (batch_size, width, height, channel)
        inputs = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2))
        # (batch_size, channel, height, width)
        batch_size, channel, height, width = inputs.shape

        assets = AssetManager(args.base_dir)
        model_dir = assets.recreate_model_dir(args.model_name)
        tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

        data = np.load(assets.get_preprocess_file_path(args.data_name))
        imgs = data['imgs'].astype(np.float32) / 255.0

        config = dict(
            img_shape=imgs.shape[1:],
            n_imgs=imgs.shape[0],
            n_classes=data['n_classes'].item(),
        )

        config.update(base_config)

        lord = Lord(config)
        lord.train_latent(
            imgs=imgs,
            classes=data['classes'],

            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir
        )

        # compute loss
        losses = self.criterion(kwargs["targets"], predictions)

        def train_op():
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

        def log_op():

            PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]
            HM_THRESH = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if self.config['pck_alpha'] not in PCK_THRESH: PCK_THRESH.append(self.config["pck_alpha"])

            coords, _ = heatmaps_to_coords(predictions.clone(), thresh=self.config["hm"]["thresh"])
            pck = {t: percentage_correct_keypoints(kwargs["kps"], coords, t, self.config["pck"]["type"]) for t in
                   PCK_THRESH}

            gridded_outputs = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
                predictions[0], nrow=10)), 0).transpose(1, 2, 3, 0)
            gridded_targets = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
                sure_to_torch(kwargs["targets"])[0], nrow=10)), 0).transpose(1, 2, 3, 0)

            logs = {
                "images": {
                    # Image input not needed, because stick animal is printed on input image
                    # "image_input": adjust_support(torch2numpy(inputs).transpose(0, 2, 3, 1), "-1->1"),
                    "first_pred": adjust_support(gridded_outputs, "-1->1", "0->1"),
                    "first_targets": adjust_support(gridded_targets, "-1->1", "0->1"),
                    "outputs": adjust_support(heatmaps_to_image(torch2numpy(predictions)).transpose(0, 2, 3, 1),
                                              "-1->1", "0->1"),
                    "targets": adjust_support(heatmaps_to_image(kwargs["targets"]).transpose(0, 2, 3, 1), "-1->1"),
                    "inputs_with_stick": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), kwargs["kps"]),
                    "stickanimal": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), predictions.clone(),
                                                    thresh=self.config["hm"]["thresh"]),
                },
                "scalars": {
                    "loss": losses["total"],
                    "learning_rate": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    f"PCK@{self.config['pck_alpha']}": np.around(pck[self.config['pck_alpha']][0], 5),
                },
                "figures": {
                    "Keypoint Mapping": plot_input_target_keypoints(torch2numpy(inputs).transpose(0, 2, 3, 1),
                                                                    # get BHWC
                                                                    torch2numpy(predictions),  # stay BCHW
                                                                    kwargs["kps"], coords),
                }
            }
            if self.config["losses"]["L2"]:
                logs["scalars"]["L2"] = losses["L2"]
            if self.config["losses"]["L1"]:
                logs["scalars"]["L1"] = losses["L1"]
            if self.config["losses"]["perceptual"]:
                logs["scalars"]["perceptual"] = losses["perceptual"]

            if self.config["pck"]["pck_multi"]:
                for key, val in pck.items():
                    # get mean value for pck at given threshold
                    logs["scalars"][f"PCK@{key}"] = np.around(val[0], 5)
                    for idx, part in enumerate(val[1]):
                        logs["scalars"][f"PCK@{key}_{self.dataset.get_idx_parts(idx)}"] = np.around(part, 5)
            return logs

        def eval_op():
            return #{
                #    "gt_kps": np.array(kwargs["kps"]),
                #    "predictions": np.array(predictions.cpu().detach().numpy()),
            #}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
